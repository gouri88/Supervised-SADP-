"""
benchmark_run_stdp.py
======================

Focused comparison driver for (reward-modulated) STDP - the STDP
counterpart to benchmark_run.py (which drives SADP). Same scope:

    4 encodings   x  tau values  x  reward modes
    (poisson_only, lbp+poisson, lbp_clbp+poisson, cnn+poisson)

on the three standard vision datasets (MNIST / Fashion-MNIST / CIFAR-10),
1STDP only, so the output is a single, directly-comparable results table.
The only thing that differs from benchmark_run.py is the learning-rule
axis: SADP's `k_shift` (window width for the shifted-kappa agreement rule)
is replaced by STDP's own `tau_plus`/`tau_minus` (eligibility-trace time
constants) - everything else (encodings, dataset handling, reward
modulation, CLI shape, output format) is the same on purpose, so the two
scripts' results tables line up for a direct comparison.

This script only depends on `stdp_core.py` (must be importable - keep both
files in the same directory, or put `stdp_core.py` on your PYTHONPATH).

-----------------------------------------------------------------------
Quick start
-----------------------------------------------------------------------
    # Fast smoke test (tiny epoch count, MNIST only) to check everything
    # is wired up correctly before committing to a long run:
    python benchmark_run_stdp.py --quick

    # The full comparison: all 4 encodings, tau in {2, 5, 10}, reward_mode
    # in {none, binary, margin}, on all three standard datasets (this is
    # also just running with no flags at all - these are the defaults):
    python benchmark_run_stdp.py \
        --datasets mnist,fmnist,cifar10 \
        --encodings poisson_only,lbp+poisson,lbp_clbp+poisson,cnn+poisson \
        --taus 2,5,10 \
        --reward-modes none,binary,margin \
        --epochs 50

    # Skip the CNN pathway (no TensorFlow installed / not needed):
    python benchmark_run_stdp.py --encodings poisson_only,lbp+poisson,lbp_clbp+poisson

    # Just the two axes of interest, holding encoding fixed:
    python benchmark_run_stdp.py --encodings lbp+poisson --taus 2,5,10 \
        --reward-modes none,binary,margin

    # Compare plain LBP against Complete LBP (CLBP) directly:
    python benchmark_run_stdp.py --encodings lbp+poisson,lbp_clbp+poisson

    # Revisit 2STDP later if you want it (not run by default):
    python benchmark_run_stdp.py --architectures 1STDP,2STDP

    # Tune potentiation/depression amplitudes (single values, not swept):
    python benchmark_run_stdp.py --a-plus 1.0 --a-minus 1.2

    # For a statistically credible comparison rather than a single noisy
    # run, sweep multiple seeds (results aggregate as mean+-std per cell):
    python benchmark_run_stdp.py --encodings lbp+poisson --seeds 42,43,44,45,46

-----------------------------------------------------------------------
A note on comparing this against benchmark_run.py's (SADP) results
-----------------------------------------------------------------------
With reward_mode='none', STDP's hidden layer is genuinely UNSUPERVISED -
the label only reaches the network through the (unchanged) output-layer
delta rule. SADP's k_shift=0 case is still label-dependent at the hidden
layer even with no extra supervision. So "reward_mode='none'" does not
mean the same thing in both result tables - keep that in mind when lining
up rows. Also: the `eta_in` default was tuned for SADP's bounded
kappa-agreement signal and is not validated for STDP's differently-scaled
trace-based signal - expect to retune `--nhid`/learning-rate-adjacent
settings once you have real numbers, rather than assuming defaults
transfer directly (see stdp_core.py's module docstring for more).

-----------------------------------------------------------------------
Adding a new dataset
-----------------------------------------------------------------------
Add one entry to the DATASET_LOADERS registry below - a zero-argument
function returning the same ((x_train,y_train),(x_test,y_test),input_shape)
shape that `stdp_core.load_dataset` returns. Nothing else in this file, or
in stdp_core.py, needs to change. For example:

    def _load_my_dataset():
        x_train, y_train, x_test, y_test = ...   # your own loading code
        # x in [0,1] float32, shape (N,H,W) or (N,H,W,C); y integer ids
        return (x_train, y_train), (x_test, y_test), x_train.shape[1:]

    DATASET_LOADERS["my_dataset"] = _load_my_dataset

Then just: python benchmark_run_stdp.py --datasets my_dataset

Every run is wrapped in a try/except: if one configuration fails (e.g. an
unsupported combination, OOM, etc.) it is logged and skipped, and the grid
continues - a single bad cell never kills a multi-hour benchmarking run.

Results so far are also checkpointed to disk after every configuration, so
an interrupted run never loses completed results.
-----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(0)

import stdp_core as sc


# ---------------------------------------------------------------------------
# Dataset registry - the one place to touch for a new dataset
# ---------------------------------------------------------------------------
# Each entry maps a name to a zero-argument loader returning exactly what
# `sc.run_experiment`'s `data=` parameter expects:
#     (x_train, y_train), (x_test, y_test), input_shape
# with x in [0,1] float32, shape (N,H,W) or (N,H,W,C), y integer class ids.
# The three built-ins just delegate to `sc.load_dataset` (which needs
# TensorFlow, for the Keras dataset download). To add your own dataset,
# write a loader function and add one line here - see the module
# docstring above for a worked example.

DATASET_LOADERS: Dict[str, Callable[[], Any]] = {
    "mnist": lambda: sc.load_dataset("mnist"),
    "fmnist": lambda: sc.load_dataset("fmnist"),
    "cifar10": lambda: sc.load_dataset("cifar10"),
}

# The four encodings this script is scoped to compare - identical set to
# benchmark_run.py's, so encoding effects aren't confounded with the
# learning-rule comparison.
DEFAULT_ENCODINGS = ["poisson_only", "lbp+poisson", "lbp_clbp+poisson", "cnn+poisson"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _csv_int_list(s: str) -> List[int]:
    return [int(x) for x in _csv_list(s)]


def _csv_float_list(s: str) -> List[float]:
    return [float(x) for x in _csv_list(s)]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare (reward-modulated) STDP across 4 encodings "
                    "(poisson_only, lbp+poisson, lbp_clbp+poisson, cnn+poisson) "
                    "x tau x reward-mode, on MNIST/FMNIST/CIFAR-10."
    )
    p.add_argument("--datasets", type=_csv_list,
                    default=list(DATASET_LOADERS.keys()),
                    help="Comma-separated dataset names. Built-in: "
                        + ", ".join(DATASET_LOADERS.keys())
                        + ". Add more to DATASET_LOADERS in this file.")
    p.add_argument("--encodings", type=_csv_list, default=DEFAULT_ENCODINGS,
                    help="Comma-separated encoding types. Available: "
                        + ", ".join(sc.ENCODING_TYPES))
    p.add_argument("--architectures", type=_csv_list, default=["1STDP"],
                    help="Comma-separated architectures: 1STDP, 2STDP. "
                        "Defaults to 1STDP only; pass --architectures "
                        "1STDP,2STDP to bring 2STDP back into the sweep.")
    p.add_argument("--timesteps", type=_csv_int_list, default=[25],
                    help="Comma-separated timestep (T) values, e.g. 25,100. "
                        "Default is just 25 to keep the 6-dimensional grid "
                        "(dataset x encoding x arch x T x tau x reward_mode) "
                        "a manageable size; add 100 for the fuller sweep.")

    # ---- The two supervision axes this script compares ----
    p.add_argument("--taus", type=_csv_float_list, default=[2.0, 5.0, 10.0],
                    help="Comma-separated tau values, e.g. 2,5,10. Each value "
                        "sets BOTH tau_plus and tau_minus (the STDP "
                        "eligibility-trace time constants, in timesteps) - "
                        "the STDP counterpart to SADP's --k-shifts sweep. "
                        "For independently-tuned tau_plus/tau_minus, call "
                        "sc.run_experiment(...) directly from Python instead "
                        "of going through this CLI.")
    p.add_argument("--reward-modes", type=_csv_list, default=["none", "binary", "margin"],
                    help="Comma-separated reward_mode values to sweep: "
                        "none, binary, margin. 'none' is plain, unsupervised "
                        "STDP at the hidden layer (the label only reaches the "
                        "network through the output layer); 'binary'/'margin' "
                        "turn this into reward-modulated STDP (R-STDP).")
    p.add_argument("--a-plus", type=float, default=1.0,
                    help="STDP potentiation amplitude. Single value, not "
                        "swept (applies to every tau/reward_mode above).")
    p.add_argument("--a-minus", type=float, default=1.0,
                    help="STDP depression amplitude. Single value, not "
                        "swept. Try slightly > --a-plus for the classic "
                        "mild-depression-bias variant.")
    p.add_argument("--reward-scale", type=float, default=1.0,
                    help="Multiplier applied to the reward signal before it "
                        "scales the STDP eligibility. Applies to every "
                        "reward_mode in the sweep above (has no effect when "
                        "reward_mode='none').")
    p.add_argument("--reward-baseline", action="store_true",
                    help="Subtract a running EMA of the batch-mean reward "
                        "before using it (REINFORCE-style advantage). Applies "
                        "to every reward_mode in the sweep above.")
    p.add_argument("--reward-baseline-decay", type=float, default=0.99,
                    help="EMA decay for --reward-baseline (closer to 1 = smoother).")

    # ---- Training / infra ----
    p.add_argument("--epochs", type=int, default=50, help="SNN training epochs per configuration.")
    p.add_argument("--batch-size", type=int, default=128, help="SNN training batch size.")
    p.add_argument("--eval-batch-size", type=int, default=256, help="Evaluation batch size.")
    p.add_argument("--eval-every", type=int, default=None,
                    help="Evaluate on held-out data every N epochs during "
                        "training (in addition to the final eval), and "
                        "record it as Epoch_Eval_Accuracies. Off by default "
                        "since it costs an extra eval pass per N epochs - "
                        "set e.g. --eval-every 1 for every epoch (most "
                        "informative, most expensive) or --eval-every 5 for "
                        "a cheaper periodic check. Weight-stability "
                        "diagnostics (Epoch_W1_CosSim_Prev, Epoch_W1_FrobNorm, "
                        "Epoch_W2_FrobNorm) are always recorded regardless - "
                        "they're effectively free.")
    p.add_argument("--nhid", type=int, default=256, help="Hidden layer size.")
    p.add_argument("--eta-in", type=float, default=2e-4,
                    help="Hidden-layer (STDP) learning rate. NOTE: this "
                        "default was tuned for SADP's bounded kappa signal, "
                        "not for STDP's differently-scaled trace signal - "
                        "measured ratio is roughly 60-90x depending on tau (STDP's raw update is "
                        "much larger at matched defaults). Use "
                        "stdp_core.calibrate_eta_in() to get a value whose "
                        "effective step size matches a given SADP run before "
                        "treating a head-to-head comparison as controlled.")
    p.add_argument("--eta-out", type=float, default=5e-4,
                    help="Output-layer learning rate (shared mechanism with SADP).")
    p.add_argument("--feature-dim", type=int, default=256, help="CNN encoder output dim (cnn+poisson only).")
    p.add_argument("--encoder-epochs", type=int, default=50, help="CNN pretraining epochs (cnn+poisson only).")
    p.add_argument("--classical-grid", type=str, default="4,4",
                    help="Grid rows,cols for the LBP/CLBP spatial blocks, e.g. '4,4'.")
    p.add_argument("--classical-bins", type=int, default=16,
                    help="Histogram bins per block, used by both 'lbp+poisson' "
                        "and 'lbp_clbp+poisson'.")
    p.add_argument("--seeds", type=_csv_int_list, default=[42],
                    help="Comma-separated random seeds, e.g. 42,43,44. Default "
                        "is a single seed (matches prior behavior); for any "
                        "claim like 'X beats Y' to be statistically credible, "
                        "use >=3 seeds and look at mean+-std in the pivot "
                        "tables, not a single run. Affects weight init, batch "
                        "order, AND (for cnn+poisson) feature-cache identity - "
                        "so each seed re-trains the CNN encoder too, not just "
                        "the SNN; this is intentional, not a cache miss bug.")
    p.add_argument("--output-dir", type=str, default=".", help="Directory for results/log files.")
    p.add_argument("--output-prefix", type=str, default="stdp_comparison",
                    help="Filename prefix for the results Excel/CSV and the log file.")
    p.add_argument("--quick", action="store_true",
                    help="Fast smoke-test mode: mnist only, poisson_only + "
                        "lbp_clbp+poisson, 1STDP only, T=25, tau in {2,5}, "
                        "reward_mode in {none,binary}, 2 epochs, small batch. "
                        "Use this first to verify the pipeline runs end to "
                        "end on your machine.")
    p.add_argument("--no-progress", action="store_true",
                    help="Disable tqdm progress bars (e.g. for non-interactive logs).")
    return p


def apply_quick_overrides(args: argparse.Namespace) -> argparse.Namespace:
    args.datasets = ["mnist"]
    args.encodings = ["poisson_only", "lbp_clbp+poisson"]
    args.architectures = ["1STDP"]
    args.timesteps = [25]
    args.taus = [2.0, 5.0]
    args.reward_modes = ["none", "binary"]
    args.epochs = 2
    args.batch_size = 64
    args.encoder_epochs = 2
    logging.getLogger("stdp").info("`--quick` mode: overriding grid to a fast smoke test.")
    return args


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_grid(args: argparse.Namespace) -> pd.DataFrame:
    logger = logging.getLogger("stdp")
    classical_grid = tuple(int(x) for x in args.classical_grid.split(","))
    show_progress = not args.no_progress

    for dname in args.datasets:
        if dname not in DATASET_LOADERS:
            raise ValueError(
                f"Unknown dataset '{dname}'. Available: {list(DATASET_LOADERS.keys())}. "
                "Add new datasets to DATASET_LOADERS at the top of this file."
            )

    # Loop order matters for cache efficiency: group everything that shares
    # the same (dataset, encoding, seed) triple together, since the encoded
    # features (and, for cnn+poisson, the CNN training that produces them)
    # depend on dataset/encoding/seed but NOT on architecture, T, tau, or
    # reward_mode. seed sits right after encoding (not innermost) because it
    # DOES invalidate the feature cache (cnn+poisson's CNN encoder is itself
    # seed-dependent) - architecture/T/tau/reward_mode are swept innermost
    # since they're the cheapest knobs to vary (pure training-time effects,
    # no re-extraction at all).
    grid: List[Dict[str, Any]] = []
    for dname in args.datasets:
        for encoding in args.encodings:
            for seed in args.seeds:
                for arch in args.architectures:
                    for T_val in args.timesteps:
                        for tau in args.taus:
                            for reward_mode in args.reward_modes:
                                grid.append({
                                    "dataset_name": dname, "encoding_type": encoding,
                                    "seed": seed, "architecture": arch, "T": T_val,
                                    "tau": tau, "reward_mode": reward_mode,
                                })

    logger.info("Comparison grid has %d configurations.", len(grid))
    logger.info("Features (and any CNN encoder training) are cached per "
                "(dataset, encoding, seed) triple and reused across "
                "architecture/T/tau/reward_mode.")

    results: List[Dict[str, Any]] = []
    checkpoint_path = os.path.join(args.output_dir, f"{args.output_prefix}_checkpoint.csv")

    dataset_cache: Dict[str, Any] = {}
    feature_cache: Dict[Any, Any] = {}

    pbar = tqdm(grid, desc="Comparison grid", disable=not show_progress)
    for cfg in pbar:
        tag = (f"{cfg['dataset_name']}|{cfg['encoding_type']}|seed={cfg['seed']}|"
               f"{cfg['architecture']}|T={cfg['T']}|tau={cfg['tau']}|reward={cfg['reward_mode']}")
        pbar.set_postfix_str(tag)
        logger.info("-" * 70)
        logger.info("Starting configuration: %s", tag)

        t_cfg0 = time.time()
        try:
            if cfg["dataset_name"] not in dataset_cache:
                dataset_cache[cfg["dataset_name"]] = DATASET_LOADERS[cfg["dataset_name"]]()
            data = dataset_cache[cfg["dataset_name"]]

            result = sc.run_experiment(
                dataset_name=cfg["dataset_name"], data=data,
                encoding_type=cfg["encoding_type"], architecture=cfg["architecture"],
                T=cfg["T"], tau_plus=cfg["tau"], tau_minus=cfg["tau"],
                a_plus=args.a_plus, a_minus=args.a_minus,
                reward_mode=cfg["reward_mode"], reward_scale=args.reward_scale,
                reward_baseline=args.reward_baseline,
                reward_baseline_decay=args.reward_baseline_decay,
                Nhid=args.nhid, eta_in=args.eta_in, eta_out=args.eta_out,
                seed=cfg["seed"], feature_dim=args.feature_dim,
                encoder_epochs=args.encoder_epochs, classical_grid=classical_grid,
                classical_n_bins=args.classical_bins,
                n_epochs=args.epochs, batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size, eval_every=args.eval_every,
                show_progress=show_progress, feature_cache=feature_cache,
            )
            result["Config_Wall_Time_s"] = time.time() - t_cfg0
            result["Status"] = "OK"
            results.append(result)

        except Exception as exc:  # noqa: BLE001 - keep the grid alive
            logger.error("Configuration FAILED: %s | %s", tag, exc)
            logger.debug(traceback.format_exc())
            results.append({
                "Dataset": cfg["dataset_name"].upper(), "Encoding": cfg["encoding_type"],
                "Seed": cfg["seed"], "Architecture": cfg["architecture"], "Timestep": cfg["T"],
                "Tau_Plus": cfg["tau"], "Tau_Minus": cfg["tau"],
                "Reward_Mode": cfg["reward_mode"],
                "Status": f"FAILED: {exc}",
                "Config_Wall_Time_s": time.time() - t_cfg0,
            })

        # Checkpoint after every configuration so partial progress is never lost.
        pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, args: argparse.Namespace) -> None:
    logger = logging.getLogger("stdp")
    os.makedirs(args.output_dir, exist_ok=True)
    excel_path = os.path.join(args.output_dir, f"{args.output_prefix}.xlsx")
    csv_path = os.path.join(args.output_dir, f"{args.output_prefix}.csv")

    df.to_csv(csv_path, index=False)
    logger.info("Saved full results CSV -> %s", csv_path)

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All_Results", index=False)
            if "Encoding" in df.columns:
                for encoding in sorted(df["Encoding"].dropna().unique()):
                    sheet_name = encoding.replace("+", "_")[:31]  # Excel sheet name limit
                    sub = df[df["Encoding"] == encoding]
                    sub.to_excel(writer, sheet_name=sheet_name, index=False)
            # One pivot per dataset: Encoding x (Tau_Plus, Reward_Mode) -> accuracy
            # mean/std/count across seeds (averaged over architecture/T too) -
            # the at-a-glance comparison view, with the seed count visible so
            # a single-seed result isn't mistaken for a statistically
            # supported one.
            if {"Dataset", "Encoding", "Tau_Plus", "Reward_Mode", "Eval_Accuracy"} <= set(df.columns):
                ok = df[df.get("Status", "OK") == "OK"]
                for dname in sorted(ok["Dataset"].dropna().unique()):
                    sub = ok[ok["Dataset"] == dname]
                    pivot = sub.pivot_table(
                        index="Encoding", columns=["Tau_Plus", "Reward_Mode"],
                        values="Eval_Accuracy", aggfunc=["mean", "std", "count"],
                    )
                    sheet_name = f"Pivot_{dname}"[:31]
                    pivot.to_excel(writer, sheet_name=sheet_name)
        logger.info("Saved results Excel workbook -> %s", excel_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write Excel file (%s); CSV was still saved.", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    if args.quick:
        args = apply_quick_overrides(args)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.output_dir, f"{args.output_prefix}_{timestamp}.log")
    sc.configure_logging(log_file=log_path, level=logging.INFO)
    logger = logging.getLogger("stdp")

    n_grid = (len(args.datasets) * len(args.encodings) * len(args.seeds)
              * len(args.architectures) * len(args.timesteps)
              * len(args.taus) * len(args.reward_modes))

    logger.info("=" * 70)
    logger.info("(Reward-modulated) STDP comparison run starting.")
    logger.info("Datasets:      %s", args.datasets)
    logger.info("Encodings:     %s", args.encodings)
    logger.info("Seeds:         %s%s", args.seeds,
                "  (single seed - add more for statistically credible comparisons)"
                if len(args.seeds) == 1 else "")
    logger.info("Architectures: %s", args.architectures)
    logger.info("Timesteps:     %s", args.timesteps)
    logger.info("Taus:          %s (a_plus=%s, a_minus=%s)",
                args.taus, args.a_plus, args.a_minus)
    logger.info("Reward modes:  %s (scale=%s, baseline=%s)",
                args.reward_modes, args.reward_scale, args.reward_baseline)
    logger.info("Epochs:        %s", args.epochs)
    logger.info("Eta_in/out:    %s / %s", args.eta_in, args.eta_out)
    logger.info("Grid size:     %d configurations", n_grid)
    logger.info("Log file:      %s", log_path)
    logger.info("=" * 70)

    t0 = time.time()
    df = run_grid(args)
    total_time = time.time() - t0

    n_ok = int((df["Status"] == "OK").sum()) if "Status" in df.columns else len(df)
    n_total = len(df)
    logger.info("Grid complete in %.1fs | %d/%d configurations succeeded.",
                total_time, n_ok, n_total)

    save_results(df, args)

    print("\n" + "=" * 70)
    print("(REWARD-MODULATED) STDP COMPARISON SUMMARY")
    print("=" * 70)
    display_cols = [c for c in [
        "Dataset", "Encoding", "Architecture", "Timestep", "Tau_Plus", "Tau_Minus",
        "Reward_Mode", "Seed", "Eval_Accuracy", "F1_score", "Avg_Time_per_Epoch_s", "Status",
    ] if c in df.columns]
    with pd.option_context("display.max_rows", None, "display.width", 180):
        print(df[display_cols].to_string(index=False))

    # Quick at-a-glance pivot in the console too: mean (and, if >1 seed was
    # run, std dev across seeds) Eval_Accuracy per dataset, encoding x
    # (tau, reward_mode) - in addition to the full table above.
    if {"Dataset", "Encoding", "Tau_Plus", "Reward_Mode", "Eval_Accuracy"} <= set(df.columns):
        ok = df[df.get("Status", "OK") == "OK"]
        for dname in sorted(ok["Dataset"].dropna().unique()):
            sub = ok[ok["Dataset"] == dname]
            if sub.empty:
                continue
            stats = (sub.groupby(["Encoding", "Tau_Plus", "Reward_Mode"])["Eval_Accuracy"]
                      .agg(["mean", "std", "count"]).reset_index())
            n_seeds_seen = int(stats["count"].max())
            pivot_mean = stats.pivot(index="Encoding", columns=["Tau_Plus", "Reward_Mode"], values="mean")
            print(f"\n--- {dname}: mean Eval_Accuracy by Encoding x (Tau, Reward_Mode) "
                  f"(n<={n_seeds_seen} seed{'s' if n_seeds_seen != 1 else ''}) ---")
            with pd.option_context("display.width", 180, "display.precision", 4):
                print(pivot_mean)
            if n_seeds_seen > 1:
                pivot_std = stats.pivot(index="Encoding", columns=["Tau_Plus", "Reward_Mode"], values="std")
                print(f"--- {dname}: std dev across seeds ---")
                with pd.option_context("display.width", 180, "display.precision", 4):
                    print(pivot_std)
            else:
                print("(single seed - std dev not meaningful; pass --seeds with "
                      ">=3 values for a real comparison)")


if __name__ == "__main__":
    main()
