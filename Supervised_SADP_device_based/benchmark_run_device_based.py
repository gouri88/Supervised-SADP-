"""
benchmark_run_device_based.py  (standalone — no other local imports required)
==============================================================================
Device-calibrated Supervised SADP benchmark runner.

ONLY TWO FILES NEEDED to run everything:
    benchmark_run_device_based.py   (this file)
    sadp_core_device_based.py       (all SADP logic + device kernel)
    conductance_device_data.xlsx    (your device data)

Quick start
-----------
    # Smoke test — runs in ~2 min to verify everything is wired up
    python benchmark_run_device_based.py --quick \\
        --device-data conductance_device_data.xlsx

    # Full benchmark (matches paper grid, 3 seeds)
    python benchmark_run_device_based.py \\
        --datasets mnist,fmnist,cifar10 \\
        --encodings poisson_only,lbp+poisson,lbp_clbp+poisson,cnn+poisson \\
        --k-shifts 5,25 --reward-modes none,binary,margin \\
        --timesteps 50 --epochs 50 --seeds 42,43,44 \\
        --device-data conductance_device_data.xlsx \\
        --output-dir results_device --output-prefix sadp_device_comparison

    # Medical datasets (requires --dataset-root)
    python benchmark_run_device_based.py \\
        --datasets colon,lung,tumour \\
        --dataset-root "C:/Users/user/Desktop/Gouri/Dataset" \\
        --encodings lbp+poisson,lbp_clbp+poisson,cnn+poisson \\
        --k-shifts 5,25 --reward-modes none,binary,margin \\
        --timesteps 50 --epochs 50 --seeds 42,43,44 \\
        --device-data conductance_device_data.xlsx \\
        --output-dir results_device_medical

    # Plot the fitted device kernel before running
    python benchmark_run_device_based.py --plot-kernel \\
        --device-data conductance_device_data.xlsx --quick

    # Normalised kernel (rescales output RMS=1, making eta_in comparable
    # to the standard SADP run for direct accuracy comparison)
    python benchmark_run_device_based.py --device-normalise ...

Notes
-----
* Results CSV has identical column structure to sadp_comparison.csv, with
  three extra columns: Device_Kernel_Path, Device_Kernel_Sheet, Device_Normalised.
  Direct pandas comparison with standard results is trivial.
* The feature cache is populated fresh each run (features are light to
  recompute for LBP; CNN features are the expensive part).
* Each failed configuration is logged and skipped; the grid continues.
* Results are checkpointed after every configuration.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(0)

# The one and only local import — sadp_core_device_based is self-contained
import sadp_core_device_based as sc


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Built-in: delegate to sc.load_dataset (requires TensorFlow for Keras data).
# Medical: PIL-based folder loader (no TensorFlow required).

_DATASET_ROOT: Path = Path(".")     # overridden by --dataset-root


def _load_builtin(name: str):
    """Load MNIST / FMNIST / CIFAR-10 via Keras/TensorFlow."""
    return sc.load_dataset(name)


def _load_folder_dataset(
    root: Path,
    target_size: Tuple[int, int] = (64, 64),
    test_split: float = 0.2,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray],
           Tuple[int, int, int]]:
    """
    Generic folder-based loader for medical datasets.

    Expected layout:
        root/
            class_A/   (contains *.jpg / *.png images)
            class_B/
            ...

    Images are resized to `target_size`, converted to RGB float32 in [0,1].
    Returns the same format as sc.load_dataset:
        (x_train, y_train), (x_test, y_test), input_shape
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for medical dataset loading. "
            "Install it with: pip install Pillow"
        ) from exc

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f"No subdirectories found in {root}. "
                         "Each class should have its own folder.")

    label_map   = {d.name: i for i, d in enumerate(class_dirs)}
    images, labels = [], []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    logger = logging.getLogger("sadp")
    logger.info("Loading images from %s | classes=%s | target_size=%s",
                root, list(label_map.keys()), target_size)

    for cls_dir in class_dirs:
        cls_label = label_map[cls_dir.name]
        img_paths = [p for p in cls_dir.iterdir()
                     if p.suffix.lower() in exts]
        logger.info("  %s: %d images", cls_dir.name, len(img_paths))
        for p in img_paths:
            try:
                img = Image.open(p).convert("RGB").resize(
                    target_size, Image.LANCZOS
                )
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(cls_label)
            except Exception as e:
                logger.warning("  Skipping %s: %s", p.name, e)

    if not images:
        raise ValueError(f"No valid images found under {root}.")

    x = np.stack(images)      # (N, H, W, 3)
    y = np.array(labels, dtype=np.int64)

    # Shuffle then split
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    split = int(len(x) * (1.0 - test_split))
    x_train, y_train = x[:split],  y[:split]
    x_test,  y_test  = x[split:],  y[split:]

    input_shape = x_train.shape[1:]   # (H, W, 3)
    logger.info("Dataset loaded | n_train=%d n_test=%d n_classes=%d",
                len(x_train), len(x_test), len(class_dirs))
    return (x_train, y_train), (x_test, y_test), input_shape


# Medical sub-folder names
_MEDICAL_SUBDIRS: Dict[str, str] = {
    "colon":  "colon_image_sets",
    "lung":   "lung_image_sets",
    "tumour": "Tumour_classification",
}


def _make_medical_loader(name: str) -> Callable:
    def loader():
        root = _DATASET_ROOT / _MEDICAL_SUBDIRS[name]
        if not root.exists():
            raise FileNotFoundError(
                f"Medical dataset folder not found: {root}\n"
                f"Pass the parent directory with --dataset-root."
            )
        return _load_folder_dataset(root)
    loader.__name__ = f"load_{name}"
    return loader


DATASET_LOADERS: Dict[str, Callable] = {
    "mnist":   lambda: _load_builtin("mnist"),
    "fmnist":  lambda: _load_builtin("fmnist"),
    "cifar10": lambda: _load_builtin("cifar10"),
    "colon":   _make_medical_loader("colon"),
    "lung":    _make_medical_loader("lung"),
    "tumour":  _make_medical_loader("tumour"),
}

DEFAULT_ENCODINGS = [
    "poisson_only", "lbp+poisson", "lbp_clbp+poisson", "cnn+poisson"
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _csv_int_list(s: str) -> List[int]:
    return [int(x) for x in _csv_list(s)]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Device-calibrated Supervised SADP benchmark. "
                    "Requires: sadp_core_device_based.py + conductance Excel file."
    )
    # ── Dataset / encoding
    p.add_argument("--datasets", type=_csv_list,
                   default=["mnist", "fmnist", "cifar10"],
                   help="Datasets: mnist fmnist cifar10 colon lung tumour. "
                        "Medical datasets need --dataset-root.")
    p.add_argument("--dataset-root", type=str, default=None,
                   help="Root folder containing medical dataset sub-folders "
                        "(colon_image_sets, lung_image_sets, Tumour_classification).")
    p.add_argument("--encodings", type=_csv_list, default=DEFAULT_ENCODINGS)
    p.add_argument("--architectures", type=_csv_list, default=["1SADP"])
    p.add_argument("--timesteps", type=_csv_int_list, default=[50],
                   help="Simulation timestep T. Default 50 (matches paper).")

    # ── SADP axes
    p.add_argument("--k-shifts", type=_csv_int_list, default=[5, 25],
                   help="K_shift values. Default {5,25} matches paper grid.")
    p.add_argument("--reward-modes", type=_csv_list,
                   default=["none", "binary", "margin"])
    p.add_argument("--weight-by-overlap", action="store_true")
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--reward-baseline", action="store_true")
    p.add_argument("--reward-baseline-decay", type=float, default=0.99)

    # ── Device kernel
    p.add_argument("--device-data", type=str,
                   default="conductance_device_data.xlsx",
                   help="Path to conductance Excel file. "
                        "Default: conductance_device_data.xlsx (same directory).")
    p.add_argument("--device-sheet", type=str, default="N=200",
                   help="Sheet name in the Excel file. Default: N=200.")
    p.add_argument("--device-col", type=str, default="Conductance",
                   help="Column name for conductance values.")
    p.add_argument("--device-normalise", action="store_true",
                   help="Normalise kernel output to RMS=1 so eta_in is "
                        "directly comparable to the standard SADP run.")
    p.add_argument("--spline-s-pot", type=float, default=0.1,
                   help="Potentiation spline smoothing. Default 0.1.")
    p.add_argument("--spline-s-dep", type=float, default=0.01,
                   help="Depression spline smoothing. Default 0.01.")
    p.add_argument("--plot-kernel", action="store_true",
                   help="Plot and save the fitted kernel to device_kernel.png "
                        "in --output-dir before running the grid.")

    # ── Training / infra
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--nhid", type=int, default=256)
    p.add_argument("--eta-in", type=float, default=2e-4)
    p.add_argument("--eta-out", type=float, default=5e-4)
    p.add_argument("--feature-dim", type=int, default=256)
    p.add_argument("--encoder-epochs", type=int, default=50)
    p.add_argument("--classical-grid", type=str, default="4,4")
    p.add_argument("--classical-bins", type=int, default=16)
    p.add_argument("--seeds", type=_csv_int_list, default=[42],
                   help="Random seeds. Use >=3 for statistically credible results. "
                        "Example: --seeds 42,43,44")
    p.add_argument("--output-dir", type=str, default="results_device")
    p.add_argument("--output-prefix", type=str, default="sadp_device_comparison")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke test: mnist, 2 encodings, 2 epochs, seed=42.")
    p.add_argument("--no-progress", action="store_true")
    return p


def apply_quick_overrides(args: argparse.Namespace) -> argparse.Namespace:
    args.datasets      = ["mnist"]
    args.encodings     = ["poisson_only", "lbp_clbp+poisson"]
    args.architectures = ["1SADP"]
    args.timesteps     = [25]
    args.k_shifts      = [5]
    args.reward_modes  = ["none"]
    args.epochs        = 2
    args.batch_size    = 64
    args.encoder_epochs = 2
    args.seeds         = [42]
    return args


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, args: argparse.Namespace) -> None:
    logger = logging.getLogger("sadp")
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path   = os.path.join(args.output_dir, f"{args.output_prefix}.csv")
    excel_path = os.path.join(args.output_dir, f"{args.output_prefix}.xlsx")

    df.to_csv(csv_path, index=False)
    logger.info("Saved results CSV -> %s", csv_path)

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All_Results", index=False)
            if "Encoding" in df.columns:
                for enc in sorted(df["Encoding"].dropna().unique()):
                    df[df["Encoding"] == enc].to_excel(
                        writer, sheet_name=enc.replace("+", "_")[:31], index=False
                    )
            if {"Dataset","Encoding","K_shift","Reward_Mode","Eval_Accuracy"} <= set(df.columns):
                ok = df[df.get("Status", pd.Series(["OK"]*len(df))) == "OK"]
                for dname in sorted(ok["Dataset"].dropna().unique()):
                    pivot = ok[ok["Dataset"]==dname].pivot_table(
                        index="Encoding",
                        columns=["K_shift","Reward_Mode"],
                        values="Eval_Accuracy",
                        aggfunc=["mean","std","count"],
                    )
                    pivot.to_excel(writer, sheet_name=f"Pivot_{dname}"[:31])
        logger.info("Saved results Excel -> %s", excel_path)
    except Exception as exc:
        logger.warning("Could not write Excel (%s); CSV was saved.", exc)


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_grid(args: argparse.Namespace,
             device_kernel: "sc.DeviceKernel") -> pd.DataFrame:
    logger         = logging.getLogger("sadp")
    classical_grid = tuple(int(x) for x in args.classical_grid.split(","))
    show_progress  = not args.no_progress

    for dname in args.datasets:
        if dname not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset '{dname}'. "
                             f"Available: {list(DATASET_LOADERS.keys())}")

    grid: List[Dict[str, Any]] = [
        {"dataset_name": ds, "encoding_type": enc, "seed": seed,
         "architecture": arch, "T": T_val, "k_shift": k,
         "reward_mode": rm}
        for ds   in args.datasets
        for enc  in args.encodings
        for seed in args.seeds
        for arch in args.architectures
        for T_val in args.timesteps
        for k    in args.k_shifts
        for rm   in args.reward_modes
    ]

    logger.info("Device grid: %d configurations | kernel=%s",
                len(grid), repr(device_kernel))

    results: List[Dict[str, Any]] = []
    checkpoint = os.path.join(args.output_dir,
                              f"{args.output_prefix}_checkpoint.csv")

    dataset_cache: Dict[str, Any] = {}
    feature_cache: Dict[Any, Any] = {}

    pbar = tqdm(grid, desc="Device grid", disable=not show_progress)
    for cfg in pbar:
        tag = (f"{cfg['dataset_name']}|{cfg['encoding_type']}|"
               f"seed={cfg['seed']}|{cfg['architecture']}|"
               f"T={cfg['T']}|k={cfg['k_shift']}|{cfg['reward_mode']}")
        pbar.set_postfix_str(tag)
        logger.info("-" * 60)
        logger.info("[device] Starting: %s", tag)
        t0 = time.time()

        try:
            if cfg["dataset_name"] not in dataset_cache:
                dataset_cache[cfg["dataset_name"]] = \
                    DATASET_LOADERS[cfg["dataset_name"]]()
            data = dataset_cache[cfg["dataset_name"]]

            result = sc.run_experiment_device(
                device_kernel=device_kernel,
                dataset_name=cfg["dataset_name"],
                data=data,
                encoding_type=cfg["encoding_type"],
                architecture=cfg["architecture"],
                T=cfg["T"],
                k_shift=cfg["k_shift"],
                weight_by_overlap=args.weight_by_overlap,
                reward_mode=cfg["reward_mode"],
                reward_scale=args.reward_scale,
                reward_baseline=args.reward_baseline,
                reward_baseline_decay=args.reward_baseline_decay,
                Nhid=args.nhid,
                eta_in=args.eta_in,
                eta_out=args.eta_out,
                seed=cfg["seed"],
                feature_dim=args.feature_dim,
                encoder_epochs=args.encoder_epochs,
                classical_grid=classical_grid,
                classical_n_bins=args.classical_bins,
                n_epochs=args.epochs,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                eval_every=args.eval_every,
                show_progress=show_progress,
                feature_cache=feature_cache,
            )
            result["Config_Wall_Time_s"] = time.time() - t0
            result["Status"] = "OK"
            results.append(result)

        except Exception as exc:
            logger.error("[device] FAILED: %s | %s", tag, exc)
            logger.debug(traceback.format_exc())
            results.append({
                "Dataset": cfg["dataset_name"].upper(),
                "Encoding": cfg["encoding_type"],
                "Seed": cfg["seed"],
                "Architecture": cfg["architecture"],
                "Timestep": cfg["T"],
                "K_shift": cfg["k_shift"],
                "Reward_Mode": cfg["reward_mode"],
                "Status": f"FAILED: {exc}",
                "Config_Wall_Time_s": time.time() - t0,
            })

        pd.DataFrame(results).to_csv(checkpoint, index=False)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    if args.quick:
        args = apply_quick_overrides(args)

    # Apply dataset-root override for medical loaders
    global _DATASET_ROOT
    if args.dataset_root is not None:
        _DATASET_ROOT = Path(args.dataset_root)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(args.output_dir,
                              f"{args.output_prefix}_{timestamp}.log")
    sc.configure_logging(log_file=log_path, level=logging.INFO)
    logger = logging.getLogger("sadp")

    # ── Initialise device kernel (once — shared across entire grid)
    logger.info("Initialising DeviceKernel | path=%s sheet=%s normalise=%s",
                args.device_data, args.device_sheet, args.device_normalise)
    device_kernel = sc.DeviceKernel(
        data_path=args.device_data,
        sheet=args.device_sheet,
        conductance_col=args.device_col,
        spline_s_pot=args.spline_s_pot,
        spline_s_dep=args.spline_s_dep,
        normalise=args.device_normalise,
    )

    if args.plot_kernel:
        plot_path = os.path.join(args.output_dir, "device_kernel.png")
        device_kernel.plot(save_path=plot_path, show=False)
        logger.info("Kernel plot saved -> %s", plot_path)

    # ── Pre-flight check for medical dataset roots
    medical_needed = [d for d in args.datasets if d in _MEDICAL_SUBDIRS]
    if medical_needed:
        for d in medical_needed:
            folder = _DATASET_ROOT / _MEDICAL_SUBDIRS[d]
            if not folder.exists():
                raise SystemExit(
                    f"\n[ERROR] Medical dataset folder not found: {folder}\n"
                    f"Pass the correct parent with --dataset-root.\n"
                )

    n_grid = (len(args.datasets) * len(args.encodings) * len(args.seeds)
              * len(args.architectures) * len(args.timesteps)
              * len(args.k_shifts) * len(args.reward_modes))

    logger.info("=" * 70)
    logger.info("Device-calibrated Supervised SADP benchmark starting.")
    logger.info("Kernel      : %s", repr(device_kernel))
    logger.info("Datasets    : %s", args.datasets)
    logger.info("Encodings   : %s", args.encodings)
    logger.info("Seeds       : %s", args.seeds)
    logger.info("K-shifts    : %s", args.k_shifts)
    logger.info("Reward modes: %s", args.reward_modes)
    logger.info("Timestep T  : %s | Epochs: %d", args.timesteps, args.epochs)
    logger.info("Grid size   : %d configurations", n_grid)
    logger.info("Output dir  : %s", args.output_dir)
    logger.info("=" * 70)

    t0 = time.time()
    df = run_grid(args, device_kernel)
    total_time = time.time() - t0

    n_ok = int((df.get("Status", pd.Series(["OK"]*len(df))) == "OK").sum())
    logger.info("Grid complete in %.1fs | %d/%d OK.", total_time, n_ok, len(df))

    save_results(df, args)

    # ── Console summary
    print("\n" + "=" * 70)
    print("DEVICE-CALIBRATED SADP — SUMMARY")
    print(f"Kernel: {repr(device_kernel)}")
    print("=" * 70)
    cols = [c for c in [
        "Dataset","Encoding","K_shift","Reward_Mode","Seed",
        "Eval_Accuracy","F1_score","Avg_Time_per_Epoch_s",
        "Device_Normalised","Status",
    ] if c in df.columns]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df[cols].to_string(index=False))

    if {"Dataset","Encoding","K_shift","Reward_Mode","Eval_Accuracy"} <= set(df.columns):
        ok = df[df.get("Status", pd.Series(["OK"]*len(df))) == "OK"]
        for dname in sorted(ok["Dataset"].dropna().unique()):
            sub   = ok[ok["Dataset"]==dname]
            stats = (sub.groupby(["Encoding","K_shift","Reward_Mode"])
                        ["Eval_Accuracy"]
                        .agg(["mean","std","count"]).reset_index())
            n_s   = int(stats["count"].max())
            pivot = stats.pivot(index="Encoding",
                                columns=["K_shift","Reward_Mode"],
                                values="mean")
            print(f"\n--- {dname}: mean Eval_Accuracy "
                  f"(n<={n_s} seed{'s' if n_s!=1 else ''}) ---")
            with pd.option_context("display.width", 200, "display.precision", 4):
                print(pivot)
            if n_s > 1:
                pstd = stats.pivot(index="Encoding",
                                   columns=["K_shift","Reward_Mode"],
                                   values="std")
                print(f"--- {dname}: std ---")
                with pd.option_context("display.width", 200, "display.precision", 4):
                    print(pstd)


if __name__ == "__main__":
    main()
