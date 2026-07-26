"""
stdp_core.py
============

Core, reusable building blocks for (reward-modulated) Spike-Timing-Dependent
Plasticity (STDP / R-STDP) experiments - the STDP counterpart to
sadp_core.py. Same encodings, same dataset handling, same reward-modulation
machinery; only the hidden-layer learning rule differs (STDP instead of
SADP's k-shifted kappa-agreement rule). See "Notes on the learning rule"
below for exactly what that means.

This module is intentionally self-contained and import-friendly: every
public function/class takes plain numpy arrays in and returns plain numpy
arrays / dataclasses out, with no hidden module-level state. That means you
can drop in a brand-new dataset (any (N, H, W) or (N, H, W, C) array
normalized to [0, 1], plus integer labels) and reuse every piece of the
pipeline - feature extraction, spike encoding, the STDP network, training
and evaluation - without touching this file.

Layout
------
1. Logging setup
2. Dataset loading helpers (MNIST / Fashion-MNIST / CIFAR-10 via Keras)
3. Temporal spike encoders (Poisson)
4. Classical (NON-neural-network) image feature extractors:
       - Single-scale LBP (the original descriptor)
       - Complete LBP (CLBP) - sign + magnitude + center decomposition
       - Simple per-block color moments (for color datasets)
       - A unified dispatcher `extract_classical_features`
5. Optional CNN encoder (kept from the original pipeline, for comparison)
6. The (reward-modulated) STDP network itself (`STDPConfig`, `STDPNetwork`)
7. Train / evaluate loops
8. High-level `run_experiment` orchestration used by the benchmarking script

Notes on encoding_type
-----------------------
Four encoding options are available - identical to sadp_core.py:

  'poisson_only'      - Raw pixel values -> Poisson spike trains. Weak but
                        fast; the lower-bound baseline.
  'lbp+poisson'        - Classic Local Binary Pattern (P=8, R=1): each pixel
                        gets an 8-bit code from s(g_p - g_c)*2^p; block-wise
                        histograms become the feature vector. Zero
                        trainable parameters.
  'lbp_clbp+poisson'   - Complete LBP (CLBP): the same 8-neighbor ring,
                        decomposed into three components - sign (= plain
                        LBP), magnitude (thresholded |g_p - g_c|), and
                        center (pixel vs. the image's global mean) - whose
                        three histograms are concatenated. Captures more of
                        the local texture than plain LBP (how much
                        brighter, not just which neighbors are brighter)
                        for roughly 2x the features. Still zero trainable
                        parameters.
  'cnn+poisson'        - Pre-trained CNN features -> Poisson spike trains.
                        Highest accuracy ceiling but introduces learned
                        parameters before the spiking layer (kept for
                        comparison, since it's exactly what Reviewer #2's
                        objection was about).

Notes on the learning rule: STDP instead of SADP
--------------------------------------------------
sadp_core.py's hidden-layer update compares a hidden neuron's spike train
against the correct-class OUTPUT neuron's spike train (an agreement /
"does this hidden neuron look like the right answer" signal) - inherently
label-dependent even with no extra supervision, since it needs to know
which output neuron is "correct" to build that target spike train.

This module replaces that with classical pair-based STDP: each hidden
neuron's synapses are updated from the relative TIMING of its own
PRE-synaptic input spikes and its own POST-synaptic output spikes, via the
standard exponential-eligibility-trace formulation (`compute_stdp_dW`):
  - a postsynaptic spike potentiates synapses from presynaptic neurons that
    fired recently (within a window set by `tau_plus`);
  - a presynaptic spike depresses synapses to postsynaptic neurons that
    fired recently (within a window set by `tau_minus`).
With `reward_mode='none'`, this is genuinely UNSUPERVISED at the hidden
layer - the label only ever reaches the network through the output layer's
(unchanged) supervised delta rule. This is a real, qualitative difference
from SADP, where even the "no extra supervision" case (k_shift=0) is still
label-dependent at the hidden layer.

`SADPConfig.k_shift` / `weight_by_overlap` (SADP-specific - they control
the width and weighting of the shifted-kappa aggregation window) are
replaced here by STDP's own natural knobs: `tau_plus` / `tau_minus` (the
eligibility-trace time constants - how many timesteps back a spike still
"counts", playing a similar role to k_shift's window width) and `a_plus` /
`a_minus` (potentiation / depression amplitudes).

Notes on reward-modulated supervision (R-STDP)
-------------------------------------------------
`STDPConfig.reward_mode` is the same API as `SADPConfig.reward_mode`, and
uses the exact same reward definitions (`binary`: +-1 for correct/
incorrect; `margin`: continuous (correct_count - best_other_count)/T) -
this is deliberate, so a side-by-side SADP-vs-STDP comparison is testing
the learning rule, not also accidentally testing a different reward
definition. Multiplying a local plasticity signal by a global reward is
exactly the standard "three-factor" reward-modulated STDP (R-STDP)
formulation from computational neuroscience (closely related to
dopamine-modulated synaptic plasticity) - turning the same opt-in
mechanism that produced reward-modulated SADP into reward-modulated STDP
here. `reward_mode="none"` (the default) skips the multiply entirely, so
it is a strictly additive, opt-in option, same as in sadp_core.py.

A performance note: unlike SADP's `compute_shifted_kappa` (which loops
over `2*k_shift+1` offsets, each a vectorized whole-window operation),
`compute_stdp_dW` needs a genuine sequential pass over all T timesteps
(the eligibility traces are recursive in time), so it adds roughly as many
Python-level loop iterations per batch as the LIF `forward()` pass already
does. Expect STDP training to run somewhat slower than SADP at small
k_shift, for the same T.

A calibration note: the `eta_in` default (2e-4) was tuned for SADP's
kappa-based signal, which is bounded to roughly [-1, 1]. STDP's raw
trace-based dW1 has a different, less tightly bounded scale (it depends on
spike rates and `tau_plus`/`tau_minus`/`a_plus`/`a_minus`), so this default
is a starting point, not a validated value for STDP - expect to need to
retune `eta_in` (and possibly `a_plus`/`a_minus`) once you have real
results, rather than assuming the SADP-tuned number transfers directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

try:
    from tqdm.auto import trange, tqdm
except ImportError:  # pragma: no cover
    def trange(*args, **kwargs):
        return range(*args)

    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(0)


# ---------------------------------------------------------------------------
# 1. Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("stdp")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


def configure_logging(log_file: Optional[str] = None, level: int = logging.INFO,
                       to_console: bool = True) -> logging.Logger:
    """Attach console / file handlers with timestamps to the `stdp` logger.

    Safe to call multiple times (clears old handlers first).
    """
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Seed numpy (and TensorFlow, if available/used for the CNN encoder)."""
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 2. Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name: str = "mnist") -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                Tuple[np.ndarray, np.ndarray],
                                                Tuple[int, ...]]:
    """Load and preprocess MNIST, Fashion-MNIST, or CIFAR-10 via Keras.

    Returns (x_train, y_train), (x_test, y_test), input_shape, with images
    normalized to [0, 1] float32 and a trailing channel dimension.
    """
    name = name.lower()
    try:
        from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required to load the built-in benchmark datasets "
            "(MNIST/Fashion-MNIST/CIFAR-10). Install tensorflow, or load your "
            "own dataset as numpy arrays and pass it via `data=...`."
        ) from exc

    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif name in ("fmnist", "fashion_mnist"):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train, y_test = y_train.squeeze(), y_test.squeeze()
    else:
        raise ValueError("Dataset must be one of: 'mnist', 'fmnist', or 'cifar10'.")

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    if x_train.ndim == 3:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    input_shape = x_train.shape[1:]
    logger.info("Loaded %s | shape=%s | train=%s test=%s",
                name.upper(), input_shape, x_train.shape, x_test.shape)

    return (x_train, y_train), (x_test, y_test), input_shape


# ---------------------------------------------------------------------------
# 3. Temporal spike encoders
# ---------------------------------------------------------------------------

def poisson_encode_features(batch_feats: np.ndarray, T: int,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Bernoulli/Poisson rate-coding: feature value in [0,1] -> spike train.

    batch_feats: (B, Nin) in [0, 1]. Returns spikes of shape (B, T, Nin).

    Always pass an explicit `rng` for reproducible runs. `train_snn` and
    `evaluate_snn` both do this (threading the same seeded generator they
    already use for batch shuffling), so every internal call site in this
    module is reproducible. Without `rng`, this falls back to numpy's
    global, unseeded random state - fine for a one-off interactive call,
    but it means spike sampling would NOT be controlled by `cfg.seed`, and
    a benchmarking run's "same seed -> comparable runs" assumption would
    silently stop holding if any new call site is added without passing
    `rng` through.
    """
    B, Nin = batch_feats.shape
    if rng is None:
        rnd = np.random.rand(B, T, Nin).astype(np.float32)
    else:
        rnd = rng.random((B, T, Nin)).astype(np.float32)
    spikes = (rnd < batch_feats[:, None, :]).astype(np.float32)
    return spikes


# ---------------------------------------------------------------------------
# 4. Classical (non-neural-network) image feature extractors
# ---------------------------------------------------------------------------

def _to_grayscale(images: np.ndarray) -> np.ndarray:
    """images: (B, H, W) or (B, H, W, C) in [0,1] -> grayscale (B, H, W)."""
    if images.ndim == 3:
        return images.astype(np.float32)
    if images.shape[-1] == 1:
        return images[..., 0].astype(np.float32)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    c = images.shape[-1]
    w = weights[:c] if c <= 3 else np.ones(c, dtype=np.float32) / c
    w = w / w.sum()
    return np.tensordot(images.astype(np.float32), w, axes=([-1], [0]))


def _shift2d(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a batch of 2D images (B,H,W) by (dy,dx) with edge replication."""
    B, H, W = img.shape
    padded = np.pad(img, ((0, 0), (1, 1), (1, 1)), mode="edge")
    y0, x0 = 1 + dy, 1 + dx
    return padded[:, y0:y0 + H, x0:x0 + W]


# 8-neighborhood offsets in clockwise order starting from top-left (P=8, R=1)
_LBP_OFFSETS: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)
]


def compute_lbp_codes(gray_batch: np.ndarray) -> np.ndarray:
    """Classic single-scale Local Binary Pattern, P=8 neighbors, R=1.

    gray_batch: (B, H, W) in [0,1]. Returns integer codes in [0,255],
    shape (B, H, W).
    """
    B, H, W = gray_batch.shape
    code = np.zeros((B, H, W), dtype=np.int32)
    center = gray_batch
    for p, (dy, dx) in enumerate(_LBP_OFFSETS):
        neighbor = _shift2d(gray_batch, dy, dx)
        bit = (neighbor >= center).astype(np.int32)
        code += bit << p
    return code


def _block_histogram(code_map: np.ndarray, grid: Tuple[int, int],
                      n_bins: int, code_range: Tuple[int, int] = (0, 256)) -> np.ndarray:
    """Per-block normalized histograms of an integer code map (vectorized).

    code_map: (B, H, W) integer codes. Returns (B, grid[0]*grid[1]*n_bins)
    features in [0,1] (each block histogram is L1-normalized).
    """
    B, H, W = code_map.shape
    gy, gx = grid
    bh, bw = H // gy, W // gx
    lo, hi = code_range
    bin_width = (hi - lo) / float(n_bins)
    bin_idx_full = np.clip(((code_map - lo) / bin_width).astype(np.int64), 0, n_bins - 1)

    feats = np.zeros((B, gy, gx, n_bins), dtype=np.float32)
    row_offset = (np.arange(B, dtype=np.int64) * n_bins)[:, None]
    for i in range(gy):
        for j in range(gx):
            block_bins = bin_idx_full[:, i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            block_bins = block_bins.reshape(B, -1)
            flat = (block_bins + row_offset).ravel()
            counts = np.bincount(flat, minlength=B * n_bins).reshape(B, n_bins)
            totals = counts.sum(axis=1, keepdims=True).astype(np.float32)
            totals[totals == 0] = 1.0
            feats[:, i, j, :] = counts.astype(np.float32) / totals
    return feats.reshape(B, -1)


def compute_color_block_moments(images: np.ndarray, grid: Tuple[int, int]) -> np.ndarray:
    """Cheap per-block, per-channel color statistics (mean & std).

    images: (B, H, W, C) in [0,1]. Returns (B, grid[0]*grid[1]*C*2) in [0,1].
    """
    if images.ndim == 3:
        return np.zeros((images.shape[0], 0), dtype=np.float32)
    B, H, W, C = images.shape
    gy, gx = grid
    bh, bw = H // gy, W // gx
    feats = np.zeros((B, gy, gx, C, 2), dtype=np.float32)
    for i in range(gy):
        for j in range(gx):
            block = images[:, i * bh:(i + 1) * bh, j * bw:(j + 1) * bw, :]
            block = block.reshape(B, -1, C)
            feats[:, i, j, :, 0] = block.mean(axis=1)
            feats[:, i, j, :, 1] = np.clip(block.std(axis=1) * 2.0, 0.0, 1.0)
    return feats.reshape(B, -1)


# ===========================================================================
# 4a. Complete LBP (CLBP) -- sign + magnitude + center decomposition
# ===========================================================================
#
# CLBP decomposes local differences into three complementary components:
#   - CLBP_S: sign component      = classic LBP (s(g_p - g_c))
#   - CLBP_M: magnitude component = thresholded |g_p - g_c|
#   - CLBP_C: center component    = thresholded g_c vs. the image's global mean
# All three are computed over the same 8-neighbor square ring used by
# `compute_lbp_codes`. Concatenating their three histograms captures more
# of the local texture (not just *which* neighbors are brighter, but *how
# much* brighter, plus where the pixel itself sits relative to the image)
# for a modest cost over plain LBP (roughly 2x the histogram features, for
# the same spatial grid). Still zero learnable parameters - the spiking
# network does all of the class-discriminative learning.
#
# (LDP, BRIEF, a combined LBP+LDP+BRIEF descriptor, rotation-invariant LBP,
# uniform LBP, and multi-block LBP were all also tried at various points;
# none beat plain LBP by enough to justify the extra complexity/cost except
# CLBP, which is the one kept here alongside the original single-scale LBP.)

def compute_clbp_features(gray_batch: np.ndarray, grid: Tuple[int, int],
                           n_bins: int = 16) -> np.ndarray:
    """Complete LBP (CLBP): concatenate histograms of CLBP_S, CLBP_M, CLBP_C.

    gray_batch: (B, H, W) in [0,1].
    Returns: (B, 2*grid[0]*grid[1]*n_bins + grid[0]*grid[1]*2) float32
             features, not yet normalized (the caller min-max normalizes
             the full concatenated descriptor).
    """
    B, H, W = gray_batch.shape
    center = gray_batch

    # CLBP_S: sign bits -- identical to standard LBP
    clbp_s = compute_lbp_codes(gray_batch)  # int codes in [0,255]

    # CLBP_M: magnitude codes -- threshold |diff| against its per-image mean
    abs_diffs = np.zeros((B, H, W, 8), dtype=np.float32)
    for p, (dy, dx) in enumerate(_LBP_OFFSETS):
        neighbor = _shift2d(gray_batch, dy, dx)
        abs_diffs[..., p] = np.abs(neighbor - center)
    m_thresh = abs_diffs.mean(axis=(1, 2, 3), keepdims=True)  # (B,1,1,1)
    mag_bits = (abs_diffs >= m_thresh).astype(np.int32)
    clbp_m = np.zeros((B, H, W), dtype=np.int32)
    for p in range(8):
        clbp_m += mag_bits[..., p] << p  # int codes [0,255]

    # CLBP_C: center intensity vs. the image's global mean (1 bit/pixel)
    img_mean = gray_batch.mean(axis=(1, 2), keepdims=True)  # (B,1,1)
    clbp_c_bin = (center >= img_mean).astype(np.int32)  # 0 or 1

    feat_s = _block_histogram(clbp_s, grid, n_bins, (0, 256))
    feat_m = _block_histogram(clbp_m, grid, n_bins, (0, 256))
    feat_c = _block_histogram(clbp_c_bin, grid, n_bins=2, code_range=(0, 2))

    return np.concatenate([feat_s, feat_m, feat_c], axis=1)


def extract_classical_features(
    images: np.ndarray,
    method: str = "lbp",
    grid: Tuple[int, int] = (4, 4),
    n_bins: int = 16,
    include_color_moments: bool = True,
    batch_size: int = 512,
    show_progress: bool = True,
) -> np.ndarray:
    """NON-trainable feature extractor for 'lbp+poisson' and 'lbp_clbp+poisson'.

    method : 'lbp' (classic single-scale LBP, P=8 R=1) or 'lbp_clbp'
              (Complete LBP: sign + magnitude + center histograms
              concatenated).
    images : (N, H, W) or (N, H, W, C) float32 in [0, 1].
    Returns: (N, D) float32 in [0, 1], globally min-max normalized, ready
              for `poisson_encode_features`.
    """
    N = images.shape[0]
    n_batches = int(np.ceil(N / batch_size))
    out_chunks = []
    iterator = trange(n_batches, desc=f"{method} features", disable=not show_progress)

    for bi in iterator:
        s, e = bi * batch_size, min((bi + 1) * batch_size, N)
        chunk = images[s:e]
        gray = _to_grayscale(chunk)

        if method == "lbp":
            codes = compute_lbp_codes(gray)
            feats_list = [_block_histogram(codes, grid, n_bins, (0, 256))]
        elif method == "lbp_clbp":
            feats_list = [compute_clbp_features(gray, grid, n_bins)]
        else:
            raise ValueError(
                f"Unknown classical method '{method}'. Choose 'lbp' or 'lbp_clbp'."
            )

        if include_color_moments and chunk.ndim == 4 and chunk.shape[-1] >= 3:
            feats_list.append(compute_color_block_moments(chunk, grid))

        out_chunks.append(np.concatenate(feats_list, axis=1).astype(np.float32))

    feats = np.concatenate(out_chunks, axis=0)
    fmin = feats.min(axis=0, keepdims=True)
    fmax = feats.max(axis=0, keepdims=True)
    frange = fmax - fmin
    frange[frange < 1e-8] = 1.0
    feats = np.clip((feats - fmin) / frange, 0.0, 1.0).astype(np.float32)

    logger.info(
        "%s features | grid=%s | n_bins=%d | D=%d | color_moments=%s",
        method, grid, n_bins, feats.shape[1],
        include_color_moments and images.ndim == 4 and images.shape[-1] >= 3,
    )
    return feats


# ---------------------------------------------------------------------------
# 5. Optional CNN encoder (kept from the original pipeline for comparison)
# ---------------------------------------------------------------------------

def build_cnn_encoder(input_shape: Tuple[int, ...] = (32, 32, 3), output_dim: int = 256):
    """The original trainable CNN feature extractor (requires TensorFlow).

    Kept for direct comparison against the non-NN classical encoders.
    """
    try:
        from tensorflow.keras import layers, models
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required for the 'cnn+poisson' encoding pathway. "
            "Install tensorflow, or use 'poisson_only', 'lbp+poisson', or 'lbp_clbp+poisson'."
        ) from exc

    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(output_dim, activation="sigmoid")(x)
    return models.Model(inp, x, name="encoder_small")


def extract_cnn_features(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                          input_shape: Tuple[int, ...], feature_dim: int = 256,
                          n_classes: Optional[int] = None, encoder_epochs: int = 50,
                          batch_size_encoder: int = 128, pretrain: bool = True,
                          seed: int = 42, verbose: int = 2
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Pretrain the CNN encoder and extract normalized features."""
    try:
        from tensorflow.keras import layers, models
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required for the 'cnn+poisson' encoding pathway."
        ) from exc

    tf.random.set_seed(seed)
    n_classes = n_classes if n_classes is not None else len(np.unique(y_train))

    encoder = build_cnn_encoder(input_shape=input_shape, output_dim=feature_dim)

    if pretrain:
        inp = encoder.input
        feat = encoder.output
        out = layers.Dense(n_classes, activation="softmax")(feat)
        clf = models.Model(inp, out)
        clf.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
        logger.info("Pretraining CNN encoder for %d epochs ...", encoder_epochs)
        clf.fit(x_train, y_train, validation_split=0.1, epochs=encoder_epochs,
                batch_size=batch_size_encoder, verbose=verbose)

    logger.info("Extracting CNN features ...")
    train_feats = encoder.predict(x_train, batch_size=256, verbose=verbose)
    test_feats = encoder.predict(x_test, batch_size=256, verbose=verbose)

    minf = train_feats.min(axis=0, keepdims=True)
    maxf = train_feats.max(axis=0, keepdims=True)
    rangef = (maxf - minf) + 1e-9
    train_feats_norm = (train_feats - minf) / rangef
    test_feats_norm = (test_feats - minf) / rangef
    return train_feats_norm.astype(np.float32), test_feats_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. The STDP / reward-modulated STDP (R-STDP) network
# ---------------------------------------------------------------------------
#
# This is a drop-in replacement for the SADP hidden-layer learning rule in
# sadp_core.py. Everything else in this file (encoders, dataset loading,
# the output-layer supervised delta rule, training/eval loops,
# run_experiment orchestration) is unchanged - only the hidden-layer
# plasticity mechanism differs:
#
#   SADP : compares the hidden neuron's spike train against the
#          correct-class OUTPUT neuron's spike train (an agreement /
#          "does this hidden neuron look like the right answer" signal).
#          Inherently label-dependent even with no extra supervision.
#
#   STDP : compares the hidden neuron's own PRE-synaptic input spikes
#          against its own POST-synaptic output spikes, using their
#          relative timing (classic pair-based STDP with exponential
#          eligibility traces). With reward_mode='none' this is genuinely
#          UNSUPERVISED at the hidden layer - the label only ever reaches
#          the network through the (unchanged) output-layer delta rule.
#          Setting reward_mode to 'binary' or 'margin' turns it into
#          reward-modulated STDP (R-STDP): the same per-sample global
#          reward factor used for SADP now scales the local STDP
#          eligibility instead of the kappa-agreement signal - the
#          standard three-factor R-STDP formulation from the
#          computational-neuroscience literature (closely related to
#          dopamine-modulated synaptic plasticity), not something bespoke.
#
# k_shift / weight_by_overlap (SADP-specific) are replaced by STDP's own
# natural knobs: tau_plus / tau_minus (how many timesteps back a spike
# still "counts" - the temporal window, playing a similar role to
# k_shift's window width) and a_plus / a_minus (the potentiation /
# depression amplitudes).

@dataclass
class STDPConfig:
    """All hyperparameters needed to build and train a (reward-modulated)
    STDP spiking network."""
    Nin: int
    Nhid: int
    Nout: int
    architecture: str = "1STDP"          # '1STDP' or '2STDP'
    T: int = 25
    lam: float = 0.9
    theta_h_base: float = 0.5
    theta_o: float = 0.5
    eta_out: float = 5e-4
    eta_in: float = 2e-4
    decay: float = 0.9995
    norm_eps: float = 1e-6
    clip_w2: float = 5.0
    # STDP-specific: exponential eligibility-trace time constants (in
    # timesteps) and potentiation/depression amplitudes.
    tau_plus: float = 5.0
    tau_minus: float = 5.0
    a_plus: float = 1.0
    a_minus: float = 1.0
    # Reward modulation - identical in spirit and API to sadp_core.py's
    # SADPConfig.reward_mode: a third, global per-sample factor multiplying
    # the local plasticity signal. 'none' = plain unsupervised STDP at the
    # hidden layer; 'binary'/'margin' = reward-modulated STDP (R-STDP).
    reward_mode: str = "none"
    reward_scale: float = 1.0
    reward_baseline: bool = False
    reward_baseline_decay: float = 0.99
    seed: int = 42

    def __post_init__(self):
        if self.architecture not in ("1STDP", "2STDP"):
            raise ValueError("architecture must be '1STDP' or '2STDP'")
        if self.tau_plus <= 0 or self.tau_minus <= 0:
            raise ValueError("tau_plus and tau_minus must be > 0")
        if self.reward_mode not in ("none", "binary", "margin"):
            raise ValueError("reward_mode must be 'none', 'binary', or 'margin'")


def compute_stdp_dW(pre_spikes: np.ndarray, post_spikes: np.ndarray,
                     tau_plus: float = 5.0, tau_minus: float = 5.0,
                     a_plus: float = 1.0, a_minus: float = 1.0,
                     reward: Optional[np.ndarray] = None) -> np.ndarray:
    """Classic pair-based STDP with exponential eligibility traces,
    optionally reward-modulated, batch-averaged.

    pre_spikes  : (B, T, Npre) binary spike train (presynaptic).
    post_spikes : (B, T, Npost) binary spike train (postsynaptic).
    reward      : optional (B,) per-sample scalar. When given, each
                  sample's entire STDP contribution (both potentiation and
                  depression) is scaled by that sample's reward before
                  being averaged into the batch update - i.e. R-STDP. When
                  None, this is plain, unmodulated, unsupervised STDP.

    Returns (Npre, Npost) weight delta, already averaged over the batch -
    same convention as `compute_shifted_kappa`'s caller in sadp_core.py, so
    `self.W += eta * dW` is all that's needed downstream.

    At every timestep t:
      - Traces are decayed FIRST, then read for the cross-term, then the
        current timestep's own spike is folded in for future timesteps.
        This makes a pre/post pair separated by `dt` timesteps contribute
        exp(-dt/tau) - the textbook STDP window - rather than
        exp(-(dt-1)/tau) (an earlier version of this function decayed
        AFTER reading the trace, which silently shifted every pairing by
        one timestep: dt=1 got zero decay, dt=2 got one decay step
        instead of two, etc. - a clean, consistent, but real off-by-one,
        confirmed by hand-tracing a single pre/post spike pair against
        the closed-form exp(-dt/tau)).
      - LTP: a postsynaptic spike potentiates synapses from any
        presynaptic neuron that fired recently (tracked via the decaying
        pre_trace).
      - LTD: a presynaptic spike depresses synapses to any postsynaptic
        neuron that fired recently (tracked via the decaying post_trace).
      - A literal same-timestep coincidence (dt=0) contributes to neither
        LTP nor LTD directly (only to future timesteps' traces) - this is
        a deliberate, standard convention for discrete-time STDP, not an
        oversight: there is no single universally "correct" way to handle
        exact ties, and excluding them is the common choice.
    Standard efficient trace-based formulation - O(T) per batch, same
    complexity class as the SADP k-shifted kappa computation for a fixed
    k_shift.
    """
    B, T, Npre = pre_spikes.shape
    Npost = post_spikes.shape[2]
    decay_plus = float(np.exp(-1.0 / tau_plus))
    decay_minus = float(np.exp(-1.0 / tau_minus))

    r = reward if reward is not None else np.ones(B, dtype=np.float32)

    pre_trace = np.zeros((B, Npre), dtype=np.float32)
    post_trace = np.zeros((B, Npost), dtype=np.float32)
    dW = np.zeros((Npre, Npost), dtype=np.float32)

    for t in range(T):
        pre_t = pre_spikes[:, t, :]
        post_t = post_spikes[:, t, :]

        # Decay first, so a spike `dt` steps ago has been decayed exactly
        # `dt` times by the time it's read below.
        pre_trace = decay_plus * pre_trace
        post_trace = decay_minus * post_trace

        # LTP: pre fired earlier (pre_trace), post fires now -> potentiate.
        dW += a_plus * np.einsum("bi,bj->ij", pre_trace, post_t * r[:, None])
        # LTD: post fired earlier (post_trace), pre fires now -> depress.
        dW -= a_minus * np.einsum("bi,bj->ij", pre_t * r[:, None], post_trace)

        # Fold in this timestep's own spike for future timesteps (NOT used
        # in this timestep's own cross-term above - see dt=0 note).
        pre_trace = pre_trace + pre_t
        post_trace = post_trace + post_t

    return dW / B


def calibrate_eta_in(cfg: "STDPConfig", sample_batch: np.ndarray,
                      target_mean_abs_dw1: float,
                      seed: Optional[int] = None) -> float:
    """Suggest an `eta_in` for STDPConfig `cfg` so that the RAW (pre-eta_in)
    hidden-layer update on `sample_batch` has a given target mean |dW1|.

    Why this exists: SADP's kappa-agreement signal is bounded to [-1,1] by
    construction; STDP's trace-based signal has no such bound and its scale
    depends on tau_plus/tau_minus/a_plus/a_minus and the data's own spike
    rates. Measured on one matched batch (random features, Nin=Nhid=256,
    T=25, default tau=5/a=1 vs default SADP k_shift=0): STDP's raw mean
    |dW1| was roughly 60-90x larger than SADP's depending on tau (~66x
    measured at tau=5). Reusing the same `eta_in` number for both rules is
    therefore NOT a controlled comparison by default - it gives STDP a
    much larger effective step size at the same nominal learning rate.

    `target_mean_abs_dw1` has no built-in default on purpose - a guessed
    number here would be just as ungrounded as reusing SADP's eta_in
    verbatim. Compute a real target from the SADP side first, e.g.:

        import sadp_core as sc, stdp_core as tc
        sadp_cfg = sc.SADPConfig(Nin=Nin, Nhid=Nhid, Nout=Nout, T=T,
                                  k_shift=YOUR_K, seed=SEED)
        net = sc.SADPNetwork(sadp_cfg)
        sh, sh2, so = net.forward(sample_batch, rng=np.random.default_rng(SEED))
        target_spikes = so[np.arange(len(sample_batch)), :, y_sample].astype("float32")
        kappa1 = sc.compute_shifted_kappa(sh, target_spikes, k_shift=YOUR_K)
        raw_sadp = np.mean(np.abs(np.einsum("bi,bj->ij", sample_batch, kappa1) / len(sample_batch)))
        target = sadp_cfg.eta_in * raw_sadp   # SADP's actual effective step

        stdp_eta_in = tc.calibrate_eta_in(stdp_cfg, sample_batch, target_mean_abs_dw1=target)

    That gives an `eta_in` for STDP whose effective per-batch step matches
    SADP's *current* effective step for the SADP settings you actually
    plan to run against - the only grounded definition of "matched" here,
    since SADP's own raw magnitude also shifts somewhat with k_shift.

    This does NOT change any training-time behavior - it is a one-off,
    explicit, opt-in measurement you call before choosing `eta_in`. Nothing
    in `STDPNetwork.update()` auto-rescales itself.

    sample_batch: (B, Nin) feature batch, e.g. a slice of your real
        training data (the magnitude depends on real feature statistics,
        not just shape, so prefer real data over synthetic stand-ins).
    Returns: suggested eta_in (float). cfg.eta_in is returned unchanged if
        the sample batch produces no signal at all (e.g. no spikes).
    """
    probe_cfg = STDPConfig(
        Nin=cfg.Nin, Nhid=cfg.Nhid, Nout=cfg.Nout, architecture=cfg.architecture,
        T=cfg.T, lam=cfg.lam, theta_h_base=cfg.theta_h_base, theta_o=cfg.theta_o,
        tau_plus=cfg.tau_plus, tau_minus=cfg.tau_minus,
        a_plus=cfg.a_plus, a_minus=cfg.a_minus, seed=cfg.seed,
    )
    net = STDPNetwork(probe_cfg)
    rng = np.random.default_rng(seed)
    S_in, spikes_h, spikes_h2, spikes_o = net.forward(sample_batch, rng=rng)
    dW1 = compute_stdp_dW(
        S_in, spikes_h, tau_plus=cfg.tau_plus, tau_minus=cfg.tau_minus,
        a_plus=cfg.a_plus, a_minus=cfg.a_minus, reward=None,
    )
    raw_mag = float(np.mean(np.abs(dW1)))
    if raw_mag < 1e-12:
        logger.warning(
            "calibrate_eta_in: raw STDP signal is ~0 on this sample batch "
            "(no spikes reached the hidden layer?) - returning cfg.eta_in unchanged."
        )
        return cfg.eta_in
    suggested = target_mean_abs_dw1 / raw_mag
    logger.info(
        "calibrate_eta_in: raw mean|dW1|=%.6g on this batch -> suggested "
        "eta_in=%.4g (targets mean|eta_in*dW1|=%.4g)",
        raw_mag, suggested, target_mean_abs_dw1,
    )
    return suggested


def _compute_reward(out_counts: np.ndarray, preds: np.ndarray, y_batch: np.ndarray,
                     T: int, mode: str = "binary") -> np.ndarray:
    """Global per-sample reward r (B,) for reward-modulated supervision.
    Identical to sadp_core.py's version - reused here unchanged so R-STDP
    and reward-modulated SADP use exactly the same reward definitions."""
    B = out_counts.shape[0]
    if mode == "binary":
        return np.where(preds == y_batch, 1.0, -1.0).astype(np.float32)
    if mode == "margin":
        correct_counts = out_counts[np.arange(B), y_batch]
        masked = out_counts.copy()
        masked[np.arange(B), y_batch] = -np.inf
        best_other = masked.max(axis=1)
        best_other = np.where(np.isfinite(best_other), best_other, 0.0)
        return ((correct_counts - best_other) / max(T, 1)).astype(np.float32)
    raise ValueError(f"Unknown reward_mode '{mode}'.")


class STDPNetwork:
    """(Reward-modulated) STDP spiking network (1STDP or 2STDP)."""

    def __init__(self, config: STDPConfig):
        self.cfg = config
        rng = np.random.default_rng(config.seed)

        Nin, Nhid, Nout = config.Nin, config.Nhid, config.Nout
        self.W1 = rng.normal(0, 0.1, (Nin, Nhid)).astype(np.float32)
        self.W2 = rng.normal(0, 0.1, (Nhid, Nout)).astype(np.float32)
        self.theta_h = (config.theta_h_base + 0.05 * rng.standard_normal(Nhid)).astype(np.float32)

        if config.architecture == "2STDP":
            self.W1_2 = rng.normal(0, 0.1, (Nhid, Nhid)).astype(np.float32)
            self.theta_h2 = (config.theta_h_base + 0.05 * rng.standard_normal(Nhid)).astype(np.float32)
        else:
            self.W1_2 = None
            self.theta_h2 = None

        self._reward_baseline = 0.0

        logger.info(
            "Initialized STDPNetwork | arch=%s | Nin=%d Nhid=%d Nout=%d | T=%d | "
            "tau_plus=%.1f tau_minus=%.1f | reward_mode=%s",
            config.architecture, Nin, Nhid, Nout, config.T,
            config.tau_plus, config.tau_minus, config.reward_mode,
        )

    def forward(self, x_batch_feats: np.ndarray,
                rng: Optional[np.random.Generator] = None
                ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """LIF forward pass over T timesteps. Identical dynamics to
        SADPNetwork.forward in sadp_core.py, except this also returns the
        input spike train S_in, which the STDP update rule needs (SADP's
        update only ever needed the static feature vector, not the actual
        per-timestep input spikes). Returns (S_in, spikes_h, spikes_h2,
        spikes_o)."""
        cfg = self.cfg
        B = x_batch_feats.shape[0]
        T = cfg.T

        Vh = np.zeros((B, cfg.Nhid), dtype=np.float32)
        Vh2 = np.zeros((B, cfg.Nhid), dtype=np.float32) if cfg.architecture == "2STDP" else None
        Vo = np.zeros((B, cfg.Nout), dtype=np.float32)

        spikes_h = np.zeros((B, T, cfg.Nhid), dtype=np.float32)
        spikes_h2 = np.zeros((B, T, cfg.Nhid), dtype=np.float32) if cfg.architecture == "2STDP" else None
        spikes_o = np.zeros((B, T, cfg.Nout), dtype=np.float32)

        S_in = poisson_encode_features(x_batch_feats, T, rng=rng)

        for t in range(T):
            I_h = np.einsum("bi,ij->bj", S_in[:, t], self.W1)
            Vh = cfg.lam * Vh + I_h
            spk_h = (Vh > self.theta_h).astype(np.float32)
            Vh[spk_h == 1] = 0.0
            spikes_h[:, t, :] = spk_h

            if cfg.architecture == "2STDP":
                I_h2 = np.einsum("bi,ij->bj", spk_h, self.W1_2)
                Vh2 = cfg.lam * Vh2 + I_h2
                spk_h2 = (Vh2 > self.theta_h2).astype(np.float32)
                Vh2[spk_h2 == 1] = 0.0
                spikes_h2[:, t, :] = spk_h2
                spk_to_output = spk_h2
            else:
                spk_to_output = spk_h

            I_o = np.einsum("bi,ij->bj", spk_to_output, self.W2)
            Vo = cfg.lam * Vo + I_o
            spk_o = (Vo > cfg.theta_o).astype(np.float32)
            Vo[spk_o == 1] = 0.0
            spikes_o[:, t, :] = spk_o

        return S_in, spikes_h, spikes_h2, spikes_o

    def update(self, S_in: np.ndarray, y_batch: np.ndarray,
               spikes_h: np.ndarray, spikes_h2: Optional[np.ndarray],
               spikes_o: np.ndarray) -> Tuple[np.ndarray, float, Optional[float]]:
        """One mini-batch weight update: supervised delta rule at the
        output layer (unchanged from SADP), (reward-modulated) STDP at the
        hidden layer(s) (replaces SADP's k-shifted kappa rule).
        Returns (preds, mean_abs_dW1, mean_reward)."""
        cfg = self.cfg
        B, T, _ = spikes_o.shape

        targets = np.zeros((B, cfg.Nout), dtype=np.float32)
        targets[np.arange(B), y_batch] = 1.0

        out_counts = spikes_o.sum(axis=1)
        preds = np.argmax(out_counts, axis=1)

        # ---- Output layer: supervised Hebbian/delta update (unchanged) ----
        errors = targets[:, None, :] - spikes_o
        pre_out = spikes_h2 if cfg.architecture == "2STDP" else spikes_h
        dW2 = np.einsum("bti,btj->ij", pre_out, errors) / B
        self.W2 += cfg.eta_out * dW2

        # ---- Optional global reward factor (identical to SADP's) ----
        r = None
        mean_reward = None
        if cfg.reward_mode != "none":
            r = _compute_reward(out_counts, preds, y_batch, T, mode=cfg.reward_mode)
            if cfg.reward_baseline:
                self._reward_baseline = (
                    cfg.reward_baseline_decay * self._reward_baseline
                    + (1.0 - cfg.reward_baseline_decay) * float(np.mean(r))
                )
                r = r - self._reward_baseline
            r = (cfg.reward_scale * r).astype(np.float32)
            mean_reward = float(np.mean(r))

        # ---- W1 (input -> hidden-1): (reward-modulated) STDP ----
        dW1 = compute_stdp_dW(
            S_in, spikes_h, tau_plus=cfg.tau_plus, tau_minus=cfg.tau_minus,
            a_plus=cfg.a_plus, a_minus=cfg.a_minus, reward=r,
        )
        self.W1 += cfg.eta_in * dW1

        # ---- W1_2 (hidden-1 -> hidden-2): STDP, 2STDP only ----
        if cfg.architecture == "2STDP":
            dW1_2 = compute_stdp_dW(
                spikes_h, spikes_h2, tau_plus=cfg.tau_plus, tau_minus=cfg.tau_minus,
                a_plus=cfg.a_plus, a_minus=cfg.a_minus, reward=r,
            )
            self.W1_2 += cfg.eta_in * dW1_2
            self.W1_2 *= cfg.decay
            self.W1_2 /= (np.linalg.norm(self.W1_2, axis=0, keepdims=True) + cfg.norm_eps)

        # ---- decay & normalize (unchanged) ----
        self.W1 *= cfg.decay
        self.W2 *= cfg.decay
        self.W1 /= (np.linalg.norm(self.W1, axis=0, keepdims=True) + cfg.norm_eps)
        np.clip(self.W2, -cfg.clip_w2, cfg.clip_w2, out=self.W2)

        mean_abs_dW1 = float(np.mean(np.abs(dW1)))
        return preds, mean_abs_dW1, mean_reward


# ---------------------------------------------------------------------------
# 7. Train / evaluate loops
# ---------------------------------------------------------------------------

def train_snn(net: "STDPNetwork", train_feats: np.ndarray, y_train: np.ndarray,
              n_epochs: int = 50, batch_size: int = 128, n_samples: Optional[int] = None,
              seed: Optional[int] = None, show_progress: bool = True,
              eval_feats: Optional[np.ndarray] = None, y_eval: Optional[np.ndarray] = None,
              eval_every: Optional[int] = None, eval_batch_size: int = 256) -> Dict[str, Any]:
    """Train `net` in-place. Returns history dict.

    Weight-stability tracking (always on, negligible cost - O(Nin*Nhid)
    per epoch vs. the O(B*T*Nin*Nhid) cost of the LIF simulation itself
    that already happens every batch). Identical fields and rationale to
    sadp_core.py's train_snn, so the two are directly comparable:
      epoch_w1_cos_sim_prev : mean cosine similarity of each W1 column to
          its OWN value at the end of the previous epoch (1.0 = direction
          unchanged, 0 = orthogonal/fully rotated, negative = flipped).
          IMPORTANT: W1's *magnitude* is renormalized to ~1 per column
          after every single batch (see STDPNetwork.update), so a naive
          ||W1|| would be trivially ~constant regardless of how stable
          training actually is - this direction metric is the part of
          "weight stability" that ISN'T pinned by that renormalization.
      epoch_w1_frob_norm    : Frobenius norm of W1 - sanity check only
          (expect ~sqrt(Nhid), roughly constant); not informative alone.
      epoch_w2_frob_norm    : Frobenius norm of W2 - only clipped, never
          renormalized, so this one CAN genuinely drift - a real signal.

    Optional periodic held-out eval (off by default - costs an extra full
    forward pass over `eval_feats` every `eval_every` epochs; unlike the
    weight diagnostics above, this is NOT free): pass eval_feats/y_eval
    and eval_every=K to evaluate every K epochs. Leave eval_every=None
    (default) to skip entirely, matching prior behavior exactly.
    """
    rng = np.random.default_rng(seed)
    n_train_total = len(train_feats)
    n_train = n_train_total if n_samples is None else min(n_samples, n_train_total)

    history = {
        "epoch_accuracy": [], "epoch_stdp_signal": [], "epoch_reward": [], "epoch_time_s": [],
        "epoch_w1_cos_sim_prev": [], "epoch_w1_frob_norm": [], "epoch_w2_frob_norm": [],
        "epoch_eval_accuracy": [],
    }
    prev_W1 = net.W1.copy()

    for epoch in range(n_epochs):
        t0 = time.time()
        idx = rng.permutation(n_train_total)[:n_train]
        num_batches = n_train // batch_size

        correct = 0
        signal_log = []
        reward_log = []

        batch_iter = trange(num_batches, desc=f"Epoch {epoch + 1}/{n_epochs}",
                             disable=not show_progress, leave=False)
        for bi in batch_iter:
            s, e = bi * batch_size, bi * batch_size + batch_size
            batch_idx = idx[s:e]
            Xb = train_feats[batch_idx]
            yb = y_train[batch_idx]

            S_in, sh, sh2, so = net.forward(Xb, rng=rng)
            preds, signal_val, reward_val = net.update(S_in, yb, sh, sh2, so)

            correct += int(np.sum(preds == yb))
            signal_log.append(signal_val)
            if reward_val is not None:
                reward_log.append(reward_val)

        epoch_time = time.time() - t0
        acc = correct / float(n_train)
        avg_s = float(np.mean(signal_log)) if signal_log else 0.0
        avg_r = float(np.mean(reward_log)) if reward_log else None

        # ---- Weight-stability diagnostics (cheap, always on) ----
        # float64 on purpose: cos_sim values close to 1.0 are exactly
        # where the informative signal is (small early-training rotation),
        # and that's also exactly where float32 runs out of precision.
        prev_W1_64, W1_64 = prev_W1.astype(np.float64), net.W1.astype(np.float64)
        cos_num = np.sum(prev_W1_64 * W1_64, axis=0)
        cos_den = np.linalg.norm(prev_W1_64, axis=0) * np.linalg.norm(W1_64, axis=0) + 1e-12
        w1_cos_sim = float(np.mean(cos_num / cos_den))
        w1_frob = float(np.linalg.norm(net.W1))
        w2_frob = float(np.linalg.norm(net.W2))
        prev_W1 = net.W1.copy()

        history["epoch_accuracy"].append(acc)
        history["epoch_stdp_signal"].append(avg_s)
        history["epoch_reward"].append(avg_r)
        history["epoch_time_s"].append(epoch_time)
        history["epoch_w1_cos_sim_prev"].append(w1_cos_sim)
        history["epoch_w1_frob_norm"].append(w1_frob)
        history["epoch_w2_frob_norm"].append(w2_frob)

        # ---- Optional periodic eval (opt-in, has real cost) ----
        eval_log_str = ""
        if eval_feats is not None and eval_every is not None and (epoch + 1) % eval_every == 0:
            eval_metrics = evaluate_snn(net, eval_feats, y_eval, batch_size=eval_batch_size,
                                         seed=seed, show_progress=False)
            history["epoch_eval_accuracy"].append(eval_metrics["accuracy"])
            eval_log_str = f" | eval_acc={eval_metrics['accuracy']:.4f}"
        else:
            history["epoch_eval_accuracy"].append(None)

        logger.info(
            "Epoch %d/%d | train_acc=%.4f | mean_abs_dW1=%.6f%s%s | "
            "w1_cos_sim=%.4f w1_norm=%.3f w2_norm=%.3f | time=%.2fs",
            epoch + 1, n_epochs, acc, avg_s,
            f" | avg_reward={avg_r:.4f}" if avg_r is not None else "", eval_log_str,
            w1_cos_sim, w1_frob, w2_frob, epoch_time
        )

    return history


def evaluate_snn(net: "STDPNetwork", test_feats: np.ndarray, y_test: np.ndarray,
                  batch_size: int = 256, n_samples: Optional[int] = None,
                  seed: Optional[int] = None, show_progress: bool = True) -> Dict[str, Any]:
    """Evaluate `net` on held-out data. Returns metrics dict."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    rng = np.random.default_rng(seed)
    n_test_total = len(test_feats)
    n_test = n_test_total if n_samples is None else min(n_samples, n_test_total)
    idx = rng.permutation(n_test_total)[:n_test]
    num_batches = int(np.ceil(n_test / batch_size))

    all_preds = np.zeros(n_test, dtype=np.int64)
    all_true = y_test[idx]

    t0 = time.time()
    batch_iter = trange(num_batches, desc="Evaluating", disable=not show_progress, leave=False)
    for bi in batch_iter:
        s, e = bi * batch_size, min((bi + 1) * batch_size, n_test)
        batch_idx = idx[s:e]
        Xb = test_feats[batch_idx]
        _, sh, sh2, so = net.forward(Xb, rng=rng)
        all_preds[s:e] = np.argmax(so.sum(axis=1), axis=1)
    eval_time = time.time() - t0

    acc = float(np.mean(all_preds == all_true))
    prec = precision_score(all_true, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_true, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

    logger.info(
        "Eval | acc=%.4f | precision=%.4f | recall=%.4f | f1=%.4f | time=%.2fs",
        acc, prec, rec, f1, eval_time
    )

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "eval_time_s": eval_time, "n_eval": n_test,
    }


# ---------------------------------------------------------------------------
# 8. High-level orchestration
# ---------------------------------------------------------------------------

# Encoding types supported out of the box.
ENCODING_TYPES = (
    "poisson_only",
    "lbp+poisson",
    "lbp_clbp+poisson",
    "cnn+poisson",
)


def extract_features_for_dataset(
    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray,
    encoding_type: str,
    feature_dim: int = 256,
    n_classes: Optional[int] = None,
    encoder_epochs: int = 50,
    batch_size_encoder: int = 128,
    pretrain_encoder: bool = True,
    classical_grid: Tuple[int, int] = (4, 4),
    classical_n_bins: int = 16,
    include_color_moments: bool = True,
    seed: int = 42,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to the right encoder. Returns (train_feats, test_feats) in [0,1].

    encoding_type options
    ---------------------
    'poisson_only'      : raw pixel flatten -> Poisson.
    'lbp+poisson'       : classic single-scale LBP (P=8, R=1) -> Poisson.
    'lbp_clbp+poisson'  : Complete LBP (sign+magnitude+center) -> Poisson.
    'cnn+poisson'       : pretrained CNN features -> Poisson (requires TF).
    """
    encoding_type = encoding_type.lower()

    if encoding_type == "poisson_only":
        train_feats = x_train.reshape(len(x_train), -1).astype(np.float32)
        test_feats = x_test.reshape(len(x_test), -1).astype(np.float32)
        m = train_feats.max()
        if m > 0:
            train_feats /= m
            test_feats /= m
        return train_feats, test_feats

    if encoding_type == "cnn+poisson":
        if input_shape is None:
            input_shape = x_train.shape[1:]
        return extract_cnn_features(
            x_train, y_train, x_test, input_shape=input_shape, feature_dim=feature_dim,
            n_classes=n_classes, encoder_epochs=encoder_epochs,
            batch_size_encoder=batch_size_encoder, pretrain=pretrain_encoder, seed=seed,
        )

    if encoding_type in ("lbp+poisson", "lbp_clbp+poisson"):
        method = "lbp" if encoding_type == "lbp+poisson" else "lbp_clbp"
        train_feats = extract_classical_features(
            x_train, method=method, grid=classical_grid, n_bins=classical_n_bins,
            include_color_moments=include_color_moments,
        )
        test_feats = extract_classical_features(
            x_test, method=method, grid=classical_grid, n_bins=classical_n_bins,
            include_color_moments=include_color_moments,
        )
        return train_feats, test_feats

    raise ValueError(
        f"Unknown encoding_type '{encoding_type}'. Choose from {ENCODING_TYPES}."
    )


def run_experiment(
    dataset_name: Optional[str] = None,
    data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray],
                          Tuple[int, ...]]] = None,
    encoding_type: str = "lbp+poisson",
    architecture: str = "1STDP",
    T: int = 25,
    tau_plus: float = 5.0,
    tau_minus: float = 5.0,
    a_plus: float = 1.0,
    a_minus: float = 1.0,
    reward_mode: str = "none",
    reward_scale: float = 1.0,
    reward_baseline: bool = False,
    reward_baseline_decay: float = 0.99,
    Nhid: int = 256,
    theta_h_base: float = 0.5,
    eta_in: float = 2e-4,
    eta_out: float = 5e-4,
    seed: int = 42,
    feature_dim: int = 256,
    encoder_epochs: int = 50,
    batch_size_encoder: int = 128,
    pretrain_encoder: bool = True,
    classical_grid: Tuple[int, int] = (4, 4),
    classical_n_bins: int = 16,
    include_color_moments: bool = True,
    n_epochs: int = 50,
    batch_size: int = 128,
    eval_batch_size: int = 256,
    eval_every: Optional[int] = None,
    show_progress: bool = True,
    feature_cache: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    """Run one full (reward-modulated) STDP experiment end to end and return
    a flat results dict (ready to be appended to a results table /
    DataFrame). Structurally identical to sadp_core.py's run_experiment -
    same encodings, same dataset handling, same reward-modulation options -
    except k_shift/weight_by_overlap are replaced by STDP's own
    tau_plus/tau_minus/a_plus/a_minus.

    Either pass a built-in `dataset_name` ('mnist' | 'fmnist' | 'cifar10'),
    OR pass your own pre-loaded `data=((x_train,y_train),(x_test,y_test),
    input_shape)` for any other dataset - everything downstream is agnostic
    to where the images came from.

    feature_cache: an optional dict (e.g. shared across many calls from a
        benchmarking grid) used to memoize the extracted features. This
        matters most for 'cnn+poisson': the encoded features (and the CNN
        training that produces them) only depend on
        (dataset, encoding_type, feature_dim, encoder_epochs, seed, ...),
        NOT on architecture/T/tau/reward_mode - so when sweeping those, the
        very same features should be reused rather than re-extracted (or,
        worse, the CNN re-trained from scratch) for every grid cell. Pass
        the same dict across calls to get this reuse for free; leave it as
        None to always recompute (e.g. for a single one-off experiment).
    """
    if data is None:
        if dataset_name is None:
            raise ValueError("Provide either `dataset_name` or `data=...`.")
        (x_train, y_train), (x_test, y_test), input_shape = load_dataset(dataset_name)
    else:
        (x_train, y_train), (x_test, y_test), input_shape = data
        dataset_name = dataset_name or "custom"

    logger.info(
        "=" * 70 + "\nRunning experiment | dataset=%s | encoding=%s | arch=%s | "
        "T=%d | tau_plus=%.1f tau_minus=%.1f", dataset_name.upper(), encoding_type,
        architecture, T, tau_plus, tau_minus
    )

    set_seed(seed)
    n_classes = len(np.unique(y_train))

    cache_key = (
        dataset_name, encoding_type, feature_dim, encoder_epochs, batch_size_encoder,
        pretrain_encoder, classical_grid, classical_n_bins, include_color_moments, seed,
    )

    t_feat0 = time.time()
    if feature_cache is not None and cache_key in feature_cache:
        train_feats, test_feats = feature_cache[cache_key]
        feature_time = time.time() - t_feat0
        logger.info(
            "Reusing cached features | Nin=%d | (arch/T/tau/reward_mode do not affect encoding)",
            train_feats.shape[1]
        )
    else:
        train_feats, test_feats = extract_features_for_dataset(
            x_train, x_test, y_train,
            encoding_type=encoding_type,
            feature_dim=feature_dim,
            n_classes=n_classes,
            encoder_epochs=encoder_epochs,
            batch_size_encoder=batch_size_encoder,
            pretrain_encoder=pretrain_encoder,
            classical_grid=classical_grid,
            classical_n_bins=classical_n_bins,
            include_color_moments=include_color_moments,
            seed=seed,
            input_shape=input_shape,
        )
        feature_time = time.time() - t_feat0
        logger.info("Feature extraction done | Nin=%d | time=%.2fs",
                    train_feats.shape[1], feature_time)
        if feature_cache is not None:
            feature_cache[cache_key] = (train_feats, test_feats)

    cfg = STDPConfig(
        Nin=train_feats.shape[1], Nhid=Nhid, Nout=n_classes,
        architecture=architecture, T=T, theta_h_base=theta_h_base,
        eta_in=eta_in, eta_out=eta_out,
        tau_plus=tau_plus, tau_minus=tau_minus, a_plus=a_plus, a_minus=a_minus,
        reward_mode=reward_mode, reward_scale=reward_scale,
        reward_baseline=reward_baseline, reward_baseline_decay=reward_baseline_decay,
        seed=seed,
    )
    net = STDPNetwork(cfg)

    t_train0 = time.time()
    history = train_snn(net, train_feats, y_train, n_epochs=n_epochs, batch_size=batch_size,
                         seed=seed, show_progress=show_progress,
                         eval_feats=test_feats, y_eval=y_test,
                         eval_every=eval_every, eval_batch_size=eval_batch_size)
    total_train_time = time.time() - t_train0
    avg_time_per_epoch = total_train_time / max(len(history["epoch_accuracy"]), 1)

    metrics = evaluate_snn(net, test_feats, y_test, batch_size=eval_batch_size,
                            seed=seed, show_progress=show_progress)

    result = {
        "Dataset": dataset_name.upper(),
        "Encoding": encoding_type,
        "Architecture": architecture,
        "Timestep": T,
        "Tau_Plus": tau_plus,
        "Tau_Minus": tau_minus,
        "A_Plus": a_plus,
        "A_Minus": a_minus,
        "Reward_Mode": reward_mode,
        "Reward_Scale": reward_scale,
        "Reward_Baseline": reward_baseline,
        "Nin": train_feats.shape[1],
        "Nhid": Nhid,
        "Eta_In": eta_in,
        "Eta_Out": eta_out,
        "Seed": seed,
        "Final_Train_Accuracy": history["epoch_accuracy"][-1] if history["epoch_accuracy"] else None,
        "Eval_Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1_score": metrics["f1"],
        "Feature_Extraction_Time_s": feature_time,
        "Total_Train_Time_s": total_train_time,
        "Avg_Time_per_Epoch_s": avg_time_per_epoch,
        "Eval_Time_s": metrics["eval_time_s"],
        "Internal_Epochs_Run": len(history["epoch_accuracy"]),
        "Epoch_Accuracies": history["epoch_accuracy"],
        "Epoch_STDP_Signal": history["epoch_stdp_signal"],
        "Epoch_Rewards": history["epoch_reward"],
        "Epoch_W1_CosSim_Prev": history["epoch_w1_cos_sim_prev"],
        "Epoch_W1_FrobNorm": history["epoch_w1_frob_norm"],
        "Epoch_W2_FrobNorm": history["epoch_w2_frob_norm"],
        "Epoch_Eval_Accuracies": history["epoch_eval_accuracy"],
    }
    logger.info(
        "RESULT | %s | %s | %s | T=%d | tau=(%.1f,%.1f) | reward=%s -> eval_acc=%.4f f1=%.4f (%.2fs/epoch)",
        result["Dataset"], encoding_type, architecture, T, tau_plus, tau_minus, reward_mode,
        result["Eval_Accuracy"], result["F1_score"], avg_time_per_epoch,
    )
    return result


# ---------------------------------------------------------------------------
# 9. Self-test (synthetic data; no internet / TensorFlow required)
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(n_train=200, n_test=60, H=28, W=28, C=1, n_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    x_train = rng.random((n_train, H, W, C)).astype(np.float32)
    x_test = rng.random((n_test, H, W, C)).astype(np.float32)
    y_train = rng.integers(0, n_classes, size=n_train)
    y_test = rng.integers(0, n_classes, size=n_test)
    return (x_train, y_train), (x_test, y_test), (H, W, C)


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    logger.info("Running stdp_core.py self-test on synthetic data.")

    data_gray = _make_synthetic_dataset(H=28, W=28, C=1)
    data_color = _make_synthetic_dataset(H=32, W=32, C=3)

    # ---- Sanity check: CLBP's sign component == plain LBP codes ----
    gray_imgs = data_gray[0][0][:8, :, :, 0]   # (8, 28, 28)
    codes_orig = compute_lbp_codes(gray_imgs)
    clbp_feats = compute_clbp_features(gray_imgs, grid=(4, 4), n_bins=16)
    expected_dim = 2 * 4 * 4 * 16 + 4 * 4 * 2  # S hist + M hist + C hist
    assert clbp_feats.shape == (8, expected_dim), (
        f"CLBP feature shape mismatch: got {clbp_feats.shape}, expected (8, {expected_dim})"
    )
    s_hist_direct = _block_histogram(codes_orig, (4, 4), 16, (0, 256))
    assert np.allclose(clbp_feats[:, :s_hist_direct.shape[1]], s_hist_direct), (
        "REGRESSION FAIL: CLBP's sign component != plain LBP block histogram!"
    )
    logger.info("PASS: CLBP sign component matches plain LBP, output shape is correct")

    # ---- Sanity check: zero amplitudes => no STDP weight change at all ----
    rng_t = np.random.default_rng(0)
    B_t, T_t, Nin_t, Nhid_t = 6, 10, 12, 8
    pre_spikes = (rng_t.random((B_t, T_t, Nin_t)) > 0.5).astype(np.float32)
    post_spikes = (rng_t.random((B_t, T_t, Nhid_t)) > 0.5).astype(np.float32)
    dW_zero = compute_stdp_dW(pre_spikes, post_spikes, a_plus=0.0, a_minus=0.0)
    assert np.allclose(dW_zero, 0.0), "REGRESSION FAIL: zero amplitudes must give zero STDP update!"
    logger.info("PASS: zero-amplitude STDP update is exactly zero")

    # ---- Sanity check: reward=None == reward=ones; reward=zeros => zero dW ----
    dW_default = compute_stdp_dW(pre_spikes, post_spikes, a_plus=1.0, a_minus=1.0, reward=None)
    dW_ones = compute_stdp_dW(pre_spikes, post_spikes, a_plus=1.0, a_minus=1.0,
                               reward=np.ones(B_t, dtype=np.float32))
    assert np.allclose(dW_default, dW_ones), "REGRESSION FAIL: reward=ones should match reward=None!"
    dW_zeros_r = compute_stdp_dW(pre_spikes, post_spikes, a_plus=1.0, a_minus=1.0,
                                  reward=np.zeros(B_t, dtype=np.float32))
    assert np.allclose(dW_zeros_r, 0.0), "REGRESSION FAIL: reward=zeros should give zero dW!"
    logger.info("PASS: reward=None == reward=ones; reward=zeros gives a zero update")

    # ---- Full experiment grid ----
    for encoding, data in [
        ("poisson_only", data_gray),
        ("lbp+poisson", data_gray),
        ("lbp_clbp+poisson", data_gray),
        ("lbp_clbp+poisson", data_color),  # also exercises the color-moments branch
    ]:
        for arch in ["1STDP", "2STDP"]:
            for tau in [2.0, 5.0]:
                for reward_mode in ["none", "binary", "margin"]:
                    res = run_experiment(
                        dataset_name="synthetic", data=data, encoding_type=encoding,
                        architecture=arch, T=25, tau_plus=tau, tau_minus=tau,
                        reward_mode=reward_mode,
                        Nhid=20, n_epochs=2, batch_size=32, show_progress=False,
                    )
                    logger.info(
                        "Self-test OK: encoding=%s arch=%s tau=%.1f reward=%s "
                        "Nin=%d eval_acc=%.3f",
                        encoding, arch, tau, reward_mode,
                        res["Nin"], res["Eval_Accuracy"],
                    )

    logger.info("All self-tests completed successfully.")
