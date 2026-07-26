"""
sadp_core.py
============

Core, reusable building blocks for Supervised Spike Agreement-Dependent
Plasticity (Supervised SADP) experiments.

This module is intentionally self-contained and import-friendly: every
public function/class takes plain numpy arrays in and returns plain numpy
arrays / dataclasses out, with no hidden module-level state. That means you
can drop in a brand-new dataset (any (N, H, W) or (N, H, W, C) array
normalized to [0, 1], plus integer labels) and reuse every piece of the
pipeline - feature extraction, spike encoding, the SADP network, training
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
6. The Supervised SADP network itself (`SADPConfig`, `SADPNetwork`)
   including the k-shifted agreement mechanism
7. Train / evaluate loops
8. High-level `run_experiment` orchestration used by the benchmarking script

Notes on encoding_type
-----------------------
Four encoding options are available:

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
                        parameters before the SADP layer (kept for
                        comparison, since it's exactly what Reviewer #2's
                        objection was about).

LDP, BRIEF, a combined LBP+LDP+BRIEF descriptor, rotation-invariant LBP,
uniform LBP, and multi-block LBP were all also tried at various points
across this project's revisions; none beat plain LBP by enough to justify
the added complexity or cost, except CLBP, which is the one new addition
kept here. A multi-scale ("ML-LBP") variant was also tried and discarded -
it was both the slowest encoding to extract and the least stable to train
(see the project history / prior results tables if you want that
comparison again; it is not implemented in this version).

Notes on the k-shifted agreement
---------------------------------
The original supervised SADP rule compares, at every timestep t, a hidden
neuron's spike s_h(t) against the spike of the correct-class output neuron
s_o*(t), and aggregates this into Cohen's kappa over the whole window T.
That is the d=0 case below.

A small synaptic/axonal transmission delay between the hidden layer and the
read-out layer is biologically unavoidable, so it is natural to ask whether
agreement computed with a small temporal offset d (hidden neuron at time t
vs. output neuron at time t + d) carries additional, complementary signal.
`compute_shifted_kappa` generalizes the original (single, d=0) computation
to an aggregate over d in {-K, ..., -1, 0, 1, ..., K} using only the
overlapping portion of the two spike trains for each offset, exactly as
described by the user. Setting K=0 reduces it identically to the original
rule (this is covered by a regression test in the test-suite at the bottom
of this file). Because K is a small constant independent of dataset size,
the asymptotic linear-time complexity claimed for SADP is preserved: the
update still costs O((2K+1)*T) = O(T) per neuron.

Notes on reward-modulated supervision
--------------------------------------
The k-shifted rule above is still a two-factor (pre-synaptic activity x
post-synaptic agreement) update - it has no notion of whether the network,
as a whole, got the trial right. `SADPConfig.reward_mode` optionally adds
a third, global factor r (a per-sample scalar reuse of quantities the
output layer already computes - no extra forward pass, no change to the
O(T) cost): `binary` is +-1 for correct/incorrect, `margin` is a continuous
(correct_count - best_other_count)/T credit. The hidden-layer update
becomes dW1 ~ pre * kappa * r instead of pre * kappa. `reward_mode="none"`
(the default) skips this multiply entirely, so it is a strictly additive,
opt-in option: every existing default-path result (including the k_shift=0
regression test) is numerically unaffected.

A first comparison run (CIFAR-10, T=50, 1SADP, 10 epochs) suggests k-shift
and reward modulation are not fully independent: looking at per-epoch train
accuracy, the unmodified rule (k_shift=0, reward_mode='none') tends to peak
early and then drift down over the remaining epochs for the non-CNN
encodings, while turning on *either* k-shift or reward modulation prevents
that drift. Combining both rarely beats either one alone. In other words,
the two options may be addressing the same underlying training-stability
issue rather than being purely additive, complementary mechanisms - keep
that in mind when interpreting a result that uses both at once.
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

logger = logging.getLogger("sadp")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


def configure_logging(log_file: Optional[str] = None, level: int = logging.INFO,
                       to_console: bool = True) -> logging.Logger:
    """Attach console / file handlers with timestamps to the `sadp` logger.

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
# the same spatial grid). Still zero learnable parameters - the SADP
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
# 6. The Supervised SADP network (with k-shifted agreement)
# ---------------------------------------------------------------------------

@dataclass
class SADPConfig:
    """All hyperparameters needed to build and train a Supervised SADP net."""
    Nin: int
    Nhid: int
    Nout: int
    architecture: str = "1SADP"
    T: int = 25
    lam: float = 0.9
    theta_h_base: float = 0.5
    theta_o: float = 0.5
    eta_out: float = 5e-4
    eta_in: float = 2e-4
    decay: float = 0.9995
    norm_eps: float = 1e-6
    clip_w2: float = 5.0
    k_shift: int = 0
    weight_by_overlap: bool = False
    kappa_eps: float = 1e-9
    reward_mode: str = "none"
    reward_scale: float = 1.0
    reward_baseline: bool = False
    reward_baseline_decay: float = 0.99
    seed: int = 42

    def __post_init__(self):
        if self.architecture not in ("1SADP", "2SADP"):
            raise ValueError("architecture must be '1SADP' or '2SADP'")
        if self.k_shift < 0:
            raise ValueError("k_shift must be >= 0")
        if self.reward_mode not in ("none", "binary", "margin"):
            raise ValueError("reward_mode must be 'none', 'binary', or 'margin'")
        if self.k_shift >= self.T:
            logger.warning(
                "k_shift=%d >= T=%d; it will be clamped to T-1 at runtime.",
                self.k_shift, self.T
            )


def _cohens_kappa(h_seg: np.ndarray, o_seg: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Chance-corrected agreement (Cohen's kappa). Returns (B, Nh)."""
    agree = np.mean(h_seg == o_seg[:, :, None], axis=1)
    pa = np.mean(h_seg, axis=1)
    pb = np.mean(o_seg, axis=1)[:, None]
    pe = pa * pb + (1.0 - pa) * (1.0 - pb)
    return (agree - pe) / (1.0 - pe + eps)


def _shifted_overlap(spikes_h: np.ndarray, target_spikes: np.ndarray,
                      d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return overlapping (hidden, output) segments for temporal offset d."""
    B, T, Nh = spikes_h.shape
    if d == 0:
        return spikes_h, target_spikes
    elif d > 0:
        return spikes_h[:, :T - d, :], target_spikes[:, d:]
    else:
        dd = -d
        return spikes_h[:, dd:, :], target_spikes[:, :T - dd]


def compute_shifted_kappa(spikes_h: np.ndarray, target_spikes: np.ndarray,
                           k_shift: int = 0, weight_by_overlap: bool = False,
                           eps: float = 1e-9) -> np.ndarray:
    """Aggregate Cohen's kappa over offsets d in {-k_shift, ..., k_shift}.

    k_shift=0 reproduces the original single-lag SADP kappa exactly.
    Returns kappa of shape (B, Nh).
    """
    B, T, Nh = spikes_h.shape
    K_eff = min(k_shift, T - 1)
    if K_eff != k_shift:
        logger.warning("Clamping k_shift from %d to %d given T=%d", k_shift, K_eff, T)

    kappas, weights = [], []
    for d in range(-K_eff, K_eff + 1):
        h_seg, o_seg = _shifted_overlap(spikes_h, target_spikes, d)
        if h_seg.shape[1] == 0:
            continue
        kappas.append(_cohens_kappa(h_seg, o_seg, eps))
        weights.append(h_seg.shape[1])

    kappas_arr = np.stack(kappas, axis=0)
    if weight_by_overlap:
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1, 1)
        return np.sum(kappas_arr * w, axis=0) / np.sum(w)
    return np.mean(kappas_arr, axis=0)


def _compute_reward(out_counts: np.ndarray, preds: np.ndarray, y_batch: np.ndarray,
                     T: int, mode: str = "binary") -> np.ndarray:
    """Global per-sample reward r (B,) for reward-modulated supervision."""
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


class SADPNetwork:
    """Supervised SADP spiking network (1SADP or 2SADP)."""

    def __init__(self, config: SADPConfig):
        self.cfg = config
        rng = np.random.default_rng(config.seed)

        Nin, Nhid, Nout = config.Nin, config.Nhid, config.Nout
        self.W1 = rng.normal(0, 0.1, (Nin, Nhid)).astype(np.float32)
        self.W2 = rng.normal(0, 0.1, (Nhid, Nout)).astype(np.float32)
        self.theta_h = (config.theta_h_base + 0.05 * rng.standard_normal(Nhid)).astype(np.float32)

        if config.architecture == "2SADP":
            self.W1_2 = rng.normal(0, 0.1, (Nhid, Nhid)).astype(np.float32)
            self.theta_h2 = (config.theta_h_base + 0.05 * rng.standard_normal(Nhid)).astype(np.float32)
        else:
            self.W1_2 = None
            self.theta_h2 = None

        self._reward_baseline = 0.0

        logger.info(
            "Initialized SADPNetwork | arch=%s | Nin=%d Nhid=%d Nout=%d | T=%d | k_shift=%d",
            config.architecture, Nin, Nhid, Nout, config.T, config.k_shift
        )

    def forward(self, x_batch_feats: np.ndarray,
                rng: Optional[np.random.Generator] = None
                ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """LIF forward pass over T timesteps."""
        cfg = self.cfg
        B = x_batch_feats.shape[0]
        T = cfg.T

        Vh = np.zeros((B, cfg.Nhid), dtype=np.float32)
        Vh2 = np.zeros((B, cfg.Nhid), dtype=np.float32) if cfg.architecture == "2SADP" else None
        Vo = np.zeros((B, cfg.Nout), dtype=np.float32)

        spikes_h = np.zeros((B, T, cfg.Nhid), dtype=np.float32)
        spikes_h2 = np.zeros((B, T, cfg.Nhid), dtype=np.float32) if cfg.architecture == "2SADP" else None
        spikes_o = np.zeros((B, T, cfg.Nout), dtype=np.float32)

        S_in = poisson_encode_features(x_batch_feats, T, rng=rng)

        for t in range(T):
            I_h = np.einsum("bi,ij->bj", S_in[:, t], self.W1)
            Vh = cfg.lam * Vh + I_h
            spk_h = (Vh > self.theta_h).astype(np.float32)
            Vh[spk_h == 1] = 0.0
            spikes_h[:, t, :] = spk_h

            if cfg.architecture == "2SADP":
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

        return spikes_h, spikes_h2, spikes_o

    def update(self, x_batch_feats: np.ndarray, y_batch: np.ndarray,
               spikes_h: np.ndarray, spikes_h2: Optional[np.ndarray],
               spikes_o: np.ndarray) -> Tuple[np.ndarray, float, Optional[float]]:
        """One mini-batch weight update. Returns (preds, mean_kappa, mean_reward)."""
        cfg = self.cfg
        B, T, _ = spikes_o.shape

        targets = np.zeros((B, cfg.Nout), dtype=np.float32)
        targets[np.arange(B), y_batch] = 1.0

        out_counts = spikes_o.sum(axis=1)
        preds = np.argmax(out_counts, axis=1)

        # Output layer: supervised Hebbian update
        errors = targets[:, None, :] - spikes_o
        pre_out = spikes_h2 if cfg.architecture == "2SADP" else spikes_h
        dW2 = np.einsum("bti,btj->ij", pre_out, errors) / B
        self.W2 += cfg.eta_out * dW2

        # Correct-class output-neuron spike train
        target_spikes = spikes_o[np.arange(B), :, y_batch].astype(np.float32)

        # Optional global reward factor
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

        # W1 (input -> hidden-1): k-shifted, reward-modulated SADP
        kappa1 = compute_shifted_kappa(
            spikes_h, target_spikes, k_shift=cfg.k_shift,
            weight_by_overlap=cfg.weight_by_overlap, eps=cfg.kappa_eps
        )
        kappa1_eff = kappa1 if r is None else kappa1 * r[:, None]
        dW1 = np.einsum("bi,bj->ij", x_batch_feats, kappa1_eff) / B
        self.W1 += cfg.eta_in * dW1

        # W1_2 (hidden-1 -> hidden-2): k-shifted SADP, 2SADP only
        if cfg.architecture == "2SADP":
            kappa2 = compute_shifted_kappa(
                spikes_h2, target_spikes, k_shift=cfg.k_shift,
                weight_by_overlap=cfg.weight_by_overlap, eps=cfg.kappa_eps
            )
            kappa2_eff = kappa2 if r is None else kappa2 * r[:, None]
            dW1_2 = np.einsum("bi,bj->ij", np.mean(spikes_h, axis=1), kappa2_eff) / B
            self.W1_2 += cfg.eta_in * dW1_2
            self.W1_2 *= cfg.decay
            self.W1_2 /= (np.linalg.norm(self.W1_2, axis=0, keepdims=True) + cfg.norm_eps)

        # Decay & normalize
        self.W1 *= cfg.decay
        self.W2 *= cfg.decay
        self.W1 /= (np.linalg.norm(self.W1, axis=0, keepdims=True) + cfg.norm_eps)
        np.clip(self.W2, -cfg.clip_w2, cfg.clip_w2, out=self.W2)

        return preds, float(np.mean(kappa1)), mean_reward


# ---------------------------------------------------------------------------
# 7. Train / evaluate loops
# ---------------------------------------------------------------------------

def train_snn(net: "SADPNetwork", train_feats: np.ndarray, y_train: np.ndarray,
              n_epochs: int = 50, batch_size: int = 128, n_samples: Optional[int] = None,
              seed: Optional[int] = None, show_progress: bool = True,
              eval_feats: Optional[np.ndarray] = None, y_eval: Optional[np.ndarray] = None,
              eval_every: Optional[int] = None, eval_batch_size: int = 256) -> Dict[str, Any]:
    """Train `net` in-place. Returns history dict.

    Weight-stability tracking (always on, negligible cost - O(Nin*Nhid)
    per epoch vs. the O(B*T*Nin*Nhid) cost of the LIF simulation itself
    that already happens every batch):
      epoch_w1_cos_sim_prev : mean cosine similarity of each W1 column to
          its OWN value at the end of the previous epoch (1.0 = direction
          unchanged, 0 = orthogonal/fully rotated, negative = flipped).
          IMPORTANT: W1's *magnitude* is renormalized to ~1 per column
          after every single batch (see SADPNetwork.update), so a naive
          ||W1|| would be trivially ~constant regardless of how stable
          training actually is - this direction metric is the part of
          "weight stability" that ISN'T pinned by that renormalization,
          and is the one actually worth watching for drift/oscillation.
      epoch_w1_frob_norm    : Frobenius norm of W1 - included mainly as a
          sanity check that the per-column renormalization is doing what
          it's supposed to (expect ~sqrt(Nhid), and roughly constant);
          NOT a meaningful stability signal by itself, see above.
      epoch_w2_frob_norm    : Frobenius norm of W2 - W2 is only clipped to
          [-clip_w2, clip_w2], never renormalized, so this one CAN
          genuinely grow, shrink, or oscillate - a real stability signal.

    Optional periodic held-out eval (off by default - costs an extra full
    forward pass over `eval_feats` every `eval_every` epochs, unlike the
    weight diagnostics above this is NOT free):
      pass eval_feats/y_eval and eval_every=K to evaluate every K epochs
      (eval_every=1 for every epoch). Leave eval_every=None (default) to
      skip entirely, matching prior behavior exactly - this is opt-in
      because, unlike the weight tracking, it isn't free.
    """
    rng = np.random.default_rng(seed)
    n_train_total = len(train_feats)
    n_train = n_train_total if n_samples is None else min(n_samples, n_train_total)

    history = {
        "epoch_accuracy": [], "epoch_kappa": [], "epoch_reward": [], "epoch_time_s": [],
        "epoch_w1_cos_sim_prev": [], "epoch_w1_frob_norm": [], "epoch_w2_frob_norm": [],
        "epoch_eval_accuracy": [],
    }
    prev_W1 = net.W1.copy()

    for epoch in range(n_epochs):
        t0 = time.time()
        idx = rng.permutation(n_train_total)[:n_train]
        num_batches = n_train // batch_size

        correct = 0
        kappa_log = []
        reward_log = []

        batch_iter = trange(num_batches, desc=f"Epoch {epoch + 1}/{n_epochs}",
                             disable=not show_progress, leave=False)
        for bi in batch_iter:
            s, e = bi * batch_size, bi * batch_size + batch_size
            batch_idx = idx[s:e]
            Xb = train_feats[batch_idx]
            yb = y_train[batch_idx]

            sh, sh2, so = net.forward(Xb, rng=rng)
            preds, kappa_val, reward_val = net.update(Xb, yb, sh, sh2, so)

            correct += int(np.sum(preds == yb))
            kappa_log.append(kappa_val)
            if reward_val is not None:
                reward_log.append(reward_val)

        epoch_time = time.time() - t0
        acc = correct / float(n_train)
        avg_k = float(np.mean(kappa_log)) if kappa_log else 0.0
        avg_r = float(np.mean(reward_log)) if reward_log else None

        # ---- Weight-stability diagnostics (cheap, always on) ----
        # float64 here on purpose: cos_sim values close to 1.0 are exactly
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
        history["epoch_kappa"].append(avg_k)
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
            "Epoch %d/%d | train_acc=%.4f | avg_kappa=%.6f%s%s | "
            "w1_cos_sim=%.4f w1_norm=%.3f w2_norm=%.3f | time=%.2fs",
            epoch + 1, n_epochs, acc, avg_k,
            f" | avg_reward={avg_r:.4f}" if avg_r is not None else "", eval_log_str,
            w1_cos_sim, w1_frob, w2_frob, epoch_time
        )

    return history


def evaluate_snn(net: "SADPNetwork", test_feats: np.ndarray, y_test: np.ndarray,
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
        sh, sh2, so = net.forward(Xb, rng=rng)
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
    architecture: str = "1SADP",
    T: int = 25,
    k_shift: int = 0,
    weight_by_overlap: bool = False,
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
    """Run one full Supervised-SADP experiment end to end and return a flat
    results dict (ready to be appended to a results table / DataFrame).

    Either pass a built-in `dataset_name` ('mnist' | 'fmnist' | 'cifar10'),
    OR pass your own pre-loaded `data=((x_train,y_train),(x_test,y_test),
    input_shape)` for any other dataset - everything downstream is agnostic
    to where the images came from.

    feature_cache: an optional dict (e.g. shared across many calls from a
        benchmarking grid) used to memoize the extracted features. This
        matters most for 'cnn+poisson': the encoded features (and the CNN
        training that produces them) only depend on
        (dataset, encoding_type, feature_dim, encoder_epochs, seed, ...),
        NOT on architecture/T/k_shift/reward_mode - so when sweeping those,
        the very same features should be reused rather than re-extracted
        (or, worse, the CNN re-trained from scratch) for every grid cell.
        Pass the same dict across calls to get this reuse for free; leave
        it as None to always recompute (e.g. for a single one-off
        experiment).
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
        "T=%d | k_shift=%d", dataset_name.upper(), encoding_type, architecture, T, k_shift
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
            "Reusing cached features | Nin=%d | (arch/T/k_shift/reward_mode do not affect encoding)",
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

    cfg = SADPConfig(
        Nin=train_feats.shape[1], Nhid=Nhid, Nout=n_classes,
        architecture=architecture, T=T, theta_h_base=theta_h_base,
        eta_in=eta_in, eta_out=eta_out,
        k_shift=k_shift, weight_by_overlap=weight_by_overlap,
        reward_mode=reward_mode, reward_scale=reward_scale,
        reward_baseline=reward_baseline, reward_baseline_decay=reward_baseline_decay,
        seed=seed,
    )
    net = SADPNetwork(cfg)

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
        "K_shift": k_shift,
        "Weight_by_overlap": weight_by_overlap,
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
        "Epoch_Kappas": history["epoch_kappa"],
        "Epoch_Rewards": history["epoch_reward"],
        "Epoch_W1_CosSim_Prev": history["epoch_w1_cos_sim_prev"],
        "Epoch_W1_FrobNorm": history["epoch_w1_frob_norm"],
        "Epoch_W2_FrobNorm": history["epoch_w2_frob_norm"],
        "Epoch_Eval_Accuracies": history["epoch_eval_accuracy"],
    }
    logger.info(
        "RESULT | %s | %s | %s | T=%d | k=%d | reward=%s -> eval_acc=%.4f f1=%.4f (%.2fs/epoch)",
        result["Dataset"], encoding_type, architecture, T, k_shift, reward_mode,
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
    logger.info("Running sadp_core.py self-test on synthetic data.")

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

    # ---- Full experiment grid ----
    for encoding, data in [
        ("poisson_only", data_gray),
        ("lbp+poisson", data_gray),
        ("lbp_clbp+poisson", data_gray),
        ("lbp_clbp+poisson", data_color),  # also exercises the color-moments branch
    ]:
        for arch in ["1SADP", "2SADP"]:
            for k in [0, 5]:
                for reward_mode in ["none", "binary", "margin"]:
                    res = run_experiment(
                        dataset_name="synthetic", data=data, encoding_type=encoding,
                        architecture=arch, T=25, k_shift=k, reward_mode=reward_mode,
                        Nhid=20, n_epochs=2, batch_size=32, show_progress=False,
                    )
                    logger.info(
                        "Self-test OK: encoding=%s arch=%s k=%d reward=%s "
                        "Nin=%d eval_acc=%.3f",
                        encoding, arch, k, reward_mode,
                        res["Nin"], res["Eval_Accuracy"],
                    )

    logger.info("All self-tests completed successfully.")
