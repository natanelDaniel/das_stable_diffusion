"""
diffusion_eval.py — evaluation harness for the class + alpha-conditional sampler.

The two primary evaluations:

  * class-recoverability: for each class c, generate K clean (alpha=0) patches and
    run a held-out DAS classifier on them. Returns per-class top-1 accuracy.
  * alpha sweep: for a chosen event class, generate K patches at each alpha in a
    sweep (e.g. {0, 0.25, 0.5, 0.75, 1.0}). Return the mean classifier probability
    on the event class; expect monotonic decrease as alpha grows.

The classifier is a held-out object: pass it as a callable so this module stays
decoupled from specific checkpoints. The existing `src/evaluation/classifier.py`
DASClassifier is for the legacy 32x256 patch geometry; a re-trained classifier on
the new [1, 8, 16384] geometry is required before these checks produce meaningful
numbers.
"""

from typing import Callable, Dict, Iterable, Sequence

import numpy as np
import torch


GeneratorFn = Callable[[str, float, int], np.ndarray]
"""Signature: (class_name, alpha, n_samples) -> patches [n_samples, 1, 8, 16384]."""

ClassifierFn = Callable[[np.ndarray], np.ndarray]
"""Signature: (patches [N, 1, 8, 16384]) -> probs [N, num_classes]."""


def recoverability(
    generator_fn: GeneratorFn,
    classifier_fn: ClassifierFn,
    classes: Sequence[str],
    n_per_class: int = 64,
) -> Dict[str, float]:
    """For each class c, generate `n_per_class` clean (alpha=0) patches; return top-1 accuracy."""
    out: Dict[str, float] = {}
    for c_idx, c in enumerate(classes):
        patches = generator_fn(c, 0.0, n_per_class)
        probs = classifier_fn(patches)
        top1 = probs.argmax(axis=1)
        out[c] = float((top1 == c_idx).mean())
    return out


def alpha_sweep_curve(
    generator_fn: GeneratorFn,
    classifier_fn: ClassifierFn,
    classes: Sequence[str],
    event_class: str,
    alphas: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    n_per: int = 32,
) -> Dict[float, float]:
    """For each alpha, generate `n_per` patches of `event_class` at that alpha.

    Returns {alpha: mean classifier probability on `event_class`}. Expect this to
    monotonically decrease as alpha grows from 0 to 1.
    """
    if event_class not in classes:
        raise ValueError(f"event_class '{event_class}' not in classes {list(classes)}")
    e_idx = list(classes).index(event_class)
    out: Dict[float, float] = {}
    for alpha in alphas:
        alpha = float(alpha)
        patches = generator_fn(event_class, alpha, n_per)
        probs = classifier_fn(patches)
        out[alpha] = float(probs[:, e_idx].mean())
    return out


def is_monotonic_decreasing(values: Sequence[float], tol: float = 0.05) -> bool:
    """Check that values decrease (within `tol` slack to permit small noise)."""
    for a, b in zip(values, values[1:]):
        if b > a + tol:
            return False
    return True


def patches_to_classifier_input(patches: np.ndarray) -> torch.Tensor:
    """Pass-through helper. patches: [N, 1, 8, 16384] float32 -> tensor."""
    return torch.from_numpy(patches.astype(np.float32))
