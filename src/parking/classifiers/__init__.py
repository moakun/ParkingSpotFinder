"""Occupied/empty classifiers (§4). The model layer is pluggable — every
classifier implements :class:`~parking.classifiers.base.Classifier`.

Build order per the plan: classical floor -> small CNN. A drop-in for a
custom backbone (e.g. MEISCF) only has to satisfy the same interface.
"""

from parking.classifiers.base import Classifier
from parking.classifiers.classical import EdgeDensityClassifier

__all__ = ["Classifier", "EdgeDensityClassifier", "build_classifier"]


def build_classifier(name: str, **kwargs) -> Classifier:
    """Factory used by the CLI. Keeps torch import lazy so the classical
    baseline runs without a torch install."""
    name = name.lower()
    if name in {"classical", "edge", "baseline"}:
        return EdgeDensityClassifier(**kwargs)
    if name in {"cnn", "mobilenet", "mobilenetv3"}:
        from parking.classifiers.cnn import CNNClassifier

        return CNNClassifier(**kwargs)
    raise ValueError(f"Unknown classifier {name!r} (have: classical, cnn)")
