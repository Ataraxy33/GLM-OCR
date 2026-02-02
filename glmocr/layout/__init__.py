"""Layout detection module."""

from .base import BaseLayoutDetector

try:
    from .layout_detector import PPDocLayoutDetector
except Exception as e:  # pragma: no cover
    PPDocLayoutDetector = None  # type: ignore
    _layout_import_error = e


def _raise_layout_import_error() -> None:
    raise ImportError(
        "Layout detection dependencies are not installed. "
        "Install with: pip install glmocr[layout]"
    ) from _layout_import_error


__all__ = ["BaseLayoutDetector", "PPDocLayoutDetector"]
