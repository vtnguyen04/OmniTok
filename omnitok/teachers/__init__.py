"""Teacher models — frozen VFM wrappers for alignment."""

# Import to trigger registration
from . import (
    depth_anything,  # noqa: F401
    dinov2,  # noqa: F401
    sam,  # noqa: F401
    siglip,  # noqa: F401
)
from .base import BaseTeacher
from .multi_teacher import MultiTeacher
from .normalizer import FeatureNormalizer, ProjectedNormalizer

__all__ = ["BaseTeacher", "MultiTeacher", "dinov2", "siglip", "sam", "depth_anything"]
