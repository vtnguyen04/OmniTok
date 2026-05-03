"""Teacher models — frozen VFM wrappers for alignment."""

# Import to trigger registration
from . import (
    depth_anything,  # noqa: F401
    dinov2,  # noqa: F401
    hog,  # noqa: F401
    sam,  # noqa: F401
    siglip,  # noqa: F401
)
from .base import BaseTeacher
from .depth_anything import DepthAnythingTeacher
from .dinov2 import DINOv2Teacher
from .hog import HOGTeacher
from .multi_teacher import MultiTeacher
from .normalizer import FeatureNormalizer, ProjectedNormalizer
from .sam import SAMTeacher
from .siglip import SigLIPTeacher

__all__ = [
    "BaseTeacher",
    "MultiTeacher",
    "DINOv2Teacher",
    "SigLIPTeacher",
    "SAMTeacher",
    "DepthAnythingTeacher",
    "HOGTeacher",
]
