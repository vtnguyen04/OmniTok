"""Teacher models — frozen VFM wrappers for alignment."""

from .base import BaseTeacher
from .normalizer import FeatureNormalizer, ProjectedNormalizer
from .multi_teacher import MultiTeacher

# Import to trigger registration
from . import dinov2  # noqa: F401
from . import siglip  # noqa: F401
