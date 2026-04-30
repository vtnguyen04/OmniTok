"""Loss modules for OmniTok training."""

from .reconstruction import ReconstructionLoss
from .kl import KLLoss
from .gan import GANLoss

# Import alignment subpackage to trigger registrations
from . import alignment  # noqa: F401
