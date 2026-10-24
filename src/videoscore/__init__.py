"""videoscore: background music for videos, synced to scene transitions."""

from .engine import plan, render
from .spec import SpecError
from .styles import STYLES

__version__ = "0.1.0"
__all__ = ["render", "plan", "SpecError", "STYLES", "__version__"]
