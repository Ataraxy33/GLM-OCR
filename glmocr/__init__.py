"""GlmOcr - Document Parsing with GLM OCR.

Document parsing service that supports layout detection and text recognition.
"""

__version__ = "0.1.1"
__author__ = "ZHIPUAI"

# Import main components
from . import dataloader
from . import layout
from . import postprocess
from . import utils
from .pipeline import Pipeline
from .config import GlmOcrConfig, load_config
from .parser_result import PipelineResult

# Import API
from .api import GlmOcr, parse

__all__ = [
    "dataloader",
    "layout",
    "postprocess",
    "utils",
    "Pipeline",
    "PipelineResult",
    "GlmOcrConfig",
    "load_config",
    "GlmOcr",
    "parse",
]
