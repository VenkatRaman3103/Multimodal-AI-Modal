"""Multimodal AI Modal - A framework for processing text, images, and other data types."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import MultimodalProcessor
from .models import TextModel, ImageModel, AudioModel

__all__ = [
    "MultimodalProcessor",
    "TextModel", 
    "ImageModel",
    "AudioModel",
]