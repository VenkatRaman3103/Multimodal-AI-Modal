"""Basic tests for the multimodal AI modal."""

import pytest
import tempfile
import os
from pathlib import Path

from multimodal_ai_modal.core import MultimodalProcessor
from multimodal_ai_modal.models import TextModel, ImageModel, AudioModel


class TestMultimodalProcessor:
    """Test cases for MultimodalProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = MultimodalProcessor()
    
    def test_init(self):
        """Test processor initialization."""
        assert self.processor.model_name == "microsoft/DialoGPT-medium"
        assert hasattr(self.processor, 'device')
        assert hasattr(self.processor, '_models')
        assert hasattr(self.processor, '_processors')
    
    def test_process_text_basic(self):
        """Test basic text processing."""
        text = "Hello, world!"
        result = self.processor.process_text(text)
        
        assert "embeddings" in result
        assert "input_text" in result
        assert "model_used" in result
        assert result["input_text"] == text
    
    def test_process_multimodal_text_only(self):
        """Test processing multimodal data with text only."""
        data = {"text": "Test input"}
        result = self.processor.process_multimodal(data)
        
        assert "text" in result
        assert "embeddings" in result["text"]
        assert result["text"]["input_text"] == "Test input"


class TestModels:
    """Test cases for individual model classes."""
    
    def test_text_model_init(self):
        """Test TextModel initialization."""
        model = TextModel()
        assert model.model_name == "microsoft/DialoGPT-medium"
        assert hasattr(model, 'device')
    
    def test_image_model_init(self):
        """Test ImageModel initialization."""
        model = ImageModel()
        assert model.model_name == "google/vit-base-patch16-224"
        assert hasattr(model, 'device')
    
    def test_audio_model_init(self):
        """Test AudioModel initialization."""
        model = AudioModel()
        assert model.model_name == "facebook/wav2vec2-base"
        assert hasattr(model, 'device')


class TestCLI:
    """Test cases for CLI functionality."""
    
    def test_cli_help(self):
        """Test CLI help functionality."""
        from multimodal_ai_modal.cli import main
        import sys
        from unittest.mock import patch
        
        with patch.object(sys, 'argv', ['multimodal-ai', '--help']):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0


if __name__ == "__main__":
    pytest.main([__file__])