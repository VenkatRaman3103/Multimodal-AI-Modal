"""Model definitions for different modalities."""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def process(self, data: Any) -> Dict[str, Any]:
        """Process input data and return results."""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model."""
        pass


class TextModel(BaseModel):
    """Text processing model."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        super().__init__(model_name)
        self.tokenizer = None
        self.model = None
    
    def load_model(self) -> None:
        """Load the text model and tokenizer."""
        try:
            import transformers
            self.model = transformers.AutoModel.from_pretrained(self.model_name)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
        except ImportError:
            raise ImportError("transformers library is required for TextModel")
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process text input."""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return {
            "embeddings": outputs.last_hidden_state.cpu().numpy(),
            "input_text": text,
            "model_used": self.model_name
        }


class ImageModel(BaseModel):
    """Image processing model."""
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        super().__init__(model_name)
        self.processor = None
        self.model = None
    
    def load_model(self) -> None:
        """Load the image model and processor."""
        try:
            import transformers
            self.model = transformers.AutoModel.from_pretrained(self.model_name)
            self.processor = transformers.AutoImageProcessor.from_pretrained(self.model_name)
            self.model.to(self.device)
        except ImportError:
            raise ImportError("transformers library is required for ImageModel")
    
    def process(self, image) -> Dict[str, Any]:
        """Process image input."""
        if self.model is None or self.processor is None:
            self.load_model()
        
        # Handle both file paths and PIL Images
        if isinstance(image, str):
            from PIL import Image as PILImage
            image = PILImage.open(image)
        
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return {
            "embeddings": outputs.last_hidden_state.cpu().numpy(),
            "image_size": image.size,
            "model_used": self.model_name
        }


class AudioModel(BaseModel):
    """Audio processing model."""
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        super().__init__(model_name)
        self.processor = None
        self.model = None
    
    def load_model(self) -> None:
        """Load the audio model and processor."""
        try:
            import transformers
            self.model = transformers.AutoModel.from_pretrained(self.model_name)
            self.processor = transformers.AutoProcessor.from_pretrained(self.model_name)
            self.model.to(self.device)
        except ImportError:
            raise ImportError("transformers library is required for AudioModel")
    
    def process(self, audio) -> Dict[str, Any]:
        """Process audio input."""
        if self.model is None or self.processor is None:
            self.load_model()
        
        # Handle different audio input formats
        if isinstance(audio, str):
            import librosa
            audio, sr = librosa.load(audio, sr=16000)
        elif isinstance(audio, tuple):
            audio, sr = audio
        else:
            sr = 16000  # Default sample rate
        
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return {
            "embeddings": outputs.last_hidden_state.cpu().numpy(),
            "sample_rate": sr,
            "model_used": self.model_name
        }