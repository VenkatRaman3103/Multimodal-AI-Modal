"""Core multimodal processing functionality."""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import transformers


class MultimodalProcessor:
    """Main class for processing multimodal data."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize the multimodal processor.
        
        Args:
            model_name: Name of the primary model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models = {}
        self._processors = {}
    
    def load_text_model(self, model_name: Optional[str] = None) -> None:
        """Load a text processing model."""
        name = model_name or self.model_name
        if "text" not in self._models:
            self._models["text"] = transformers.AutoModel.from_pretrained(name)
            self._processors["text"] = transformers.AutoTokenizer.from_pretrained(name)
            self._models["text"].to(self.device)
    
    def load_image_model(self, model_name: str = "google/vit-base-patch16-224") -> None:
        """Load an image processing model."""
        if "image" not in self._models:
            self._models["image"] = transformers.AutoModel.from_pretrained(model_name)
            self._processors["image"] = transformers.AutoImageProcessor.from_pretrained(model_name)
            self._models["image"].to(self.device)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text input."""
        if "text" not in self._models:
            self.load_text_model()
        
        tokenizer = self._processors["text"]
        model = self._models["text"]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        return {
            "embeddings": outputs.last_hidden_state.cpu().numpy(),
            "input_text": text,
            "model_used": self.model_name
        }
    
    def process_image(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        """Process image input."""
        if "image" not in self._models:
            self.load_image_model()
        
        if isinstance(image, str):
            image = Image.open(image)
        
        processor = self._processors["image"]
        model = self._models["image"]
        
        inputs = processor(images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        return {
            "embeddings": outputs.last_hidden_state.cpu().numpy(),
            "image_size": image.size,
            "model_used": "google/vit-base-patch16-224"
        }
    
    def process_multimodal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process multiple data types together."""
        results = {}
        
        if "text" in data:
            results["text"] = self.process_text(data["text"])
        
        if "image" in data:
            results["image"] = self.process_image(data["image"])
        
        return results