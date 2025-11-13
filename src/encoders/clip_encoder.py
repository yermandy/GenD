import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        """
        Models:
        1. openai/clip-vit-base-patch16 | 768 features
        2. openai/clip-vit-base-patch32 | 768 features
        3. openai/clip-vit-large-patch14 | 1024 features

        See more in src/config.py
        """

        super().__init__()

        try:
            self._preprocess = CLIPProcessor.from_pretrained(model_name)
        except Exception:
            self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        clip: CLIPModel = CLIPModel.from_pretrained(model_name)

        # take vision model from CLIP, maps image to vision_embed_dim
        self.vision_model = clip.vision_model

        self.model_name = model_name

        self.features_dim = self.vision_model.config.hidden_size

        # take visual_projection, maps vision_embed_dim to projection_dim
        self.visual_projection = clip.visual_projection

    def preprocess(self, image: Image) -> torch.Tensor:
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, preprocessed_images: torch.Tensor) -> torch.Tensor:
        return self.vision_model(preprocessed_images).pooler_output

    def get_features_dim(self):
        return self.features_dim


if __name__ == "__main__":
    import autorootcwd  # noqa: F401

    from src.config import Backbone
    from src.encoders._common import inference

    model = CLIPEncoder(Backbone.CLIP_B_16.value)
    inference(model)
