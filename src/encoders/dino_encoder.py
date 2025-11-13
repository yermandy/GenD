import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, Dinov2Model, Dinov2WithRegistersModel


class DINOEncoder(nn.Module):
    def __init__(
        self, model_name="facebook/dinov2-with-registers-base", merge_cls_token_with_patches: None | str = None
    ):
        """
        See models in src/config.py
        """

        super().__init__()

        self._preprocess = AutoImageProcessor.from_pretrained(model_name)
        self.backbone: Dinov2Model | Dinov2WithRegistersModel = AutoModel.from_pretrained(model_name)
        self.merge_cls_token_with_patches = merge_cls_token_with_patches

        self.features_dim = self.backbone.config.hidden_size
        if self.merge_cls_token_with_patches == "cat":
            self.features_dim *= 2

        self.merge_cls_token_with_patches = merge_cls_token_with_patches

    def preprocess(self, image: Image) -> torch.Tensor:
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(inputs)

        cls_token = outputs.last_hidden_state[:, 0]

        if self.merge_cls_token_with_patches is None:
            embeddings = cls_token
        elif self.merge_cls_token_with_patches == "cat":
            patches = outputs.last_hidden_state[:, -16 * 16 :].mean(dim=1)
            embeddings = torch.cat([cls_token, patches], dim=1)
        elif self.merge_cls_token_with_patches == "mean":
            patches = outputs.last_hidden_state[:, -16 * 16 :].mean(dim=1)
            embeddings = (cls_token + patches) / 2
        else:
            raise ValueError(f"Unknown merge_cls_token_with_patches strategy: {self.merge_cls_token_with_patches}")

        return embeddings

    def get_features_dim(self) -> int:
        return self.features_dim


if __name__ == "__main__":
    import autorootcwd  # noqa: F401

    from src.config import Backbone
    from src.encoders._common import inference

    model = DINOEncoder(Backbone.DINOv3_ViT_B.value, None)
    inference(model)
