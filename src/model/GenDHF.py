import os
from typing import override

import torch
import torchvision.transforms as T
from PIL import Image

from src.config import Config, CustomPreprocessing
from src.heads.head import HeadOutput
from src.hf.modeling_gend import GenD
from src.model.base import BaseDeepakeDetectionModel
from src.utils import logger


class GenDHF(BaseDeepakeDetectionModel):
    def __init__(self, config: Config):
        super().__init__(config, verbose=True)
        self.model = GenD.from_pretrained(config.checkpoint)
        self.model.eval()

    @override
    def forward(self, inputs: torch.Tensor) -> HeadOutput:
        return HeadOutput(logits_labels=self.model(inputs))

    @override
    def test_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        outputs = self.forward(batch.images)
        probs = outputs.logits_labels.softmax(dim=1)

        # Save outputs for metrics calculation
        self.test_step_outputs.labels.update(batch.labels)
        self.test_step_outputs.probs.update(probs.detach())
        self.test_step_outputs.idx.update(batch.idx)

    @override
    def load_checkpoint(self, checkpoint_path: str):
        """Load the model checkpoint."""
        pass  # Handled by from_pretrained

    @override
    def get_preprocessing(self):
        return self.model.feature_extractor.preprocess


if __name__ == "__main__":
    config = Config(
        checkpoint="yermandy/GenD_CLIP_L_14",
    )
    model = GenDHF(config)
    model.load_checkpoint(config.checkpoint)

    image = Image.open("datasets/FF/DF/001_870/000.png")
    # image = Image.open("datasets/FF/real/001/000.png")
    preprocessed_image = model.get_preprocessing()(image)  # Convert to tensor
    batch = preprocessed_image.unsqueeze(0)  # Add batch dimension
    outputs = model.forward(batch)
    print(outputs.logits_labels.softmax(dim=-1))
