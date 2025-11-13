from typing import override

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms as T

from src.config import Config
from src.heads.head import HeadOutput
from src.model.base import BaseDeepakeDetectionModel, OutputsForMetrics
from src.model.forada.ds import DS
from src.utils import logger


class ForAda(BaseDeepakeDetectionModel):
    def __init__(self, config: Config):
        super().__init__(config, verbose=True)

        # load yaml file relative to the current file
        config_path = __file__.replace("forensics_adapter.py", "forensics_adapter_model/config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.model = DS(
            clip_name=config["clip_model_name"],
            adapter_vit_name=config["vit_name"],
            num_quires=config["num_quires"],
            fusion_map=config["fusion_map"],
            mlp_dim=config["mlp_dim"],
            mlp_out_dim=config["mlp_out_dim"],
            head_num=config["head_num"],
        )
        self.eval()

    @override
    def forward(self, inputs: torch.Tensor) -> HeadOutput:
        outputs = self.model({"image": inputs}, inference=True)
        return HeadOutput(logits_labels=outputs["logits"])

    @override
    def on_test_epoch_start(self):
        self.test_step_outputs = OutputsForMetrics()
        # move model to the device
        self.model.to(self.trainer.strategy.root_device)

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
        logger.print_info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        self.print_checkpoint_keys(incompatible_keys)

    @override
    def get_preprocessing(self):
        def preprocess(image: Image) -> torch.Tensor:
            return preprocessing(image)

        return preprocess


_preprocess = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


def preprocessing(image: Image) -> torch.Tensor:
    image = np.array(image)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype=np.uint8)
    image = _preprocess(image)
    # image = F.interpolate(
    #     image.unsqueeze(0),
    #     size=(224, 224),
    #     mode="bilinear",
    #     align_corners=False,
    # )[0]
    return image


if __name__ == "__main__":
    #! Run as module:
    #! python -m src.model.forensics_adapter

    from PIL import Image

    from src.config import Config
    from src.model.ForAda import ForAda

    config = Config()
    model = ForAda(config)

    model.load_checkpoint("weights/forensics_adapter/ForensicsAdapter.pth")

    path = "datasets/FF/real/000/000.png"
    image = Image.open(path)  # Load image
    preprocessed_image = model.get_preprocessing()(image)  # Convert to tensor
    batch = preprocessed_image.unsqueeze(0)  # Add batch dimension
    outputs = model(batch)

    print(outputs.logits_labels)  # Print logits labels
    print(outputs.logits_labels.softmax(dim=1))  # Print probabilities
