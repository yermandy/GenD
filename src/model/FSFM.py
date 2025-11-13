import os
from typing import override

import torch
import torchvision.transforms as T
from PIL import Image

from src.config import Config, CustomPreprocessing
from src.heads.head import HeadOutput
from src.model.base import BaseDeepakeDetectionModel
from src.model.fsfm import models_vit, models_vit_fs_adapter
from src.utils import logger


def download_model_if_needed(checkpoint_path: str, link: str):
    if not os.path.exists(checkpoint_path):
        logger.print_warning_once(f"Checkpoint '{checkpoint_path}' not found, downloading...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        os.system(f"wget {link} -O {checkpoint_path}")


class FSFM(BaseDeepakeDetectionModel):
    def __init__(self, config: Config):
        super().__init__(config, verbose=True)
        self.initialize_model(config.checkpoint)
        self.model.eval()

    def initialize_model(self, checkpoint_path: str):
        if checkpoint_path == "weights/FS-VFM/FS-VFM-ViT-L-Adapter.pth":
            link = "https://hf.co/Wolowolo/fsfm-3c/resolve/main/finetuned_models/FS-VFM_extensions/finetune_fs-adapter/cross_dataset_DfD_and_DiFF/ViT-L_VF2_600e/FT_on_FF%2B%2B_c23_32frames/checkpoint-min_val_loss.pth?download=true"
            download_model_if_needed(checkpoint_path, link)
            self.model = models_vit_fs_adapter.vit_large_patch16(num_classes=2, drop_path_rate=0.1, global_pool=True)

        elif checkpoint_path == "weights/FS-VFM/FS-VFM-ViT-L.pth":
            link = "https://hf.co/Wolowolo/fsfm-3c/resolve/main/finetuned_models/FS-VFM_extensions/cross_dataset_DFD_and_DiFF/ViT-L_VF2_600e/FT_on_FF%2B%2B_c23_32frames/checkpoint-min_val_loss.pth?download=true"
            download_model_if_needed(checkpoint_path, link)
            self.model = models_vit.vit_large_patch16(
                num_classes=2,
                drop_path_rate=0.1,
                global_pool=True,
            )

        else:
            raise ValueError(f"Unknown FS-VFM checkpoint path: {checkpoint_path}")

    @override
    def forward(self, inputs: torch.Tensor) -> HeadOutput:
        outputs = self.model(inputs)
        outputs = outputs[..., [1, 0]]  # Swap 0 and 1 rows to have [real, fake]
        return HeadOutput(logits_labels=outputs)

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
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        incompatible_keys = self.model.load_state_dict(checkpoint["model"], strict=False)
        self.print_checkpoint_keys(incompatible_keys)

    @override
    def get_preprocessing(self):
        if self.config.custom_preprocessing is None:
            logger.print_warning_once("This model might expect a zoom in to the facial image. Make sure to tune it.")

        def preprocess(image: Image) -> torch.Tensor:
            image = self.custom_preprocessing(image)
            return transform(image)

        return preprocess


transform = T.Compose(
    [
        T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(
            [0.5482207536697388, 0.42340534925460815, 0.3654651641845703],
            [0.2789176106452942, 0.2438540756702423, 0.23493893444538116],
        ),
    ]
)


if __name__ == "__main__":
    config = Config(
        checkpoint="weights/FS-VFM/FS-VFM-ViT-L.pth",
        custom_preprocessing=CustomPreprocessing(zoom_factor=1.3),
    )
    model = FSFM(config)
    model.load_checkpoint(config.checkpoint)

    image = Image.open("datasets/FF/DF/001_870/000.png")
    # image = Image.open("datasets/FF/real/001/000.png")
    preprocessed_image = model.get_preprocessing()(image)  # Convert to tensor
    batch = preprocessed_image.unsqueeze(0)  # Add batch dimension
    outputs = model.forward(batch)
    print(outputs.logits_labels.softmax(dim=-1))
