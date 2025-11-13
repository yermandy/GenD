from typing import override

import torch
import torchvision.transforms as T
from PIL import Image

from src.config import Config
from src.heads.head import HeadOutput
from src.model.base import BaseDeepakeDetectionModel, OutputsForMetrics
from src.model.effort.model import EffortModel
from src.utils import logger

preprocessing_alternative = T.Compose(
    [
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


class Effort(BaseDeepakeDetectionModel):
    def __init__(self, config: Config):
        super().__init__(config, verbose=True)
        self.detector = EffortModel()
        self.test_step_outputs = OutputsForMetrics()

        self.detector.eval()

    @override
    def forward(self, inputs: torch.Tensor) -> HeadOutput:
        logits, l2_embeddings = self.detector(inputs)
        return HeadOutput(logits_labels=logits, l2_embeddings=l2_embeddings)

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
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        incompatible_keys = self.detector.load_state_dict(state_dict, strict=False)
        self.print_checkpoint_keys(incompatible_keys)

    @override
    def get_preprocessing(self):
        def preprocess(image: Image) -> torch.Tensor:
            return preprocessing_alternative(image)

        return preprocess


if __name__ == "__main__":
    # Example usage
    model = Effort()
    print(model)

    model.load_checkpoint("weights/effort/effort_clip_L14_trainOn_FaceForensic.pth")

    image = Image.open("datasets/FF/real/000/000.png")
    tensor = preprocessing_alternative(image).unsqueeze(0)  # Add batch dimension
    outputs = model({"image": tensor})

    print(outputs)
