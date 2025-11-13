import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import PretrainedConfig, PreTrainedModel


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes, normalize_inputs=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.normalize_inputs = normalize_inputs

    def forward(self, x: torch.Tensor, **kwargs):
        if self.normalize_inputs:
            x = F.normalize(x, p=2, dim=1)

        return self.linear(x)


class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super().__init__()

        from transformers import CLIPModel, CLIPProcessor

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


class DINOEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-with-registers-base"):
        super().__init__()

        from transformers import AutoImageProcessor, AutoModel, Dinov2Model, Dinov2WithRegistersModel

        self._preprocess = AutoImageProcessor.from_pretrained(model_name)
        self.backbone: Dinov2Model | Dinov2WithRegistersModel = AutoModel.from_pretrained(model_name)

        self.features_dim = self.backbone.config.hidden_size

    def preprocess(self, image: Image) -> torch.Tensor:
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs).last_hidden_state[:, 0]

    def get_features_dim(self) -> int:
        return self.features_dim


class PerceptionEncoder(nn.Module):
    def __init__(self, model_name="vit_pe_core_large_patch14_336"):
        super().__init__()

        import timm
        from timm.models.eva import Eva

        self.backbone: Eva = timm.create_model(
            model_name,
            pretrained=True,
            dynamic_img_size=True,
        )

        # Get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.backbone)
        data_config["input_size"] = (3, 224, 224)

        self._preprocess = timm.data.create_transform(**data_config, is_training=False)

        # Remove head
        self.backbone.head = nn.Identity()

        self.features_dim = self.backbone.num_features

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

    def get_features_dim(self) -> int:
        return self.features_dim


class GenDConfig(PretrainedConfig):
    model_type = "GenD"

    def __init__(self, backbone: str = "openai/clip-vit-large-patch14", head: str = "linear", **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.head = head


class GenD(PreTrainedModel):
    config_class = GenDConfig

    def __init__(self, config):
        super().__init__(config)

        self.head = config.head
        self.backbone = config.backbone
        self.config = config

        self._init_feature_extractor()
        self._init_head()

    def _init_feature_extractor(self):
        backbone = self.backbone
        backbone_lowercase = backbone.lower()

        if "clip" in backbone_lowercase:
            self.feature_extractor = CLIPEncoder(backbone)

        elif "vit_pe" in backbone_lowercase:
            self.feature_extractor = PerceptionEncoder(backbone)

        elif "dino" in backbone_lowercase:
            self.feature_extractor = DINOEncoder(backbone)

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _init_head(self):
        features_dim = self.feature_extractor.get_features_dim()

        match self.head:
            case "linear":
                self.model = LinearProbe(features_dim, 2)

            case "LinearNorm":
                self.model = LinearProbe(features_dim, 2, True)

            case _:
                raise ValueError(f"Unknown head: {self.head}")

    def forward(self, inputs: torch.Tensor):
        features = self.feature_extractor(inputs)
        outputs = self.model.forward(features)
        return outputs
