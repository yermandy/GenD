from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel as Validation
from pydantic import field_validator

Scheduler = Literal[
    "cosine",  # CosineAnnealingLR
    "cyclic",  # CosineAnnealingWarmRestarts
]

Precision = Literal[
    16,
    32,
    64,
    "16",
    "16-true",
    "16-mixed",
    "bf16-true",
    "bf16-mixed",
    "32",
    "32-true",
    "64",
    "64-true",
]


class ValidateEnum(str, Enum):
    @classmethod
    def get_all_values(cls) -> list[str]:
        return [value.value for value in cls]

    @classmethod
    def validate(cls, value: str) -> str:
        values = cls.get_all_values()
        if value not in values:
            raise ValueError(f"\n\nInvalid value: '{value}'\n\nPossible values are: {values}\n\nSee {__file__}\n\n")
        return value


class Optimizer(ValidateEnum):
    AdamW = "AdamW"
    SGD = "SGD"


class InferenceStrategy(ValidateEnum):
    SOFTMAX = "softmax"


class Head(ValidateEnum):
    Linear = "linear"
    NLinear = "LinearNorm"


class Backbone(ValidateEnum):
    # https://hf.co/docs/transformers/en/model_doc/clip
    # https://hf.co/openai/models?search=clip
    CLIP_B_16 = "openai/clip-vit-base-patch16"
    CLIP_B_32 = "openai/clip-vit-base-patch32"
    CLIP_L_14 = "openai/clip-vit-large-patch14"
    CLIP_L_14_336 = "openai/clip-vit-large-patch14-336"

    # https://hf.co/collections/facebook/perception-encoder-67f977c9a65ca5895a7f6ba1
    PerceptionEncoder_B_p16_224 = "vit_pe_core_base_patch16_224"  # (from timm)
    PerceptionEncoder_L_p14_336 = "vit_pe_core_large_patch14_336"  # (from timm)
    PerceptionEncoder_G_p14_448 = "vit_pe_core_gigantic_patch14_448"  # (from timm)

    # https://hf.co/models?search=facebook/dinov3
    DINOv3_ViT_B = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    DINOv3_ViT_L = "facebook/dinov3-vitl16-pretrain-lvd1689m"


class BackboneArgs(Validation, validate_assignment=True):
    img_size: None | int = 224  # Image size for the backbone
    merge_cls_token_with_patches: None | Literal["cat", "mean"] = None  # Concatenate CLS token with patches


class Loss(Validation, validate_assignment=True):
    # Cross-entropy loss (multi-class classification)
    ce_labels: float = 0.0  # Loss weight
    label_smoothing: float = 0.0  # Loss weight
    # Uniformity and alignment loss
    uniformity: float = 0.0  # Loss weight
    alignment_labels: float = 0.0  # Loss weight


class LoRA(Validation, validate_assignment=True):
    target_modules: list[str] | str = ["out_proj"]  # Target modules
    rank: int = 1  # Rank of the decomposition
    alpha: int = 32  # Scaling factor
    dropout: float = 0.05  # Dropout probability
    bias: str = "none"  # Bias configuration
    use_rslora: bool = False  # Use rsLoRA
    use_dora: bool = False  # Use DoRA


class PEFT(Validation, validate_assignment=True):
    lora: None | LoRA = None  # LORA configuration


class CustomPreprocessing(Validation, validate_assignment=True):
    zoom_factor: float = 1.0  # Zoom factor for the input images
    image_size: None | list[int] = None  # Target image size (width, height)
    flip_left_right: bool = False  # Whether to flip the image left-right (mirror)


class Augmentations(Validation, validate_assignment=True):
    random_horizontal_flip: float = 0.5  # Probability of random horizontal flip, 0 - no augmentations
    random_affine_degrees: int = 10  # Random affine rotation degrees, 0 - no rotation
    random_affine_translate: None | list[float] = [0.1, 0.1]  # Random affine translation, None - no translation
    random_affine_scale: None | list[float] = [0.9, 1.1]  # Random affine scale, None - no scaling
    gaussian_blur_prob: float = 0.1  # Probability of applying Gaussian blur, 0 - no blur
    gaussian_blur_kernel_size: int | list[int] = 7  # Gaussian blur kernel size, 0 - no blur
    gaussian_blur_sigma: float | list[float] = [0.1, 2.0]  # Gaussian blur sigma
    color_jitter_brightness: float = 0.1  # Brightness jitter factor, 0 - no brightness jitter
    color_jitter_contrast: float = 0.1  # Contrast jitter factor, 0 - no contrast jitter
    jpeg_quality: int | list[int] = [40, 100]  # JPEG quality range, 100 - no JPEG compression
    resize: None | int | list[int] = None  # Resize to (width, height), None - no resizing
    # 0:nearest,  1:lanczos, 2:bilinear, 3:bicubic, 4:box, 5:hamming
    resize_interpolation: int = 2  # Interpolation method for resizing, see InterpolationMode or Pillow integer constant
    gaussian_noise_sigma: float = 0.0  # Standard deviation of Gaussian noise to add, 0 - no noise

    @staticmethod
    def get_empty() -> Self:
        return Augmentations(
            random_horizontal_flip=0.0,
            random_affine_degrees=0,
            random_affine_translate=None,
            random_affine_scale=None,
            gaussian_blur_prob=0.0,
            gaussian_blur_kernel_size=0,
            gaussian_blur_sigma=0.0,
            color_jitter_brightness=0.0,
            color_jitter_contrast=0.0,
            jpeg_quality=100,
            resize=None,
        )


class Config(Validation, validate_assignment=True):
    # Run configuration
    run_name: str = "exp-name-1"  # Name of the run
    run_dir: str = "runs/exp"  # Directory to save the run
    seed: int = 42  # Random seed for reproducibility
    throw_exception_if_run_exists: bool = False  # Throw an exception if the run directory exists
    remove_if_run_exists: bool = False  # Remove existing run directory if it exists

    # Model configuration
    num_classes: int = 2
    num_sources: int = 5
    checkpoint: None | str = None  # Path to a checkpoint to load
    backbone: str = Backbone.CLIP_B_32  # Backbone model to use
    backbone_args: None | BackboneArgs = None  # Arguments for the backbone model
    freeze_feature_extractor: bool = True  # Freeze the feature extractor
    unfreeze_layers: list[str] = []  # Layers to unfreeze
    head: str = Head.Linear  # Head model to use
    inference_strategy: str = "softmax"  # Inference strategy to use

    # PEFT configuration
    peft_v2: None | PEFT = None

    # Data configuration
    trn_files: list[str] | dict[str, list[str]] = []  # Files containing paths to training samples
    val_files: list[str] | dict[str, list[str]] = []  # Files containing paths to validation samples
    tst_files: list[str] | dict[str, list[str]] = []  # Files containing paths to test samples
    limit_trn_files: None | int = None  # Limit the number of training files
    limit_val_files: None | int = None  # Limit the number of validation files
    limit_tst_files: None | int = None  # Limit the number of test files
    binary_labels: bool = True  # Use binary labels
    custom_preprocessing: None | CustomPreprocessing = None  # Custom preprocessing pipeline
    augmentations: None | Augmentations = Augmentations()  # Training augmentations
    test_augmentations: None | Augmentations = None  # Test-time augmentations
    load_pairs: bool = False  # Whether to load csv files with paired videos

    # Optimization configuration
    lr: float = 0.0003  # Learning rate (initial / base)
    min_lr: float = 1e-6  # Minimum learning rate
    lr_scheduler: None | Scheduler = "cosine"  # Learning rate scheduler
    warmup_epochs: float = 0  # Number of warmup epochs (can be a fraction)
    num_epochs_in_cycle: float = 1  # Number of epochs in a cycle (for cyclic schedulers)
    optimizer: str = "AdamW"  # Optimizer to use
    weight_decay: float = 0.0  # AdamW weight decay
    betas: list[float] = [0.9, 0.999]  # First and second moment coefficients for SGD and AdamW
    loss: Loss = Loss()  # Loss function to use

    # Training configuration (managed by Lightning Trainer)
    max_epochs: int = 1  # Number of epochs to train
    batch_size: int = 512  # Required batch size to perform one step
    mini_batch_size: int = 512  # Mini batch size per device
    num_workers: int = 12  # Number of workers for the DataLoader
    devices: list[int] | str | int = "auto"  # Devices to use for training
    precision: Precision = "bf16-mixed"  # Precision for the model
    fast_dev_run: int | bool = False  # Run a fast development run
    overfit_batches: int | float = 0.0  # Overfit on a subset of the data
    limit_train_batches: None | int | float = None  # Limit the number of training batches
    limit_test_batches: None | int | float = None  # Limit the number of test batches
    limit_val_batches: None | int | float = None  # Limit the number of validation batches
    deterministic: None | bool = None  # Set random seed for reproducibility
    detect_anomaly: bool = False  # Detect anomalies in the model
    early_stopping_patience: int = -1  # Early stopping patience, -1 to disable
    checkpoint_name: str = "best_mAP"  # Checkpoint to use for testing
    monitor_metric: str = "val/mAP_video"  # Metric to monitor for early stopping and checkpointing
    monitor_metric_mode: str = "max"  # Mode for monitoring metric ("max" or "min")

    # Logging
    wandb: bool = False  # Log metrics to Weights & Biases
    wandb_tags: list[str] = []  # Tags to use for Weights & Biases
    wandb_group: None | str = None  # Group to use for Weights & Biases

    # Post-processing
    make_binary_before_video_aggregation: bool = True  # Make binary labels before video aggregation
    reduce_video_predictions: Literal["mean", "median"] = "mean"  # Reduce strategy for frame to video probs

    # Validation
    @field_validator("head")
    @classmethod
    def validate_head(cls, head: str) -> str:
        return Head.validate(head)

    @field_validator("backbone")
    @classmethod
    def validate_backbone(cls, backbone: str) -> str:
        return Backbone.validate(backbone)

    @field_validator("inference_strategy")
    @classmethod
    def validate_inference_strategy(cls, inference_strategy: str) -> str:
        return InferenceStrategy.validate(inference_strategy)

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, optimizer: str) -> str:
        return Optimizer.validate(optimizer)

    def set_values_from_dict(self, dict: dict) -> Self:
        """
        Set values in the config from a dictionary. The keys of the dictionary can be
        either the names of the attributes in the config or a dot-separated path to the
        attribute. For example, if the config has an attribute `a.b.c`, you can set its
        value by passing a dictionary with the key `a.b.c`.
        """
        # Iterate over the dictionary and set the values in the config
        for key, value in dict.items():
            # If key contains a dot, traverse the config to the last key
            if "." in key:
                keys = key.split(".")
                # Traverse the config to the last key
                last_dict = self
                for next_key in keys[:-1]:
                    last_dict = getattr(last_dict, next_key)
                setattr(last_dict, keys[-1], value)
            else:
                setattr(self, key, value)
        return self


def load_config(path: str) -> Config:
    import yaml

    # read yaml config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # overwrite config
    config = Config(**config)
    return config
