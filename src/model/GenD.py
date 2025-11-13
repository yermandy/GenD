from typing import Callable

import torch
from lightning import seed_everything
from PIL import Image
from torch import optim

from src import config as C
from src.config import Config, Head
from src.heads import head
from src.loss import Loss, LossInputs, LossOutputs
from src.losses import unifalign
from src.model.base import BaseDeepakeDetectionModel, Batch
from src.utils import logger
from src.utils.decorators import TryExcept


class GenD(BaseDeepakeDetectionModel):
    def __init__(self, config: Config, verbose: bool = False):
        super().__init__(config, verbose)
        self.config = config
        self.save_hyperparameters(config.model_dump())
        self.is_debug_mode = "tmp" in config.run_name

        if verbose:
            logger.print(config)

        seed_everything(self.config.seed, workers=True, verbose=verbose)

        self._init_specific_attributes(verbose)

    def _init_specific_attributes(self, verbose: bool = False):
        self._init_feature_extractor()
        self._init_head()
        self._freeze_parameters()
        self._init_peft()
        self._init_loss()

        if verbose:
            self.print_trainable_parameters()

    def print_trainable_parameters(self):
        logger.print("\n🔥 [red bold]Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                logger.print(f"[red]- {name} shape = {tuple(param.shape)}")

        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.print(
            f"Total parameters: {all_params}, trainable: {trainable_params}, %: {trainable_params / all_params * 100:.4f}"
        )

    def _init_feature_extractor(self):
        logger.print("\n[blue]Initializing image encoder...")

        backbone = self.config.backbone
        backbone_lowercase = backbone.lower()

        if "clip" in backbone_lowercase:
            from src.encoders.clip_encoder import CLIPEncoder

            self.feature_extractor = CLIPEncoder(backbone)

        elif "vit_pe" in backbone_lowercase:
            from src.encoders.perception_encoder import PerceptionEncoder

            self.feature_extractor = PerceptionEncoder(backbone, self.config.backbone_args.img_size)

        elif "dino" in backbone_lowercase:
            from src.encoders.dino_encoder import DINOEncoder

            if self.config.backbone_args is not None:
                merge_cls_token_with_patches = self.config.backbone_args.merge_cls_token_with_patches
            else:
                merge_cls_token_with_patches = None

            self.feature_extractor = DINOEncoder(backbone, merge_cls_token_with_patches)

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        logger.print(self.feature_extractor)

        # self.feature_extractor.eval()
        # self.feature_extractor.to(self.device)

    def _init_peft(self):
        if self.config.peft_v2 is not None:
            from peft import get_peft_model

            if self.config.peft_v2.lora is not None:
                from peft import LoraConfig

                peft_config = LoraConfig(
                    target_modules=self.config.peft_v2.lora.target_modules,
                    r=self.config.peft_v2.lora.rank,
                    lora_alpha=self.config.peft_v2.lora.alpha,
                    lora_dropout=self.config.peft_v2.lora.dropout,
                    bias=self.config.peft_v2.lora.bias,
                    use_rslora=self.config.peft_v2.lora.use_rslora,
                    use_dora=self.config.peft_v2.lora.use_dora,
                )

            else:
                raise ValueError("Unknown PEFT configuration")

            backbone = self.feature_extractor
            training_parameters = {name for name, param in backbone.named_parameters() if param.requires_grad}

            self.feature_extractor = get_peft_model(self.feature_extractor, peft_config)

            for name, param in backbone.named_parameters():
                if name in training_parameters:
                    param.requires_grad = True

    def _init_head(self):
        logger.print("\n[blue]Initializing head...")

        features_dim = self.feature_extractor.get_features_dim()

        match self.config.head:
            case Head.Linear:
                self.model = head.LinearProbe(features_dim, self.config.num_classes)

            case Head.NLinear:
                self.model = head.LinearProbe(features_dim, self.config.num_classes, True)

            case _:
                raise ValueError(f"Unknown head: {self.config.head}")

        # self.model.eval()
        # self.model.to(self.device)

        logger.print(self.model)

    def _freeze_parameters(self):
        # Freeze feature extractor
        self.feature_extractor.requires_grad_(not self.config.freeze_feature_extractor)

        if len(self.config.unfreeze_layers) > 0:
            for name, param in self.named_parameters():
                if any(layer in name for layer in self.config.unfreeze_layers):
                    param.requires_grad = True

    def _init_loss(self):
        self.criterion = Loss(self.config.loss)

    def get_preprocessing(self) -> Callable[[Image.Image], torch.Tensor]:
        def preprocessing(image: Image.Image) -> torch.Tensor:
            image = self.custom_preprocessing(image)
            image = self.feature_extractor.preprocess(image)
            return image

        return preprocessing

    def forward(self, inputs: torch.Tensor) -> head.HeadOutput:
        features = self.feature_extractor(inputs)
        outputs = self.model.forward(features)
        return outputs

    def log_loss(self, loss: LossOutputs, stage: str, batch_size: int):
        common = {"prog_bar": self.is_debug_mode, "on_epoch": True, "on_step": False, "batch_size": batch_size}
        if loss.total is not None:
            self.log(f"{stage}/loss", loss.total, **common)
        if loss.ce_labels is not None:
            self.log(f"{stage}/loss_ce", loss.ce_labels, **common)

    def log_aliunif(self, outputs: head.HeadOutput, labels: torch.Tensor, stage: str, batch_size: int):
        alignment = unifalign.alignment(outputs.l2_embeddings, labels)
        uniformity = unifalign.uniformity(outputs.l2_embeddings)
        common = {"prog_bar": self.is_debug_mode, "on_epoch": True, "on_step": False, "batch_size": batch_size}
        self.log(f"{stage}/alignment", alignment, **common)
        self.log(f"{stage}/uniformity", uniformity, **common)

    def get_probs(self, outputs: head.HeadOutput):
        if self.config.inference_strategy == C.InferenceStrategy.SOFTMAX:
            return outputs.logits_labels.softmax(1)

        raise NotImplementedError("Unknown inference strategy")

    def get_batch(self, batch: dict) -> Batch:
        return Batch.from_dict(batch)

    def on_train_start(self):
        logger.print(f"[blue]Logs: {self.logger.log_dir}")
        self.log("num_train_files", len(self.trainer.datamodule.train_dataset))
        self.log("num_val_files", len(self.trainer.datamodule.val_dataset))

    def on_train_epoch_start(self):
        # Log learning rate for the current epoch
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

    def training_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        # outputs = self.forward(batch.images)
        features = self.feature_extractor(batch.images)
        outputs = self.model.forward(features)

        loss_inputs = LossInputs(
            logits_labels=outputs.logits_labels,
            labels=batch.labels,
            l2_embeddings=outputs.l2_embeddings,
        )
        loss = self.criterion(loss_inputs)

        probs = self.get_probs(outputs)  # Get probabilities based on the inference strategy

        # Log metrics
        self.log_loss(loss, "train", batch_size=len(batch.images))
        self.log_aliunif(outputs, batch.labels, "train", batch_size=len(batch.images))

        # Save outputs for metrics calculation
        self.train_step_outputs.labels.update(batch.labels)
        self.train_step_outputs.probs.update(probs.detach())
        self.train_step_outputs.idx.update(batch.idx)

        return loss.total

    def on_train_epoch_end(self):
        if self.logger.log_dir is None:
            # TODO: figure out why logger.log_dir can be None
            return

        # Log weights norms
        with TryExcept(verbose=False):
            self.log("model/linear-W-norm", self.model.linear.weight.norm().item())
            self.log("model/linear-b-norm", self.model.linear.bias.norm().item())

        dataset = self.trainer.datamodule.train_dataset
        self.log_all_metrics(self.train_step_outputs, "train", dataset)

    def validation_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        outputs = self.forward(batch.images)
        loss_inputs = LossInputs(
            logits_labels=outputs.logits_labels,
            labels=batch.labels,
            l2_embeddings=outputs.l2_embeddings,
        )
        loss = self.criterion(loss_inputs)
        probs = self.get_probs(outputs)

        self.log_loss(loss, "val", len(batch.images))
        self.log_aliunif(outputs, batch.labels, "val", len(batch.images))

        # Save outputs for metrics calculation
        self.val_step_outputs.labels.update(batch.labels)
        self.val_step_outputs.probs.update(probs.detach())
        self.val_step_outputs.idx.update(batch.idx)

    def test_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        outputs = self.forward(batch.images)
        loss_inputs = LossInputs(
            logits_labels=outputs.logits_labels,
            labels=batch.labels,
            l2_embeddings=outputs.l2_embeddings,
        )
        loss = self.criterion(loss_inputs)
        probs = self.get_probs(outputs)

        self.log_loss(loss, "test", len(batch.images))
        self.log_aliunif(outputs, batch.labels, "test", len(batch.images))

        # Save outputs for metrics calculation
        self.test_step_outputs.labels.update(batch.labels)
        self.test_step_outputs.probs.update(probs.detach())
        self.test_step_outputs.idx.update(batch.idx)

    def on_validation_epoch_end(self):
        if self.logger.log_dir is None:
            # TODO: figure out why logger.log_dir can be None
            return

        dataset = self.trainer.datamodule.val_dataset
        self.log_all_metrics(self.val_step_outputs, "val", dataset)

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()  # because we need an access to the dataloader
        config = self.config

        # Separate parameters for weight decay and no weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

            optimizer_grouped_parameters = [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]

        # Configure optimizer
        if config.optimizer == C.Optimizer.AdamW:
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas,
            )
        elif config.optimizer == C.Optimizer.SGD:
            optimizer = optim.SGD(
                optimizer_grouped_parameters,
                lr=config.lr,
                momentum=config.betas[0],
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        optimizers = {"optimizer": optimizer}

        scheduler = None

        # Configure LR scheduler
        if config.lr_scheduler == "cosine":
            #! be careful when running experiments with limit_train_batches
            if config.limit_train_batches is not None:
                logger.print_warning_once("lr scheduling and limit_train_batches are not compatible")
            T_max = config.max_epochs * len(self.trainer.train_dataloader)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=config.min_lr)

        elif config.lr_scheduler == "cyclic":
            cycle_length_in_epochs = int(config.num_epochs_in_cycle * len(self.trainer.train_dataloader))
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cycle_length_in_epochs, T_mult=1, eta_min=config.min_lr
            )

        # Configure warmup
        if config.warmup_epochs > 0:
            total_warmup_steps = int(config.warmup_epochs * len(self.trainer.train_dataloader))
            warmup = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=config.min_lr / config.lr, total_iters=total_warmup_steps
            )

            if scheduler is not None:
                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer, [warmup, scheduler], milestones=[total_warmup_steps]
                )
            else:
                scheduler = warmup

        if scheduler is not None:
            optimizers["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        return optimizers
