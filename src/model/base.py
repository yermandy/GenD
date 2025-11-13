from dataclasses import dataclass
from typing import Callable, Literal

import lightning as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from sklearn import metrics as M
from torchmetrics import CatMetric

from src import metrics, plots
from src.config import Config
from src.dataset.base import BaseDataset
from src.utils import logger
from src.utils.decorators import TryExcept


class OutputsForMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.probs = CatMetric()
        self.labels = CatMetric()
        self.idx = CatMetric()

    def reset(self):
        self.probs.reset()
        self.labels.reset()
        self.idx.reset()


@dataclass
class Batch:
    images: torch.Tensor
    labels: None | torch.Tensor
    identity: None | torch.Tensor
    source_uids: None | torch.Tensor
    idx: None | torch.Tensor

    def __getitem__(self, key):
        # if batch["image"] is called, return batch.images
        return getattr(self, key)

    @staticmethod
    def from_dict(batch: dict):
        assert "image" in batch, "Batch must contain 'image' key"

        return Batch(
            images=batch.get("image"),
            labels=batch.get("label"),
            identity=batch.get("identity"),
            source_uids=batch.get("source_uid"),
            idx=batch.get("idx"),
        )


def compute_across_videos(files: list, probs: np.ndarray, labels: np.ndarray, reduce: Literal["mean", "median"]):
    """
    Calculate mean probs for each video across all frames
    """

    # Get all before the last /
    # For example: a/b/c/d -> a/b/c
    videos = [f[: -f[::-1].find("/")] for f in files]

    # Group by video: video -> [indices]
    video2idx = {v: [] for v in videos}
    for i, v in enumerate(videos):
        video2idx[v].append(i)

    # Calculate mean probs for each video across all frames
    video2probs = {v: [] for v in videos}
    video2labels = {v: [] for v in videos}
    for v, idxs in video2idx.items():
        if reduce == "mean":
            video2probs[v] = np.mean(probs[idxs], axis=0)
        elif reduce == "median":
            video2probs[v] = np.median(probs[idxs], axis=0)
        else:
            raise ValueError(f"Unknown reduce method: {reduce}")
        video2labels[v] = labels[idxs[0]]  # Assume all frames have the same label

    video_probs = np.array(list(video2probs.values()))
    video_labels = np.array(list(video2labels.values()))

    return video_probs, video_labels


class BaseDeepakeDetectionModel(pl.LightningModule):
    def __init__(self, config: Config, verbose: bool = False):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump())
        self.is_debug_mode = "tmp" in config.run_name

        if verbose:
            logger.print(config)

        seed_everything(self.config.seed, workers=True, verbose=verbose)

        self._init_metrics()

    def _init_metrics(self):
        self.train_step_outputs = OutputsForMetrics()
        self.val_step_outputs = OutputsForMetrics()
        self.test_step_outputs = OutputsForMetrics()

    def get_preprocessing(self) -> Callable[[Image.Image], torch.Tensor]:
        raise NotImplementedError("get_preprocessing must be implemented in the child class")

    def get_batch(self, batch: dict) -> Batch:
        return Batch.from_dict(batch)

    def on_train_epoch_end(self):
        if self.logger.log_dir is None:
            # TODO: figure out why logger.log_dir can be None
            return

        # Log weights norms
        with TryExcept(verbose=False):
            self.log("model/linear-W-norm", self.model.linear.weight.norm().item())
            self.log("model/linear-b-norm", self.model.linear.bias.norm().item())

        # Log learned temperature
        with TryExcept(verbose=False):
            self.log("model/criterion/compactness_loss/temp", self.criterion.compactness_loss.temp.item())

        with TryExcept(verbose=False):
            self.log("model/criterion/dispersion_loss/temp", self.criterion.dispersion_loss.temp.item())

        dataset = self.trainer.datamodule.train_dataset
        self.log_all_metrics(self.train_step_outputs, "train", dataset)

    def log_metrics(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        stage: Literal["train", "test", "val"],
        prefix: str,
        level: Literal["frame", "video"],
        dataset: BaseDataset,
    ):
        """
        Images are saved to
        `log_dir / prefix / level_metrics / metric.png`
        """

        log_dir = self.logger.log_dir

        Stage = stage.capitalize()

        # Compute ROC and PR curves for every class
        fprs, tprs, roc_ths, ovr_macro_auroc = metrics.ovr_roc(labels, probs)
        precs, recs, pr_ths, ovr_macro_ap = metrics.ovr_prc(labels, probs)

        if self.config.num_classes == 2:
            # Compute EER (Equal Error Rate)
            eer, eer_th = metrics.calculate_eer(labels, probs, True)
            self.log(f"{prefix}/eer_{level}", eer)
            self.log(f"{prefix}/eer_th_{level}", eer_th)

            # Compute TPR at selected FPRs, e.g., 0.1%, 1%, 5%
            selected_fprs = [0.001, 0.01, 0.05]
            tpr_at_fprs = metrics.calculate_tpr_at_fpr(labels, probs, selected_fprs)
            for target_fpr, tpr in zip(selected_fprs, tpr_at_fprs):
                self.log(f"{prefix}/TPR@FPR={target_fpr}_{level}", tpr)

            plots.plot_fpr_fnr_curve(
                fprs,
                tprs,
                roc_ths,
                title=f"{Stage} FPR vs FNR ({level}-level)",
                path=f"{log_dir}/{prefix}/{level}_metrics/{stage}_fpr_fnr_curve.png",
                eer=eer,
            )

            W1_sep_real, W1_sep_fake, W1_conf_real, W1_conf_fake = metrics.compute_wasserstein1_metrics(probs, labels)

            if W1_sep_real is not None:
                self.log(f"{prefix}/W1-sep-real_{level}", W1_sep_real)
                self.log(f"{prefix}/W1-sep-fake_{level}", W1_sep_fake)

                # A mean of Wasserstein distances
                self.log(f"{prefix}/W1-sep_{level}", (W1_sep_real + W1_sep_fake) / 2)

                self.log(f"{prefix}/W1-conf-real_{level}", W1_conf_real)
                self.log(f"{prefix}/W1-conf-fake_{level}", W1_conf_fake)

                # A mean of Wasserstein distances
                self.log(f"{prefix}/W1-conf_{level}", (W1_conf_real + W1_conf_fake) / 2)

            # Compute predictions by EER threshold
            preds = np.where(probs[:, 1] > eer_th, 1, 0)

        else:
            # Compute predictions by argmax rule
            preds = probs.argmax(1)

        # Log metrics
        self.log(f"{prefix}/auroc_{level}", ovr_macro_auroc)
        self.log(f"{prefix}/acc_{level}", M.accuracy_score(labels, preds))
        self.log(f"{prefix}/balanced_acc_{level}", M.balanced_accuracy_score(labels, preds))
        self.log(f"{prefix}/f1_score_{level}", M.f1_score(labels, preds, average="macro"))
        self.log(f"{prefix}/mAP_{level}", ovr_macro_ap)

        class_names = dataset.get_class_names()

        plots.plot_probs_distribution(
            probs,
            labels,
            class_names,
            f"{log_dir}/{prefix}/{level}_metrics/{stage}_probs_distribution.png",
        )

        plots.plot_roc_curve(
            fprs,
            tprs,
            roc_ths,
            f"{Stage} ROC ({level}-level)",
            f"{log_dir}/{prefix}/{level}_metrics/{stage}_roc_{level}.png",
            0.01,
            class_names,
        )

        plots.plot_prc_curve(
            precs,
            recs,
            pr_ths,
            f"{Stage} PR Curve ({level}-level)",
            f"{log_dir}/{prefix}/{level}_metrics/{stage}_pr_curve.png",
            0.01,
            class_names,
        )

        plots.plot_f1_curve(
            precs,
            recs,
            pr_ths,
            f"{Stage} F1 Curve ({level}-level)",
            f"{log_dir}/{prefix}/{level}_metrics/{stage}_f1_curve.png",
            0.01,
            class_names,
        )

        # Confusion matrix
        conf = M.confusion_matrix(labels, preds)
        plots.plot_confusion_matrix(
            conf,
            class_names,
            f"{Stage} Confusion Matrix ({level}-level)",
            f"{log_dir}/{prefix}/{level}_metrics/{stage}_confusion.png",
        )
        plots.plot_confusion_matrix(
            conf,
            class_names,
            f"{Stage} Confusion Matrix ({level}-level)",
            f"{log_dir}/{prefix}/{level}_metrics/{stage}_confusion_norm.png",
            True,
        )

        wandb_logger = self.get_wandb_logger()
        if wandb_logger is not None:
            wandb_logger.log_metrics(
                {
                    f"confusion/{prefix}/{stage}_{level}": wandb.plot.confusion_matrix(
                        y_true=labels,
                        preds=preds,
                        class_names=["real", "fake"],
                        title=f"{Stage} Confusion Matrix {level.capitalize()}",
                    )
                }
            )

    def sources_probs_to_binary(self, probs: np.ndarray) -> np.ndarray:
        # probs[:, 0]  # is real probs
        # probs[:, 1:]  # is fake probs (for each generator)
        return np.stack([probs[:, 0], probs[:, 1:].max(axis=1)], 1)

    def log_all_metrics(
        self,
        outputs_for_metrics: OutputsForMetrics,
        stage: Literal["train", "test", "val"],
        dataset: BaseDataset,
    ):
        # Merge all predictions and labels across processes
        labels = outputs_for_metrics.labels.compute().cpu().int().numpy()
        probs = outputs_for_metrics.probs.compute().cpu().numpy()
        idx = outputs_for_metrics.idx.compute().cpu().int().numpy()
        files = [dataset.files[i] for i in idx]  # Get files in the same order as the rest
        outputs_for_metrics.reset()

        if self.config.make_binary_before_video_aggregation:
            if probs.shape[1] > 2:
                probs = self.sources_probs_to_binary(probs)

        # Compute probs and labels for videos
        video_probs, video_labels = compute_across_videos(files, probs, labels, self.config.reduce_video_predictions)

        # Convery to binary if sources are used
        if not self.config.make_binary_before_video_aggregation:
            if probs.shape[1] > 2:
                probs = self.sources_probs_to_binary(probs)
                video_probs = self.sources_probs_to_binary(video_probs)

        self.log_metrics(probs, labels, stage, stage, "frame", dataset)
        self.log_metrics(video_probs, video_labels, stage, stage, "video", dataset)

        # if trn_files / val_files / tst_files is dict, separate metrics for each dataset
        if dataset.dataset2files is not None:
            if not self.config.make_binary_before_video_aggregation:
                logger.print_warning(
                    "`make_binary_before_video_aggregation=False` is not supported when trn_files / val_files / tst_files is dict"
                )

            file2index = {f: i for i, f in enumerate(files)}
            for dataset_name, dataset_files in dataset.dataset2files.items():
                # Get files only for current dataset
                dataset_files = np.intersect1d(files, dataset_files)
                file_indices = [file2index[f] for f in dataset_files]
                dataset_probs = probs[file_indices]
                dataset_labels = labels[file_indices]
                dataset_files = [files[i] for i in file_indices]

                self.log_metrics(
                    dataset_probs,
                    dataset_labels,
                    stage,
                    f"{stage}/dataset/{dataset_name}",
                    "frame",
                    dataset,
                )

                dataset_video_probs, dataset_video_labels = compute_across_videos(
                    dataset_files, dataset_probs, dataset_labels, self.config.reduce_video_predictions
                )

                self.log_metrics(
                    dataset_video_probs,
                    dataset_video_labels,
                    stage,
                    f"{stage}/dataset/{dataset_name}",
                    "video",
                    dataset,
                )

    def custom_preprocessing(self, image: Image.Image) -> Image.Image:
        if self.config.custom_preprocessing is None:
            return image

        if self.config.custom_preprocessing.zoom_factor != 1.0:
            zoom_factor = self.config.custom_preprocessing.zoom_factor

            width, height = image.size
            # Calculate crop size (smaller portion of the image to simulate zoom-in)
            crop_w = width // zoom_factor
            crop_h = height // zoom_factor

            # Center crop coordinates
            left = (width - crop_w) // 2
            top = (height - crop_h) // 2
            right = left + crop_w
            bottom = top + crop_h

            # Crop and resize back to original size
            cropped_img = image.crop((left, top, right, bottom))

            if self.config.custom_preprocessing.image_size is not None:
                image = cropped_img.resize(self.config.custom_preprocessing.image_size, Image.BILINEAR)
            else:
                # Use bilinear interpolation to preserve artifacts
                image = cropped_img.resize((width, height), Image.BILINEAR)

        if self.config.custom_preprocessing.image_size is not None:
            image = image.resize(self.config.custom_preprocessing.image_size, Image.BILINEAR)

        if self.config.custom_preprocessing.flip_left_right:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def get_wandb_logger(self) -> WandbLogger | None:
        """
        Get the WandbLogger instance from the current loggers.
        Returns None if no WandbLogger is found.
        """
        for l in self.loggers:
            if isinstance(l, WandbLogger):
                return l
        return None

    def on_test_start(self):
        logger.print(f"[blue]Logs: {self.logger.log_dir}")
        self.log("num_test_files", len(self.trainer.datamodule.test_dataset))

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("test_step must be implemented in the child class")

    def on_test_epoch_end(self):
        if self.logger.log_dir is None:
            # TODO: figure out why logger.log_dir can be None
            return

        # Concatenate all predictions and labels
        probs = self.test_step_outputs.probs.compute().cpu().numpy()
        labels = self.test_step_outputs.labels.compute().cpu().int().numpy()
        idx = self.test_step_outputs.idx.compute().cpu().int().numpy()

        dataset = self.trainer.datamodule.test_dataset

        files = [dataset.files[i] for i in idx]

        # preds is a 2D array of shape (num_samples, num_classes)
        probs = {f"prob_class_{i}": np.round(probs[:, i], 4) for i in range(probs.shape[1])}
        table = pd.DataFrame({"files": files, "labels": labels, **probs})

        # Save to CSV
        table.to_csv(f"{self.logger.log_dir}/test_predictions.csv", index=False, float_format="%.4f")

        self.log_all_metrics(self.test_step_outputs, "test", dataset)

    def load_checkpoint(self, checkpoint: str):
        if checkpoint:
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
            incompatible_keys = self.load_state_dict(state_dict, strict=False)
            self.print_checkpoint_keys(incompatible_keys)

    def print_checkpoint_keys(self, incompatible_keys):
        missing_keys = set(incompatible_keys.missing_keys)
        unexpected_keys = set(incompatible_keys.unexpected_keys)

        logger.print("\n[blue bold]Keys in checkpoint:")
        logger.print("[red bold]- Missing")
        logger.print("[yellow bold]? Unexpected")
        logger.print("[green bold]+ Matched\n")

        for key in self.state_dict().keys():
            if key in missing_keys:
                logger.print(f"[red]- {key}")
            elif key in unexpected_keys:
                logger.print(f"[orange]? {key}")
            else:
                logger.print(f"[green]+ {key}")
