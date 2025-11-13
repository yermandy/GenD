from typing import Callable

from torch.utils.data import DataLoader

from src.config import Config
from src.utils import logger

from .augmentations import init_augmentations
from .base import BaseDataModule
from .dataset import DeepfakeDataset


class DeepfakeDataModule(BaseDataModule):
    def __init__(self, config: Config, preprocess: None | Callable = None):
        super().__init__(config, preprocess)

    def setup(self, stage: str):
        # Initialize datasets
        if stage == "fit" or stage == "validate":
            logger.print("\n[blue]Creating training dataset")
            self.train_dataset = DeepfakeDataset(
                self.config.trn_files,
                self.preprocess,
                augmentations=init_augmentations(self.config.augmentations),
                binary=self.config.binary_labels,
                limit_files=self.config.limit_trn_files,
                load_pairs=self.config.load_pairs,
            )
            self.train_dataset.print_statistics()

            logger.print("\n[blue]Creating validation dataset")
            self.val_dataset = DeepfakeDataset(
                self.config.val_files,
                self.preprocess,
                shuffle=True,
                binary=self.config.binary_labels,
                limit_files=self.config.limit_val_files,
            )
            self.val_dataset.print_statistics()

        if stage == "test":
            logger.print("\nCreating test dataset")
            self.test_dataset = DeepfakeDataset(
                self.config.tst_files,
                self.preprocess,
                augmentations=init_augmentations(self.config.test_augmentations),
                binary=self.config.binary_labels,
                limit_files=self.config.limit_tst_files,
            )
            self.test_dataset.print_statistics()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.mini_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.mini_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.mini_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
