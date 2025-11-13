import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

from src.utils import logger
from src.utils.logger import print

from .base import BaseDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DeepfakeDataset(BaseDataset):
    """
    DeepfakeDataset is any dataset that follows this structure:
    ... / <dataset_name> / <source_name> / <video_name> / <frame_name>

    <dataset_name> - Name of the dataset, e.g. FF, CDF, DFD, DFDC...
    <source_name> - Name of the source, e.g. real, fake, or any name of generator, e.g. FaceSwap, Face2Face...
    <video_name> - Name of the video, e.g. 000, 000_003, ...
    <frame_name> - Any name of the frame, e.g. 000001.jpg, 000002.jpg, ...

    Labels are automatically created from <source_name> such that:
    - If <source_name> has "real" substring, the label is 0
    - Otherwise, the label is 1

    """

    def __init__(
        self,
        files_with_paths: list[str] | dict[str, list[str]],
        preprocess: None | Callable = None,
        augmentations: None | Callable = None,
        shuffle: bool = False,  # Shuffles the dataset once
        binary: bool = False,
        limit_files: None | int = None,
        load_pairs: bool = False,
    ):
        files = []
        labels = []
        logger.print_info("Loading files")

        if binary:
            label2name = {0: "real", 1: "fake"}
        else:
            raise NotImplementedError("Only binary classification is supported now")

        source2label = {v: k for k, v in label2name.items()}

        self.label2name = label2name

        dataset2files = None

        if isinstance(files_with_paths, dict):
            dataset2files_with_paths = files_with_paths.copy()
            dataset2files = {dataset_name: [] for dataset_name in dataset2files_with_paths.keys()}
            files_with_paths = [item for sublist in files_with_paths.values() for item in sublist]

        max_workers = min(64, os.cpu_count())

        for file_with_paths in sorted(set(files_with_paths)):
            with open(file_with_paths, "r") as f:
                paths = f.readlines()
                paths = [path.strip() for path in paths]

                # If files do not exist, append root of 'file' to each path
                root = file_with_paths.rsplit("/", 1)[0]

                def process_path(root, path):
                    if not os.path.exists(path):
                        path = f"{root}/{path}"
                    assert os.path.exists(path), f"File not found: {path}"
                    return path

                with ThreadPoolExecutor(max_workers) as executor:
                    process_with_root = partial(process_path, root)
                    paths = list(
                        tqdm(
                            executor.map(process_with_root, paths),
                            total=len(paths),
                            desc=f"Processing paths in {file_with_paths}",
                            leave=True,
                        )
                    )

                files.extend(paths)

                if dataset2files is not None:
                    for dataset_name, files_with_paths in dataset2files_with_paths.items():
                        if file_with_paths in files_with_paths:
                            dataset2files[dataset_name].extend(paths)

        # Remove duplicate paths
        files = np.unique(files).tolist()

        # Limit the number of files
        if limit_files is not None:
            files = self.limit_files(files, limit_files)

        # Get labels from paths
        for path in files:
            source = self.get_source_from_file(path)

            if binary:
                if "real" in source:
                    source = "real"
                else:
                    source = "fake"

            label = source2label[source]
            labels.append(label)

        logger.print_info("Files loaded")

        super().__init__(files, labels, preprocess, augmentations, shuffle, dataset2files)

        self.source2uid = self._source2uid()
        self.video_path2uid = self._video_path2uid()

        self.file2index = {f: i for i, f in enumerate(self.files)}

    def limit_files(self, files: list[str], limit: int) -> list[str]:
        """Limits number of files by considering unique videos"""
        # Select unique videos
        video_paths = [self.get_video_path(file) for file in files]
        unique_videos = list(np.unique(video_paths))

        # For each video, select files
        video2files = {video: [] for video in unique_videos}
        for file, video in zip(files, video_paths):
            video2files[video].append(file)

        # Shuffle videos with fixed seed
        np.random.RandomState(42).shuffle(unique_videos)

        # Select files from shuffled videos
        selected_files = []
        for video in unique_videos:
            selected_files.extend(video2files[video])

            if len(selected_files) >= limit:
                break

        return selected_files[:limit]

    def _source2uid(self) -> dict[str, int]:
        sources = [self.get_source_from_file(file) for file in self.files]
        sources = np.unique(sources)

        assert any("real" in g for g in sources), "No real source found"
        sources = [str(g) for g in sources]

        # Map all real sources to 0 and fake sources to 1, 2, 3, ...
        real_sources = [g for g in sources if "real" in g]
        fake_sources = [g for g in sources if "real" not in g]

        source2uid = dict.fromkeys(real_sources, 0)
        for i, s in enumerate(fake_sources, start=1):
            source2uid[s] = i

        return source2uid

    def _video_path2uid(self) -> dict[str, int]:
        video_paths = [self.get_video_path(file) for file in self.files]
        unique_videos = list(np.unique(video_paths))
        return {video: i for i, video in enumerate(unique_videos)}

    @staticmethod
    def get_frame_from_file(file_path: str) -> str:
        # ... / <dataset_name> / <source_name> / <video_name> / <frame_name>
        # returns <frame_name>
        return file_path.split("/")[-1]

    @staticmethod
    def get_video_from_file(file_path: str) -> str:
        # ... / <dataset_name> / <source_name> / <video_name> / <frame_name>
        # returns <video_name>
        return file_path.split("/")[-2]

    @staticmethod
    def get_source_from_file(file_path: str) -> str:
        # ... / <dataset_name> / <source_name> / <video_name> / <frame_name>
        # returns <source_name>
        return file_path.split("/")[-3]

    @staticmethod
    def get_dataset_from_file(file_path: str) -> str:
        # ... / <dataset_name> / <source_name> / <video_name> / <frame_name>
        # returns <dataset_name>
        return file_path.split("/")[-4]

    @staticmethod
    def get_video_path(file_path: str) -> str:
        # ... / <dataset_name> / <source_name> / <video_name> / <frame_name>
        # file_path[::-1].find("/") finds the last occurrence of "/"
        # returns .../<dataset_name>/<source_name>/<video_name>/
        return file_path[: -file_path[::-1].find("/")]

    def get_class_names(self) -> dict[int, str]:
        return self.label2name

    def print_statistics(self):
        super().print_statistics()

        video_paths = [self.get_video_path(file) for file in self.files]

        files_by_dataset = [self.get_dataset_from_file(file) for file in self.files]

        print(f"Total number of frames: {len(self.files)}")
        print(f"Total number of videos: {len(set(video_paths))}")

        # For each dataset, print number of frames and videos
        df = pd.DataFrame({"dataset": files_by_dataset, "video": video_paths})

        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]
            videos_count = dataset_df["video"].nunique()
            frames_count = len(dataset_df)
            print(f"Dataset: {dataset}, videos: {videos_count}, frames: {frames_count}")

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path)
        source = self.get_source_from_file(path)
        video_path = self.get_video_path(path)
        label = self.labels[idx]

        # Apply augmentations defined in from config.Augmentations
        if self.augmentations is not None:
            image = self.augmentations(image)

        # Apply preprocessing defined by the model input requirements
        if self.preprocess is not None:
            image = self.preprocess(image)

        output = {
            "idx": idx,
            "image": image,
            "label": label,
            "path": path,
            "video": self.get_video_from_file(path),
            "source_uid": self.source2uid[source],
            "frame": self.get_frame_from_file(path),
            "video_uid": self.video_path2uid[video_path],
        }

        return output
