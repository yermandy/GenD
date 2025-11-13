from concurrent.futures import ThreadPoolExecutor

from lightning.pytorch import callbacks as pl_callbacks
from typing_extensions import override

from src.utils import logger


class ModelCheckpointParallel(pl_callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threads = []
        self.thread_pool = ThreadPoolExecutor(1, thread_name_prefix="ModelCheckpointParallel")

    @override
    def on_train_batch_end(self, *args, **kwargs):
        trainer = args[0]
        if self._should_skip_saving_checkpoint(trainer):
            return
        self.threads.append(self.thread_pool.submit(super().on_train_batch_end, *args, **kwargs))

    @override
    def on_train_epoch_end(self, *args, **kwargs):
        trainer = args[0]
        if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
            self.threads.append(self.thread_pool.submit(super().on_train_epoch_end, *args, **kwargs))

    @override
    def on_validation_end(self, *args, **kwargs):
        trainer = args[0]
        if not self._should_skip_saving_checkpoint(trainer) and not self._should_save_on_train_epoch_end(trainer):
            self.threads.append(self.thread_pool.submit(super().on_validation_end, *args, **kwargs))

    def wait(self):
        for thread in self.threads:
            try:
                thread.result()
            except Exception as e:
                logger.print_error(f"Exception during checkpoint saving in thread: {e}")
        self.thread_pool.shutdown(wait=True)
        self.thread_pool = ThreadPoolExecutor(1, thread_name_prefix="ModelCheckpointParallel")
        self.threads = []

    @override
    def on_train_end(self, *args, **kwargs):
        self.wait()

    @override
    def on_test_start(self, *args, **kwargs):
        self.wait()
