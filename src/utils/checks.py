import os
import shutil

from ..config import Config
from . import logger


def checks(config: Config):
    save_dir = f"{config.run_dir}/{config.run_name}"

    if "tmp" in config.run_name:
        logger.print_warning("Using 'tmp' in run name. Wandb will not be used.")
        config.wandb = False

    if os.path.exists(save_dir) and "tmp" not in save_dir:
        if config.throw_exception_if_run_exists:
            raise FileExistsError(f"Folder {save_dir} exists, remove it or include 'tmp' in run name")
        logger.print()
        logger.print_warning(f"folder [magenta]{save_dir}[/] exists, remove it or include 'tmp' in run name")
        if config.remove_if_run_exists:
            logger.print_warning(f"Folder [magenta]{save_dir}[/] is removed")
            shutil.rmtree(str(save_dir))
        else:
            logger.print("Enter [green bold]R[/] to replace")
            # Interactively ask
            key = input()
            if key not in ["R"]:
                logger.print_error("Aborted")
                exit()
            if key == "R":
                logger.print_warning(f"Folder [magenta]{save_dir}[/] is removed")
                shutil.rmtree(str(save_dir))

    if config.binary_labels and config.num_classes != 2:
        raise ValueError("Binary labels is only supported for 2 classes")

    def get_files_from_dict_values(d: list[str] | dict[str, list[str]]):
        if isinstance(d, list):
            return d
        return [f for sublist in d.values() for f in sublist]

    trn_files = get_files_from_dict_values(config.trn_files)
    if not all(os.path.exists(f) for f in trn_files):
        not_found = [f for f in trn_files if not os.path.exists(f)]
        raise FileNotFoundError(f"Some train files are not found: {not_found}")

    val_files = get_files_from_dict_values(config.val_files)
    if not all(os.path.exists(f) for f in val_files):
        not_found = [f for f in val_files if not os.path.exists(f)]
        raise FileNotFoundError(f"Some val files are not found: {not_found}")

    tst_files = get_files_from_dict_values(config.tst_files)
    if not all(os.path.exists(f) for f in tst_files):
        not_found = [f for f in tst_files if not os.path.exists(f)]
        raise FileNotFoundError(f"Some test files are not found: {not_found}")
