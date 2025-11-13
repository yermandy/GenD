import traceback
from copy import deepcopy

import fire

from run import main
from src import config as C
from src.config import Config
from src.exp import experiments
from src.utils import files, logger


def get_val_files():
    return [
        *files.DeepSpeak_v2.my_val,
        *files.DeepSpeak_v1_1.my_val,
        *files.CDFv2.val,
        *files.FFIW.val,
    ]


def get_test_files():
    return {
        "FF": files.FF.test,
        "FF-DF": files.FF.DF.test,
        "FF-F2F": files.FF.F2F.test,
        "FF-FS": files.FF.FS.test,
        "FF-NT": files.FF.NT.test,
        "CDF": files.CDFv2.test,
        "FaceFusion": files.FaceFusion.CDF.test,
        "DFD": files.DFD.test,
        "DFDC": files.DFDC.test,
        "FSh": files.FSh.test,
        "UADFD": files.UADFV.test,
        "DFDM": files.DFDM.test,
        "FFIW": files.FFIW.test,
        "DeepSpeak-1.1": files.DeepSpeak_v1_1.test,
        "DeepSpeak-2.0": files.DeepSpeak_v2.test,
        "KoDF": files.KoDF.test,
        "KoDF-adv": files.KoDF.adversarial,
        "FakeAVCeleb": files.FakeAVCeleb.test,
        "FAVC-FV-RA-WL": files.FakeAVCeleb.FV_RA_WL.test,
        "FAVC-FV-FA-FS": files.FakeAVCeleb.FV_FA_FS.test,
        "FAVC-FV-FA-GAN": files.FakeAVCeleb.FV_FA_GAN.test,
        "FAVC-FV-FA-WL": files.FakeAVCeleb.FV_FA_WL.test,
        "PolyGlotFake": files.PolyGlotFake.test,
        "IDForge-v1": files.IDForge_v1.test,
    } | {
        k: v.map(lambda x: x.replace("/CDFv3/", "/CDFv3-x1.3-th0.5-all/subset/uniform-32-frames/"))
        for k, v in files.CDFv3.get_test_dict().items()
    }


def get_default_train_config() -> Config:
    config = Config()

    config.run_dir = "runs/rebuttal"
    config.wandb = True
    config.wandb_tags.append("rebuttal")
    config.throw_exception_if_run_exists = True

    config.num_workers = 12
    config.devices = "auto"

    config.backbone = C.Backbone.CLIP_L_14
    config.freeze_feature_extractor = True
    config.num_classes = 2

    config.batch_size = config.mini_batch_size = 128
    config.lr_scheduler = "cosine"
    config.lr = 3e-4
    config.min_lr = 1e-5
    config.weight_decay = 0
    config.max_epochs = 1 + 50
    config.warmup_epochs = 1

    config.trn_files = files.FF.train
    config.val_files = get_val_files()
    config.tst_files = get_test_files()

    return config


def get_default_test_config(orig_run_name, new_run_name) -> Config:
    orig_run_dir = files.find_run_dir(orig_run_name)
    orig_config_path = f"{orig_run_dir}/hparams.yaml"
    checkpoint = "best_mAP.ckpt"  # Default checkpoint name

    # Load run specific config
    config = C.load_config(orig_config_path)

    config.run_name = new_run_name  # Rename the run
    config.run_dir = "runs/test"  # Set default test dir
    config.checkpoint = f"{orig_run_dir}/checkpoints/{checkpoint}"

    config.wandb = True
    config.wandb_tags.extend(["test"])

    config.num_workers = 12
    config.batch_size = config.mini_batch_size = 1024
    config.devices = "auto"

    config.tst_files = get_test_files()

    return config


def get_debug_config(config: Config) -> Config:
    #! Debug

    config.run_dir = "runs/tmp"
    config.run_name = "tmp"
    # config.num_workers = 0
    config.max_epochs = 1
    config.limit_train_batches = 12
    config.limit_val_batches = 12
    config.limit_test_batches = 12
    # config.batch_size = config.mini_batch_size = 2
    # config.deterministic = True
    # config.detect_anomaly = True

    config.trn_files = files.FF.train
    config.val_files = files.FF.val
    config.tst_files = files.FF.val

    return config


experiments = {
    **experiments,  # Include all experiments defined in src.exp
}


def entry(
    exp_names: str | list[str],
    debug: bool = False,
    test: bool = False,
    from_exp: str | None = None,
    **kwargs,
):
    if test:
        if from_exp is not None:
            if isinstance(exp_names, list):
                if len(exp_names) != 1:
                    raise Exception("When running in test mode, you can provide only one experiment name.")
            config = get_default_test_config(from_exp, exp_names[0])
        else:
            logger.print_warning("Running in test mode, but 'from_exp' is not provided. Using default test config.")
            config = C.Config()
    else:
        config = get_default_train_config()

    # parse name to list
    if isinstance(exp_names, str):
        exp_names = [exp_names]

    for exp_name in exp_names:
        exp_name = exp_name.strip()

        if exp_name not in experiments:
            logger.print_error(f"Experiment '{exp_name}' is not defined in 'src/exp/__init__.py:1'")
            logger.print(f"Available experiments: {list(experiments.keys())}")
            continue

        modifiers = experiments[exp_name]
        config_exp = deepcopy(config)

        config_exp.run_name = exp_name
        for modify in modifiers:
            if isinstance(modify, Config):
                # If the modifier is a Config object, change only different values
                difference = modify.model_dump(exclude_unset=True)
                # TODO: maybe set_values_from_dict(difference)?
                config_exp = Config(**config_exp.model_copy(update=difference).model_dump())
                # config_exp = config_exp.model_copy(update=difference)
            else:
                config_exp = modify(config_exp)

        config_exp = Config(**config_exp.model_dump())  # Parse and validate config

        if debug:
            config_exp = config_exp.model_copy(update=get_debug_config(config_exp).model_dump())

        # Update config with kwargs
        config_exp.set_values_from_dict(kwargs)

        # Revalidate the config - checks if user provided valid values
        config_exp = Config(**config_exp.model_dump())

        # logger.print(config_exp)
        # exit()

        try:
            main(config_exp, not test)

        except Exception as e:
            traceback.print_exc()  # Print complete exception traceback
            logger.print_error(f"Error occurred while running experiment '{exp_name}':")
            logger.print(e)

            save_dir = f"{config_exp.run_dir}/{config_exp.run_name}"
            # Save the exception traceback to a file
            with open(f"{save_dir}/failed.log", "a") as f:
                f.write(f"\nTraining failed: {e}\n")
                f.write(traceback.format_exc())


if __name__ == "__main__":
    fire.Fire(entry)
