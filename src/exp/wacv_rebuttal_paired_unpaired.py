from copy import deepcopy

from .. import config as C
from ..config import Config
from ..utils import files

experiments = {
    "wacv-LN": [
        Config(
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.Linear,
            unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
            loss=C.Loss(ce_labels=1.0),
        ),
    ],
    "wacv-PE_L-LN": [
        Config(
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(img_size=224),
            head=C.Head.Linear,
            unfreeze_layers=["norm_pre", "norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0),
        ),
    ],
}


# Add common settings
for run_name, modifieres in experiments.items():
    config = modifieres[0]
    config.run_dir = "runs/rebuttal"
    config.wandb_tags = ["rebuttal"]
    config.lr_scheduler = "cyclic"
    config.num_epochs_in_cycle = 10
    config.max_epochs = 30
    config.early_stopping_patience = 30

    config.val_files = [
        *files.DeepSpeak_v2.my_val,
        *files.DeepSpeak_v1_1.my_val,
        *files.CDFv3.my_val.map(lambda x: x.replace("/CDFv3/", "/CDFv3-x1.3-th0.5-all/subset/uniform-32-frames/")),
        *files.FFIW.val,
    ]

    config.tst_files = {
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


registered_experiments = deepcopy(experiments)
experiments = {}

# add 20 splits with different seeds and different trn splits
for paired_or_unpaired in ["paired", "unpaired"]:
    for seed in range(20):
        for run_name, modifieres in registered_experiments.items():
            config = modifieres[0]
            config = deepcopy(config)

            config.trn_files = files.FF.train.map(
                lambda x: x.replace(
                    "/FF/", f"/FF-x1.3-th0.5-all/subset/paired-unpaired/split-{seed}/{paired_or_unpaired}/"
                )
            )

            config.seed = seed

            run_name = run_name.replace("wacv-", f"wacv-{paired_or_unpaired}-")
            run_name = f"{run_name}-seed{seed:02d}"

            experiments[run_name] = [config]


#! Add 20 splits training on CDFv2, remove CDFv2 from val_files, add FF++ validation set
for paired_or_unpaired in ["paired", "unpaired"]:
    for seed in range(20):
        for run_name, modifieres in registered_experiments.items():
            config = modifieres[0]
            config = deepcopy(config)

            config.trn_files = [
                f"config/datasets/CDFv3-x1.3-th0.5-all/subset/paired-unpaired/split-{seed}/{paired_or_unpaired}/train/Celeb-DF-v2.txt",
                f"config/datasets/CDFv3-x1.3-th0.5-all/subset/paired-unpaired/split-{seed}/{paired_or_unpaired}/train/Celeb-real.txt",
            ]

            config.val_files = [
                *files.FF.val,
                *files.DeepSpeak_v2.my_val,
                *files.DeepSpeak_v1_1.my_val,
                *files.FFIW.val,
            ]

            config.seed = seed

            run_name = run_name.replace("wacv-", f"wacv-{paired_or_unpaired}-CDFv2-")
            run_name = f"{run_name}-seed{seed:02d}"

            experiments[run_name] = [config]


#! Add 20 splits training on FAVC
for paired_or_unpaired in ["paired", "unpaired"]:
    for seed in range(20):
        for run_name, modifieres in registered_experiments.items():
            config = modifieres[0]
            config = deepcopy(config)

            config.trn_files = [
                f"config/datasets/FakeAVCeleb/subset/paired-unpaired/split-{seed}/{paired_or_unpaired}/train/fake.txt",
                f"config/datasets/FakeAVCeleb/subset/paired-unpaired/split-{seed}/{paired_or_unpaired}/train/real.txt",
            ]

            config.seed = seed

            run_name = run_name.replace("wacv-", f"wacv-{paired_or_unpaired}-FAVC-")

            # Create a group
            config.wandb_group = run_name

            # Add seed to name
            run_name = f"{run_name}-seed{seed:02d}"

            experiments[run_name] = [config]


#! List all experiments
# for run_name in sorted(experiments.keys()):
#     print(run_name)
