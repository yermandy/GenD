from copy import deepcopy

from .. import config as C
from ..config import Config
from ..utils import files

experiments = {
    #! CLIP
    "wacv-Baseline": [
        Config(
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.Linear,
            loss=C.Loss(ce_labels=1.0),
        )
    ],
    "wacv-LN": [
        Config(
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.Linear,
            unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
            loss=C.Loss(ce_labels=1.0),
        ),
    ],
    "wacv-LN-noaug": [
        Config(
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.Linear,
            unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
            loss=C.Loss(ce_labels=1.0),
            augmentations=None,
        ),
    ],
    "wacv-LN+L2": [
        Config(
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.NLinear,
            unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
            loss=C.Loss(ce_labels=1.0),
        ),
    ],
    "wacv-LN+L2+UnAl": [
        Config(
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.NLinear,
            unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
            loss=C.Loss(ce_labels=1.0, uniformity=0.5, alignment_labels=0.1),
        ),
    ],
    #! CLIP+components without LN
    "wacv-NoLN-L2": [
        Config(
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.NLinear,
            loss=C.Loss(ce_labels=1.0),
        ),
    ],
    "wacv-NoLN-L2+UnAl": [
        Config(
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.NLinear,
            loss=C.Loss(ce_labels=1.0, uniformity=0.5, alignment_labels=0.1),
        ),
    ],
    #! PE
    "wacv-PE_L-Baseline": [
        Config(
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(image_size=224),
            head=C.Head.Linear,
            loss=C.Loss(ce_labels=1.0),
        )
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
    "wacv-PE_L-LN-noaug": [
        Config(
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(img_size=224),
            head=C.Head.Linear,
            unfreeze_layers=["norm_pre", "norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0),
            augmentations=None,
        ),
    ],
    "wacv-PE_L-LN+L2": [
        Config(
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(img_size=224),
            head=C.Head.NLinear,
            unfreeze_layers=["norm_pre", "norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0),
        ),
    ],
    "wacv-PE_L-LN+L2+UnAl": [
        Config(
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(img_size=224),
            head=C.Head.NLinear,
            unfreeze_layers=["norm_pre", "norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0, uniformity=0.5, alignment_labels=0.1),
        ),
    ],
    "wacv-PE_L-LN+L2+UA-U1.0-A0.5": [
        Config(
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(img_size=224),
            head=C.Head.NLinear,
            unfreeze_layers=["norm_pre", "norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0, uniformity=1.0, alignment_labels=0.5),
        ),
    ],
    "wacv-PE_L-LN+L2+UA-U1.0-A0.1": [
        Config(
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(img_size=224),
            head=C.Head.NLinear,
            unfreeze_layers=["norm_pre", "norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0, uniformity=1.0, alignment_labels=0.1),
        ),
    ],
    "wacv-PE_L-LN+L2+UA-U0.5-A0.0": [
        Config(
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(img_size=224),
            head=C.Head.NLinear,
            unfreeze_layers=["norm_pre", "norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0, uniformity=0.5, alignment_labels=0.0),
        ),
    ],
    "wacv-PE_L-LN+L2+UA-U1.0-A0.0": [
        Config(
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(img_size=224),
            head=C.Head.NLinear,
            unfreeze_layers=["norm_pre", "norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0, uniformity=1.0, alignment_labels=0.0),
        ),
    ],
    #! DINOv3
    "wacv-DINOv3L-Baseline": [
        Config(
            backbone=C.Backbone.DINOv3_ViT_L,
            head=C.Head.Linear,
            loss=C.Loss(ce_labels=1.0),
        )
    ],
    "wacv-DINOv3L-LN": [
        Config(
            backbone=C.Backbone.DINOv3_ViT_L,
            head=C.Head.Linear,
            unfreeze_layers=["norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0),
        ),
    ],
    "wacv-DINOv3L-LN-noaug": [
        Config(
            backbone=C.Backbone.DINOv3_ViT_L,
            head=C.Head.Linear,
            unfreeze_layers=["norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0),
            augmentations=None,
        ),
    ],
    "wacv-DINOv3L-LN+L2": [
        Config(
            backbone=C.Backbone.DINOv3_ViT_L,
            head=C.Head.NLinear,
            unfreeze_layers=["norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0),
        ),
    ],
    "wacv-DINOv3L-LN+L2+UnAl": [
        Config(
            backbone=C.Backbone.DINOv3_ViT_L,
            head=C.Head.NLinear,
            unfreeze_layers=["norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0, uniformity=0.5, alignment_labels=0.1),
        ),
    ],
    "wacv-DINOv3L-LN+L2+UA-U0.5-A0.5": [
        Config(
            backbone=C.Backbone.DINOv3_ViT_L,
            head=C.Head.NLinear,
            unfreeze_layers=["norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0, uniformity=0.5, alignment_labels=0.5),
        ),
    ],
    "wacv-DINOv3L-LN+L2+UA-U0.1-A0.5": [
        Config(
            backbone=C.Backbone.DINOv3_ViT_L,
            head=C.Head.NLinear,
            unfreeze_layers=["norm1", "norm2", "norm"],
            loss=C.Loss(ce_labels=1.0, uniformity=0.1, alignment_labels=0.5),
        ),
    ],
}


def get_common():
    config = Config()
    config.run_dir = "runs/rebuttal"
    config.wandb_tags = ["rebuttal"]
    config.lr_scheduler = "cyclic"
    config.num_epochs_in_cycle = 10
    config.early_stopping_patience = 30
    config.max_epochs = 30

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

    return config


def set_common_settings(experiments):
    for run_name, modifieres in experiments.items():
        experiments[run_name][0] = Config(
            **{
                **get_common().model_dump(exclude_unset=True),  # get default settings
                **modifieres[0].model_dump(exclude_unset=True),  # override with specific experiment settings
            }
        )


set_common_settings(experiments)

registered_experiments = deepcopy(experiments)

#! Add 5 splits with different seeds and different trn splits
for seed in range(5):
    for run_name, modifieres in registered_experiments.items():
        config = modifieres[0]
        config = deepcopy(config)

        config.trn_files = files.FF.train.map(
            lambda x: x.replace("/FF/", f"/FF-x1.3-th0.5-all/subset/random-32-frames/split-{seed}/")
        )

        config.seed = seed

        config.wandb_group = f"{run_name}"

        run_name = f"{run_name}-seed{seed}"

        experiments[run_name] = [config]


#! Add Uniformity-Alignment α, β hyperparameter sweep
seeds = [0]
alphas = [0.0, 0.1, 0.5, 1.0, 5.0]
betas = [0.0, 0.1, 0.5, 1.0, 5.0]

for seed in seeds:
    for alpha in alphas:
        for beta in betas:
            for run_name, modifieres in registered_experiments.items():
                if "UnAl" not in run_name:
                    continue

                config = modifieres[0]
                config = deepcopy(config)

                config.trn_files = files.FF.train.map(
                    lambda x: x.replace("/FF/", f"/FF-x1.3-th0.5-all/subset/random-32-frames/split-{seed}/")
                )

                config.seed = seed
                config.loss.uniformity = alpha
                config.loss.alignment_labels = beta

                config.wandb_group = f"{run_name}-sweep"

                run_name = f"{run_name}-sweep-A{alpha}-U{beta}-seed{seed}"

                experiments[run_name] = [config]
