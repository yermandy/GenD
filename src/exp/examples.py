from .. import config as C
from ..config import Config

experiments = {
    "example-training": [
        Config(
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.Linear,
            unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
            loss=C.Loss(ce_labels=1.0),
            run_dir="runs/example",
            trn_files=[
                "config/datasets/FF/test/DF.txt",
                "config/datasets/FF/test/F2F.txt",
                "config/datasets/FF/test/FS.txt",
                "config/datasets/FF/test/NT.txt",
                "config/datasets/FF/test/real.txt",
            ],
            val_files=[
                "config/datasets/FF/test/DF.txt",
                "config/datasets/FF/test/F2F.txt",
                "config/datasets/FF/test/FS.txt",
                "config/datasets/FF/test/NT.txt",
                "config/datasets/FF/test/real.txt",
            ],
            tst_files=[
                "config/datasets/CDFv2/test/Celeb-real.txt",
                "config/datasets/CDFv2/test/Celeb-synthesis.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
            ],
            batch_size=2,
            mini_batch_size=2,
            max_epochs=1,
            wandb=False,
            devices=[0],
        )
    ],
    "example-test": [
        Config(
            run_dir="runs/test",
            tst_files=[
                "config/datasets/CDFv2/test/Celeb-real.txt",
                "config/datasets/CDFv2/test/Celeb-synthesis.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
            ],
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
        )
    ],
    "GenD_CLIP--CDFv2-example": [
        Config(
            run_dir="runs/test",
            tst_files=[
                "config/datasets/CDFv2/test/Celeb-real.txt",
                "config/datasets/CDFv2/test/Celeb-synthesis.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
            ],
            checkpoint="yermandy/GenD_CLIP_L_14",
            max_epochs=1,
            wandb=False,
            devices=[0],
        )
    ],
    "GenD_PE--CDFv2-example": [
        Config(
            run_dir="runs/test",
            tst_files=[
                "config/datasets/CDFv2/test/Celeb-real.txt",
                "config/datasets/CDFv2/test/Celeb-synthesis.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
            ],
            checkpoint="yermandy/GenD_PE_L",
            max_epochs=1,
            wandb=False,
            devices=[0],
        )
    ],
    "GenD_DINO--CDFv2-example": [
        Config(
            run_dir="runs/test",
            tst_files=[
                "config/datasets/CDFv2/test/Celeb-real.txt",
                "config/datasets/CDFv2/test/Celeb-synthesis.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
            ],
            checkpoint="yermandy/GenD_DINOv3_L",
            max_epochs=1,
            wandb=False,
            devices=[0],
        )
    ],
}
