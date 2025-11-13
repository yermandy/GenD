from .. import config as C
from ..config import Config
from ..utils import files

experiments = {
    "Effort-tmp": [
        Config(
            checkpoint="weights/effort/effort_clip_L14_trainOn_FaceForensic.pth",
        ),
    ],
    "ForAda-tmp": [
        Config(
            checkpoint="weights/forensics_adapter/ForensicsAdapter.pth",
        ),
    ],
    **{
        f"FS-VFM-{zoom_factor}-bilinear": [
            Config(
                checkpoint="weights/FS-VFM/FS-VFM-ViT-L.pth",
                custom_preprocessing=C.CustomPreprocessing(zoom_factor=zoom_factor),
                mini_batch_size=1024,
                batch_size=1024,
            ),
        ]
        for zoom_factor in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    },
}


def get_common():
    config = Config()
    config.run_dir = "runs/test"
    config.num_workers = 12
    config.wandb = True
    config.wandb_tags = ["test"]

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
