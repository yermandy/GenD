from copy import deepcopy

from .. import config as C
from ..config import Config
from ..utils import files

_experiments = {
    # "wacv-Baseline": [Config(test_augmentations=get_empty_augmentations())],
    # "wacv-LN": [Config(test_augmentations=get_empty_augmentations())],
    # "wacv-LN+L2": [Config(test_augmentations=get_empty_augmentations())],
    "wacv-LN+L2+UnAl": [Config(test_augmentations=C.Augmentations.get_empty())],  #! Select this group of runs
    # "wacv-PE_L-Baseline": [Config(test_augmentations=get_empty_augmentations())],
    # "wacv-PE_L-LN": [Config(test_augmentations=get_empty_augmentations())],
    "wacv-PE_L-LN+L2": [Config(test_augmentations=C.Augmentations.get_empty())],  #! Select this group of runs
    # "wacv-PE_L-LN+L2+UnAl": [Config(test_augmentations=get_empty_augmentations())],
}


# Add common settings
for run_name, modifieres in _experiments.items():
    config = modifieres[0]
    config.run_dir = "runs/test-aug-robustness"
    config.wandb_tags = ["rebuttal", "test"]
    config.seed = 0  # We want augmentations to be the same for all runs

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


jpeg_quality_levels = [100, 80, 60, 40, 20, 10]


experiments = {}
# Test for all 5 seeds
for seed in range(5):
    for q in jpeg_quality_levels:
        for run_name, modifieres in _experiments.items():
            config = modifieres[0]
            config = deepcopy(config)

            config.test_augmentations.jpeg_quality = [q, q]

            run_name = f"{run_name}-seed{seed}-jpeg{q}"

            experiments[run_name] = [config]


blur_levels = [(0, 0.0), (5, 0.5), (7, 1.0), (11, 1.5), (13, 2.0), (19, 3.0)]

for seed in range(5):
    for k, s in blur_levels:
        for run_name, modifieres in _experiments.items():
            config = modifieres[0]
            config = deepcopy(config)

            config.test_augmentations.gaussian_blur_kernel_size = k
            config.test_augmentations.gaussian_blur_sigma = (s, s)

            run_name = f"{run_name}-seed{seed}-blur-{k}-{s}"

            experiments[run_name] = [config]


resize_levels = [224, 112, 64]
interpolations = [0, 1, 2, 3, 4, 5]

for seed in range(5):
    for resize in resize_levels:
        for interp in interpolations:
            for run_name, modifieres in _experiments.items():
                config = modifieres[0]
                config = deepcopy(config)

                config.test_augmentations.resize = resize
                config.test_augmentations.resize_interpolation = interp

                run_name = f"{run_name}-seed{seed}-resize-{resize}-{interp}"

                experiments[run_name] = [config]


gaussian_noise_levels = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2]
for l in gaussian_noise_levels:
    for run_name, modifieres in _experiments.items():
        config = modifieres[0]
        config = deepcopy(config)

        config.test_augmentations.gaussian_noise_sigma = l

        run_name = f"{run_name}-seed0-gaussian_noise-{l}"

        experiments[run_name] = [config]
