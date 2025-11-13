from glob import glob


def find_run_dir(run_name: str) -> str:
    runs = list(glob(f"runs/*/{run_name}", recursive=True))
    if len(runs) == 0:
        raise FileNotFoundError(f"Directory for run '{run_name}' is not found")
    if len(runs) > 1:
        raise FileExistsError(f"Multiple directories found for run '{run_name}': {runs}")
    return runs[0]


# Extend list definition with map function
class Files(list):
    def __init__(self, *files):
        # If a single non-string iterable is passed, use it directly; otherwise treat args as items
        if len(files) == 1 and not isinstance(files[0], (str, bytes)):
            super().__init__(files[0])
        else:
            super().__init__(files)

    def map(self, func):
        return Files(map(func, self))

    def unique(self):
        return Files(sorted(set(self)))

    def cat(self, other):
        return Files(self + other)


class FF:
    """https://arxiv.org/abs/1901.08971"""

    class DF:
        test = Files(
            "config/datasets/FF/test/DF.txt",
            "config/datasets/FF/test/real.txt",
        )

    class F2F:
        test = Files(
            "config/datasets/FF/test/F2F.txt",
            "config/datasets/FF/test/real.txt",
        )

    class FS:
        test = Files(
            "config/datasets/FF/test/FS.txt",
            "config/datasets/FF/test/real.txt",
        )

    class NT:
        test = Files(
            "config/datasets/FF/test/NT.txt",
            "config/datasets/FF/test/real.txt",
        )

    def to_train(f) -> str:
        return f.replace("/test/", "/train/")

    def to_val(f) -> str:
        return f.replace("/test/", "/val/")

    def to_x1_5(f) -> str:
        return f.replace("/FF/", "/FF-x1.5/")

    def to_x2(f) -> str:
        return f.replace("/FF/", "/FF-x2.0/")

    def to_rmbg_x1_5(f) -> str:
        return f.replace("/FF/", "/FF-rmbg-x1.5/")

    test = Files(DF.test + F2F.test + FS.test + NT.test).unique()
    train = test.map(to_train)
    val = test.map(to_val)


class CDFv2:
    """https://arxiv.org/abs/1909.12962"""

    test = Files(
        "config/datasets/CDFv2/test/Celeb-synthesis.txt",
        "config/datasets/CDFv2/test/YouTube-real.txt",
        "config/datasets/CDFv2/test/Celeb-real.txt",
    )

    # It is not an official validation set but generated from {all}\{test} files
    # using scripts/datasets/create_validation_set.py
    val = Files(
        "config/datasets/CDFv2/val/Celeb-synthesis.txt",
        "config/datasets/CDFv2/val/YouTube-real.txt",
        "config/datasets/CDFv2/val/Celeb-real.txt",
    )

    my_train = Files(
        "config/datasets/CDFv2/my-train/Celeb-synthesis.txt",
        "config/datasets/CDFv2/my-train/YouTube-real.txt",
        "config/datasets/CDFv2/my-train/Celeb-real.txt",
    )


class CDFv3:
    """https://arxiv.org/abs/2507.18015v1"""

    class FS:
        """Face-swap"""

        class CDFv2:
            test = Files(
                "config/datasets/CDFv3/test/Celeb-DF-v2.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class BlendFace:
            test = Files(
                "config/datasets/CDFv3/test/BlendFace.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class GHOST:
            test = Files(
                "config/datasets/CDFv3/test/GHOST.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class HifiFace:
            test = Files(
                "config/datasets/CDFv3/test/HifiFace.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class InSwapper:
            test = Files(
                "config/datasets/CDFv3/test/InSwapper.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class MobileFaceSwap:
            test = Files(
                "config/datasets/CDFv3/test/MobileFaceSwap.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class SimSwap:
            test = Files(
                "config/datasets/CDFv3/test/SimSwap.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class UniFace:
            test = Files(
                "config/datasets/CDFv3/test/UniFace.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        test = Files(
            CDFv2.test
            + BlendFace.test
            + GHOST.test
            + HifiFace.test
            + InSwapper.test
            + MobileFaceSwap.test
            + SimSwap.test
            + UniFace.test
        ).unique()

    class FR:
        """Face Reenectment"""

        class DaGAN:
            test = Files(
                "config/datasets/CDFv3/test/DaGAN.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class FSRT:
            test = Files(
                "config/datasets/CDFv3/test/FSRT.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class HyperReenact:
            test = Files(
                "config/datasets/CDFv3/test/HyperReenact.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class LIA:
            test = Files(
                "config/datasets/CDFv3/test/LIA.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class LivePortrait:
            test = Files(
                "config/datasets/CDFv3/test/LivePortrait.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class MCNET:
            test = Files(
                "config/datasets/CDFv3/test/MCNET.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class TPSMM:
            test = Files(
                "config/datasets/CDFv3/test/TPSMM.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        test = Files(
            DaGAN.test + FSRT.test + HyperReenact.test + LIA.test + LivePortrait.test + MCNET.test + TPSMM.test
        ).unique()

    class TF:
        """Talking Face"""

        class AniTalker:
            test = Files(
                "config/datasets/CDFv3/test/AniTalker.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class EchoMimic:
            test = Files(
                "config/datasets/CDFv3/test/EchoMimic.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class EDTalk:
            test = Files(
                "config/datasets/CDFv3/test/EDTalk.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class FLOAT:
            test = Files(
                "config/datasets/CDFv3/test/FLOAT.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class IP_LAP:
            test = Files(
                "config/datasets/CDFv3/test/IP_LAP.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class Real3DPortrait:
            test = Files(
                "config/datasets/CDFv3/test/Real3DPortrait.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class SadTalker:
            test = Files(
                "config/datasets/CDFv3/test/SadTalker.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        test = Files(
            AniTalker.test
            + EchoMimic.test
            + EDTalk.test
            + FLOAT.test
            + IP_LAP.test
            + Real3DPortrait.test
            + SadTalker.test
        ).unique()

    def to_train(f) -> str:
        return f.replace("/test/", "/train/")

    def to_my_val(f) -> str:
        return f.replace("/test/", "/my-val/")

    def to_x1_5(f) -> str:
        return f.replace("/CDFv3/", "/CDFv3-x1.5/")

    def to_x2(f) -> str:
        return f.replace("/CDFv3/", "/CDFv3-x2.0/")

    def to_rmbg_x1_5(f) -> str:
        return f.replace("/CDFv3/", "/CDFv3-rmbg-x1.5/")

    def to_x1_3_th0_5_all(f) -> str:
        return f.replace("/CDFv3/", "/CDFv3-x1.3-th0.5-all/")

    test = Files(FS.test + FR.test + TF.test).unique()
    train = test.map(to_train)
    my_val = test.map(to_my_val)

    @classmethod
    def get_test_dict(cls) -> dict[str, list[str]]:
        return {
            "CDFv3": cls.test,
            "CDFv3-FS": cls.FS.test,
            "CDFv3-FR": cls.FR.test,
            "CDFv3-TF": cls.TF.test,
            "CDFv3-FS-CDFv2": cls.FS.CDFv2.test,
            "CDFv3-FS-BlendFace": cls.FS.BlendFace.test,
            "CDFv3-FS-GHOST": cls.FS.GHOST.test,
            "CDFv3-FS-HifiFace": cls.FS.HifiFace.test,
            "CDFv3-FS-InSwapper": cls.FS.InSwapper.test,
            "CDFv3-FS-MobileFaceSwap": cls.FS.MobileFaceSwap.test,
            "CDFv3-FS-SimSwap": cls.FS.SimSwap.test,
            "CDFv3-FS-UniFace": cls.FS.UniFace.test,
            "CDFv3-FR-DaGAN": cls.FR.DaGAN.test,
            "CDFv3-FR-FSRT": cls.FR.FSRT.test,
            "CDFv3-FR-HyperReenact": cls.FR.HyperReenact.test,
            "CDFv3-FR-LIA": cls.FR.LIA.test,
            "CDFv3-FR-LivePortrait": cls.FR.LivePortrait.test,
            "CDFv3-FR-MCNET": cls.FR.MCNET.test,
            "CDFv3-FR-TPSMM": cls.FR.TPSMM.test,
            "CDFv3-TF-AniTalker": cls.TF.AniTalker.test,
            "CDFv3-TF-EchoMimic": cls.TF.EchoMimic.test,
            "CDFv3-TF-EDTalk": cls.TF.EDTalk.test,
            "CDFv3-TF-FLOAT": cls.TF.FLOAT.test,
            "CDFv3-TF-IP_LAP": cls.TF.IP_LAP.test,
            "CDFv3-TF-Real3DPortrait": cls.TF.Real3DPortrait.test,
            "CDFv3-TF-SadTalker": cls.TF.SadTalker.test,
        }


class DFD:
    test = Files(
        "config/datasets/DFD/fake.txt",
        "config/datasets/DFD/real.txt",
    )


class DFDC:
    test = Files(
        "config/datasets/DFDC/test/fake.txt",
        "config/datasets/DFDC/test/real.txt",
    )


class FSh:
    """
    FSh: https://github.com/maum-ai/faceshifter
    FF++: https://github.com/ondyari/FaceForensics
    """

    test = Files(
        "config/datasets/FSh/test/fake.txt",
        "config/datasets/FSh/test/real.txt",
    )


class UADFV:
    """https://arxiv.org/abs/1806.02877"""

    test = Files(
        "config/datasets/UADFD/fake.txt",
        "config/datasets/UADFD/real.txt",
    )


class DFDM:
    """https://arxiv.org/abs/2202.12951"""

    test = Files(
        "config/datasets/DFDM/all/dfaker.txt",
        "config/datasets/DFDM/all/dfl.txt",
        "config/datasets/DFDM/all/iae.txt",
        "config/datasets/DFDM/all/lightweight.txt",
        "config/datasets/CDFv2/all/Celeb-real.txt",
    )


class FFIW:
    """https://arxiv.org/abs/2103.16076"""

    test = Files(
        "config/datasets/FFIW/test-fake.txt",
        "config/datasets/FFIW/test-real.txt",
    )

    val = Files(
        "config/datasets/FFIW/val-fake.txt",
        "config/datasets/FFIW/val-real.txt",
    )

    train = Files(
        "config/datasets/FFIW/train-fake.txt",
        "config/datasets/FFIW/train-real.txt",
    )

    # My subsets of FFIW
    train_subset_1024 = Files(
        "config/datasets/FFIW/subsets/train-fake-subset-1024.txt",
        "config/datasets/FFIW/subsets/train-real-subset-1024.txt",
    )

    # My subsets of FFIW created using scripts/datasets/FFIW/create_FFIW_subset.py
    train_subset_2048 = Files(
        "config/datasets/FFIW/subset-2048/train-fake.txt",
        "config/datasets/FFIW/subset-2048/train-real.txt",
    )


class DeepSpeak_v1_1:
    """https://arxiv.org/abs/2408.05366"""

    test = Files(
        "config/datasets/DeepSpeak-1.1/test/test-facefusion_gan.txt",
        "config/datasets/DeepSpeak-1.1/test/test-facefusion_live.txt",
        "config/datasets/DeepSpeak-1.1/test/test-facefusion.txt",
        "config/datasets/DeepSpeak-1.1/test/test-real.txt",
        "config/datasets/DeepSpeak-1.1/test/test-retalking.txt",
        "config/datasets/DeepSpeak-1.1/test/test-wav2lip.txt",
    )

    train = test.map(lambda x: x.replace("/test/test-", "/train/train-"))

    # DeepSpeak-1.1 has a train folder. my-val is sampled from train
    my_val = test.map(lambda x: x.replace("/test/test-", "/my-val/val-"))

    # DeepSpeak-1.1 has a train folder. my-val is sampled from train, my-train is train \ my-val
    my_train = test.map(lambda x: x.replace("/test/test-", "/my-train/train-"))


class DeepSpeak_v2:
    """https://arxiv.org/abs/2408.05366"""

    test = Files(
        "config/datasets/DeepSpeak-2.0/test/test-diff2lip.txt",
        "config/datasets/DeepSpeak-2.0/test/test-facefusion.txt",
        "config/datasets/DeepSpeak-2.0/test/test-hellomeme.txt",
        "config/datasets/DeepSpeak-2.0/test/test-latentsync.txt",
        "config/datasets/DeepSpeak-2.0/test/test-liveportrait.txt",
        "config/datasets/DeepSpeak-2.0/test/test-memo.txt",
        "config/datasets/DeepSpeak-2.0/test/test-real.txt",
    )

    train = test.map(lambda x: x.replace("/test/test-", "/train/train-"))

    # DeepSpeak-2.0 has a train folder. my-val is sampled from train
    my_val = test.map(lambda x: x.replace("/test/test-", "/my-val/val-"))

    # DeepSpeak-2.0 has a train folder. my-val is sampled from train, my-train is train \ my-val
    my_train = test.map(lambda x: x.replace("/test/test-", "/my-train/train-"))


class KoDF:
    """https://arxiv.org/abs/2103.10094"""

    test = Files(
        "config/datasets/KoDF/real.txt",
        "config/datasets/KoDF/fake-audio-driven.txt",
        "config/datasets/KoDF/fake-dffs.txt",
        "config/datasets/KoDF/fake-dfl.txt",
        "config/datasets/KoDF/fake-fo.txt",
        "config/datasets/KoDF/fake-fsgan.txt",
    )

    adversarial = Files(
        "config/datasets/KoDF/fake-adv.txt",
        "config/datasets/KoDF/real-adv.txt",
    )


class FaceFusion:
    """Dataset created by VRG group"""

    class FF:
        train = Files(
            "config/datasets/FaceFusion/train/ff_inswapper_128_fp16.txt",
        )

    class CDF:
        test = Files(
            "config/datasets/FaceFusion/test/cdf_hififace_unofficial_256.txt",
            "config/datasets/FaceFusion/test/cdf_inswapper_128_fp16.txt",
            "config/datasets/CDFv2/test/YouTube-real.txt",
            "config/datasets/CDFv2/test/Celeb-real.txt",
        )


class VRG:
    """Dataset created by VRG group"""

    class CSFD:
        files = Files(
            "config/datasets/CSFD/real.txt",
        )


class AVSpeech:
    files = Files(
        "config/datasets/AVSpeech/real.txt",
    )


class FakeAVCeleb:
    """https://arxiv.org/abs/2108.05080"""

    test = Files(
        "config/datasets/FakeAVCeleb/FV-FA-faceswap-wav2lip.txt",
        "config/datasets/FakeAVCeleb/FV-FA-fsgan-wav2lip.txt",
        "config/datasets/FakeAVCeleb/FV-FA-wav2lip.txt",
        "config/datasets/FakeAVCeleb/FV-RA-faceswap.txt",
        "config/datasets/FakeAVCeleb/FV-RA-fsgan.txt",
        "config/datasets/FakeAVCeleb/FV-RA-wav2lip.txt",
        "config/datasets/FakeAVCeleb/RV-RA-real.txt",
    )

    class FV_RA_WL:
        test = Files(
            "config/datasets/FakeAVCeleb/FV-RA-wav2lip.txt",
            "config/datasets/FakeAVCeleb/RV-RA-real.txt",
        )

    class FV_FA_FS:
        test = Files(
            "config/datasets/FakeAVCeleb/FV-FA-faceswap-wav2lip.txt",
            "config/datasets/FakeAVCeleb/RV-RA-real.txt",
        )

    class FV_FA_GAN:
        test = Files(
            "config/datasets/FakeAVCeleb/FV-FA-fsgan-wav2lip.txt",
            "config/datasets/FakeAVCeleb/RV-RA-real.txt",
        )

    class FV_FA_WL:
        test = Files(
            "config/datasets/FakeAVCeleb/FV-FA-wav2lip.txt",
            "config/datasets/FakeAVCeleb/RV-RA-real.txt",
        )


class PolyGlotFake:
    """https://arxiv.org/abs/2405.08838"""

    test = Files(
        "config/datasets/PolyGlotFake/real-ar.txt",
        "config/datasets/PolyGlotFake/real-en.txt",
        "config/datasets/PolyGlotFake/real-es.txt",
        "config/datasets/PolyGlotFake/real-fr.txt",
        "config/datasets/PolyGlotFake/real-ja.txt",
        "config/datasets/PolyGlotFake/real-ru.txt",
        "config/datasets/PolyGlotFake/real-zh.txt",
        "config/datasets/PolyGlotFake/ar2en_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ar2en_video_retalking.txt",
        "config/datasets/PolyGlotFake/ar2es_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ar2es_video_retalking.txt",
        "config/datasets/PolyGlotFake/ar2fr_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ar2fr_video_retalking.txt",
        "config/datasets/PolyGlotFake/ar2ja_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ar2ja_video_retalking.txt",
        "config/datasets/PolyGlotFake/ar2ru_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ar2ru_video_retalking.txt",
        "config/datasets/PolyGlotFake/ar2zh_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ar2zh_video_retalking.txt",
        "config/datasets/PolyGlotFake/en2ar_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/en2ar_video_retalking.txt",
        "config/datasets/PolyGlotFake/en2es_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/en2es_video_retalking.txt",
        "config/datasets/PolyGlotFake/en2fr_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/en2fr_video_retalking.txt",
        "config/datasets/PolyGlotFake/en2ja_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/en2ja_video_retalking.txt",
        "config/datasets/PolyGlotFake/en2ru_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/en2ru_video_retalking.txt",
        "config/datasets/PolyGlotFake/en2zh_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/en2zh_video_retalking.txt",
        "config/datasets/PolyGlotFake/es2ar_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/es2ar_video_retalking.txt",
        "config/datasets/PolyGlotFake/es2en_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/es2en_video_retalking.txt",
        "config/datasets/PolyGlotFake/es2fr_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/es2fr_video_retalking.txt",
        "config/datasets/PolyGlotFake/es2ja_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/es2ja_video_retalking.txt",
        "config/datasets/PolyGlotFake/es2ru_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/es2ru_video_retalking.txt",
        "config/datasets/PolyGlotFake/es2zh_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/es2zh_video_retalking.txt",
        "config/datasets/PolyGlotFake/fr2ar_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/fr2ar_video_retalking.txt",
        "config/datasets/PolyGlotFake/fr2en_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/fr2en_video_retalking.txt",
        "config/datasets/PolyGlotFake/fr2es_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/fr2es_video_retalking.txt",
        "config/datasets/PolyGlotFake/fr2ja_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/fr2ja_video_retalking.txt",
        "config/datasets/PolyGlotFake/fr2ru_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/fr2ru_video_retalking.txt",
        "config/datasets/PolyGlotFake/fr2zh_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/fr2zh_video_retalking.txt",
        "config/datasets/PolyGlotFake/ja2ar_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ja2ar_video_retalking.txt",
        "config/datasets/PolyGlotFake/ja2en_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ja2en_video_retalking.txt",
        "config/datasets/PolyGlotFake/ja2es_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ja2es_video_retalking.txt",
        "config/datasets/PolyGlotFake/ja2fr_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ja2fr_video_retalking.txt",
        "config/datasets/PolyGlotFake/ja2ru_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ja2ru_video_retalking.txt",
        "config/datasets/PolyGlotFake/ja2zh_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ja2zh_video_retalking.txt",
        "config/datasets/PolyGlotFake/ru2ar_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ru2ar_video_retalking.txt",
        "config/datasets/PolyGlotFake/ru2en_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ru2en_video_retalking.txt",
        "config/datasets/PolyGlotFake/ru2es_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ru2es_video_retalking.txt",
        "config/datasets/PolyGlotFake/ru2fr_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ru2fr_video_retalking.txt",
        "config/datasets/PolyGlotFake/ru2ja_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ru2zh_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ru2zh_video_retalking.txt",
        "config/datasets/PolyGlotFake/zh2ar_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/zh2ar_video_retalking.txt",
        "config/datasets/PolyGlotFake/zh2en_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/zh2en_video_retalking.txt",
        "config/datasets/PolyGlotFake/zh2es_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/zh2es_video_retalking.txt",
        "config/datasets/PolyGlotFake/zh2fr_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/zh2fr_video_retalking.txt",
        "config/datasets/PolyGlotFake/zh2ja_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/zh2ru_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/zh2ru_video_retalking.txt",
    )


class IDForge_v1:
    """https://arxiv.org/abs/2401.11764"""

    train = Files(
        "config/datasets/IDForge-v1/train/train-face_tts_infoswap.txt",
        "config/datasets/IDForge-v1/train/train-face_tts_roop.txt",
        "config/datasets/IDForge-v1/train/train-face_tts_simswap.txt",
        "config/datasets/IDForge-v1/train/train-real.txt",
    )

    val = Files(
        "config/datasets/IDForge-v1/val/val-face_tts_infoswap.txt",
        "config/datasets/IDForge-v1/val/val-face_tts_roop.txt",
        "config/datasets/IDForge-v1/val/val-face_tts_simswap.txt",
        "config/datasets/IDForge-v1/val/val-real.txt",
    )

    test = Files(
        "config/datasets/IDForge-v1/test/test-face_tts_infoswap.txt",
        "config/datasets/IDForge-v1/test/test-face_tts_roop.txt",
        "config/datasets/IDForge-v1/test/test-face_tts_simswap.txt",
        "config/datasets/IDForge-v1/test/test-real.txt",
    )


class DF40:
    """https://arxiv.org/abs/2406.13495"""

    class CDF:
        class SadTalker:
            test = Files(
                "config/datasets/DF40/test/test_cdf_sadtalker.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )

        class FOMM:
            test = Files(
                "config/datasets/DF40/test/test_cdf_fomm.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )

        class FaceDancer:
            test = Files(
                "config/datasets/DF40/test/test_cdf_facedancer.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )

        class MobileSwap:
            test = Files(
                "config/datasets/DF40/test/test_cdf_mobileswap.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )

        class SimSwap:
            test = Files(
                "config/datasets/DF40/test/test_cdf_simswap.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )

        class InSwapper:
            test = Files(
                "config/datasets/DF40/test/test_cdf_inswap.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )

        class Uniface:
            test = Files(
                "config/datasets/DF40/test/test_cdf_uniface.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )


class FFv2:
    """
    FaceFusion v2 dataset created by VRG group
    """

    class FF:
        train = Files(
            "config/datasets/FF/train/real.txt",
            "config/datasets/FFv2/train/FF_blendswap_256.txt",
            "config/datasets/FFv2/train/FF_ghost_1_256.txt",
            # "config/datasets/FFv2/train/FF_ghost_2_256.txt",
            # "config/datasets/FFv2/train/FF_ghost_3_256.txt",
            "config/datasets/FFv2/train/FF_hififace_unofficial_256.txt",
            "config/datasets/FFv2/train/FF_hyperswap_1a_256.txt",
            # "config/datasets/FFv2/train/FF_hyperswap_1c_256.txt",
            "config/datasets/FFv2/train/FF_inswapper_128_fp16.txt",
            # "config/datasets/FFv2/train/FF_inswapper_128.txt",
            "config/datasets/FFv2/train/FF_simswap_256.txt",
            # "config/datasets/FFv2/train/FF_simswap_unofficial_512.txt",
            "config/datasets/FFv2/train/FF_uniface_256.txt",
        )

    class SS:
        train = Files(
            "config/datasets/FF/train/real.txt",
            "config/datasets/FFv2/train/SS_blendswap_256.txt",
            "config/datasets/FFv2/train/SS_ghost_1_256.txt",
            # "config/datasets/FFv2/train/SS_ghost_2_256.txt",
            # "config/datasets/FFv2/train/SS_ghost_3_256.txt",
            "config/datasets/FFv2/train/SS_hififace_unofficial_256.txt",
            "config/datasets/FFv2/train/SS_hyperswap_1a_256.txt",
            # "config/datasets/FFv2/train/SS_hyperswap_1c_256.txt",
            "config/datasets/FFv2/train/SS_inswapper_128_fp16.txt",
            # "config/datasets/FFv2/train/SS_inswapper_128.txt",
            "config/datasets/FFv2/train/SS_simswap_256.txt",
            # "config/datasets/FFv2/train/SS_simswap_unofficial_512.txt",
            "config/datasets/FFv2/train/SS_uniface_256.txt",
        )


if __name__ == "__main__":
    import pandas as pd

    def get_video(file_path: str) -> str:
        return file_path.split("/")[-2]

    val_files = [
        # *CDFv3.test.map(CDFv3.to_train).map(CDFv3.to_x1_5),
        # *FF.train.map(FF.to_x1_5),
        # *FF.train.map(FF.to_rmbg_x1_5),
        # *CDFv3.train.map(CDFv3.to_x1_3_th0_5_all),
        # *DeepSpeak_v1_1.train
        # *DeepSpeak_v2.train.cat(DeepSpeak_v1_1.train)
        *FF.train.map(lambda x: x.replace("/FF/", "/FF-x1.3-th0.5-all/subset/1st-frame/")),
        *DeepSpeak_v1_1.train.map(lambda x: x.replace("/DeepSpeak-1.1/", "/DeepSpeak-1.1/subset/1st-frame/")),
        *DeepSpeak_v2.train.map(lambda x: x.replace("/DeepSpeak-2.0/", "/DeepSpeak-2.0/subset/1st-frame/")),
        *FFIW.train.map(lambda x: x.replace("/FFIW/", "/FFIW/subset/1st-frame/")),
    ]

    total_videos = 0
    for file in val_files:
        # read with pandas
        df = pd.read_csv(file, names=["files"])

        df["video"] = df["files"].apply(lambda x: get_video(x))

        # unique values
        unique_videos = df["video"].unique()

        print(f"Unique videos in {file} : {len(unique_videos)}")

        total_videos += len(unique_videos)

    print(f"Total unique videos: {total_videos}")
