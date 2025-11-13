from torchvision.transforms import v2 as T

from src.config import Augmentations


def init_augmentations(augs: Augmentations):
    # TODO: for each augmentation, add a probability parameter to the config
    if augs is None:
        return None

    composed_transforms = []

    if augs.random_horizontal_flip != 0.0:
        composed_transforms.append(T.RandomHorizontalFlip(p=augs.random_horizontal_flip))

    if (
        augs.random_affine_degrees != 0
        or augs.random_affine_translate is not None
        or augs.random_affine_scale is not None
    ):
        composed_transforms.append(
            T.RandomAffine(
                degrees=augs.random_affine_degrees,
                translate=augs.random_affine_translate,
                scale=augs.random_affine_scale,
            )
        )

    if augs.gaussian_blur_prob != 0.0:
        ks = augs.gaussian_blur_kernel_size
        if (isinstance(ks, int) and ks != 0) or (isinstance(ks, list) and sum(ks) != 0):
            composed_transforms.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=ks, sigma=augs.gaussian_blur_sigma)],
                    p=augs.gaussian_blur_prob,
                )
            )

    if augs.color_jitter_brightness != 0.0 or augs.color_jitter_contrast != 0.0:
        composed_transforms.append(
            T.ColorJitter(
                brightness=augs.color_jitter_brightness,
                contrast=augs.color_jitter_contrast,
            )
        )

    if (isinstance(augs.jpeg_quality, int) and augs.jpeg_quality != 100) or (
        isinstance(augs.jpeg_quality, list) and augs.jpeg_quality[0] != 100
    ):
        composed_transforms.append(T.JPEG(augs.jpeg_quality))

    if augs.resize is not None:
        composed_transforms.append(T.Resize(augs.resize, augs.resize_interpolation))

    if augs.gaussian_noise_sigma != 0.0:
        composed_transforms.append(
            T.Compose(
                [
                    T.ToTensor(),
                    T.GaussianNoise(0.0, augs.gaussian_noise_sigma),
                    T.ToPILImage(),
                ]
            )
        )

    if len(composed_transforms) == 0:
        return None

    return T.Compose(composed_transforms)
