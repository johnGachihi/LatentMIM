import random
from typing import Tuple
from PIL import ImageFilter, ImageOps

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


def paired_random_crop_resize(
    hr_img, lr_img,
    size: Tuple[int, int], hr_img_size=None,
    scale=(0.3, 1.0), ratio=(3 / 4, 4 / 3)
):
    """
    Apply same random crop to both hr and lr images, then resize to target size.
    Used for paired images like Venus (HR) and Sentinel2 (LR).
    """
    if hr_img_size is None:
        hr_img_size = size

    def lr_crop_params(i, j, h, w):
        hr_h, hr_w = hr_img.shape[-2:]
        lr_h, lr_w = lr_img.shape[-2:]

        scaled_i = int(i * lr_h / hr_h)
        scaled_j = int(j * lr_w / hr_w)
        scaled_h = int(h * lr_h / hr_h)
        scaled_w = int(w * lr_w / hr_w)

        return scaled_i, scaled_j, scaled_h, scaled_w

    i, j, h, w = T.RandomResizedCrop.get_params(hr_img, scale=scale, ratio=ratio)
    lr_i, lr_j, lr_h, lr_w = lr_crop_params(i, j, h, w)

    hr_img = TF.crop(hr_img, i, j, h, w)
    lr_img = TF.crop(lr_img, lr_i, lr_j, lr_h, lr_w)

    hr_img = TF.resize(hr_img, hr_img_size)
    lr_img = TF.resize(lr_img, size, interpolation=T.InterpolationMode.NEAREST)

    return hr_img, lr_img


def paired_resize(hr_img, lr_img, size: Tuple[int, int], hr_img_size=None):
    """
    Resize both hr and lr images to the target size.
    Used for paired images like Venus (HR) and Sentinel2 (LR).
    """
    if hr_img_size is None:
        hr_img_size = size

    hr_img = TF.resize(hr_img, hr_img_size)
    lr_img = TF.resize(lr_img, size, interpolation=T.InterpolationMode.NEAREST)

    return hr_img, lr_img


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


def random_crop_resize_img_and_mask(
    img, mask,
    size: Tuple[int, int],
    scale=(0.3, 1.0), ratio=(3 / 4, 4 / 3)
):
    """
    Apply same random crop to both image and mask, then resize to target size.
    Mask is resized using nearest neighbor interpolation to preserve label values.
    """
    i, j, h, w = T.RandomResizedCrop.get_params(img, scale=scale, ratio=ratio)

    img = TF.crop(img, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    img = TF.resize(img, size)
    mask = TF.resize(mask, size, interpolation=T.InterpolationMode.NEAREST)

    return img, mask


def resize_img_and_mask(img, mask, size: Tuple[int, int]):
    """
    Resize both image and mask to the target size.
    Mask is resized using nearest neighbor interpolation to preserve label values.
    """
    img = TF.resize(img, size)
    mask = TF.resize(mask, size, interpolation=T.InterpolationMode.NEAREST)

    return img, mask
