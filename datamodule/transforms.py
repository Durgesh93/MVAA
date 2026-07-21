"""
Custom batchgeneratorsv2-style intensity transforms for the SSL dataloader.

RicianNoiseTransform is a local, import-fixed copy of
batchgeneratorsv2.transforms.noise.rician.RicianNoiseTransform (installed
v0.3.4): upstream's get_parameters() calls np.random.uniform but the
module never imports numpy at module scope, so calling it raises
NameError. Verified directly against the installed package. Algorithm is
copied verbatim from upstream -- only the missing import is fixed here,
so this stays a drop-in replacement if/when upstream fixes it.

SmokeHazeTransform and BleedingBlobTransform are new, targeting the two
corruption modes SegSTRONG-C (arxiv 2407.11906) found dominant in real
surgical video besides low brightness (which the existing
MultiplicativeBrightnessTransform already covers by biasing its range
down -- see datamodule.py's video intensity builder). Both compute their
reference intensity relative to the sample's own mean/std rather than an
absolute constant, since these datasets are Z-score normalized.

PickNTransforms is a generic composition container (not task-specific)
used to build TrU's strong-view intensity pipeline -- see its own
docstring and transform_builders.py's _build_intensity_transforms_strong.
"""

from typing import Tuple

import numpy as np
import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform, ImageOnlyTransform
from batchgeneratorsv2.transforms.local.local_transform import LocalTransform
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar


class RicianNoiseTransform(ImageOnlyTransform):
    """
    Adds Rician noise (signal-dependent, closer to ultrasound speckle
    statistics than additive Gaussian noise) -- see module docstring for
    why this is a local copy rather than importing upstream directly.
    """

    def __init__(self, noise_variance: Tuple[float, float] = (0.0, 0.1)):
        super().__init__()
        self.noise_variance = noise_variance

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        variance = float(np.random.uniform(*self.noise_variance))
        return {"variance": variance}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        var = params["variance"]
        noise_real = torch.empty_like(img).normal_(mean=0.0, std=var)
        noise_imag = torch.empty_like(img).normal_(mean=0.0, std=var)

        min_val = img.min()
        shifted = img - min_val

        rician = torch.sqrt((shifted + noise_real).pow_(2).add_(noise_imag.pow_(2)))
        rician = rician + min_val

        input_mean, input_std = img.mean(), img.std()
        rician_mean, rician_std = rician.mean(), rician.std()

        if rician_std > 0:
            rician = (rician - rician_mean) / rician_std * input_std + input_mean
        else:
            rician = rician * 0 + input_mean

        return rician


class SmokeHazeTransform(ImageOnlyTransform):
    """
    Global atmospheric-scattering-style haze blend: img = img*(1-alpha) +
    light*alpha, where "light" is computed from the image's own per-
    channel mean/std (light_offset standard deviations above the mean)
    rather than a fixed absolute constant, since channels are Z-score
    normalized -- an absolute "white" pixel value would be meaningless.
    """

    def __init__(self, alpha_range: Tuple[float, float] = (0.1, 0.4), light_offset: float = 2.0):
        super().__init__()
        self.alpha_range = alpha_range
        self.light_offset = light_offset

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        alpha = float(np.random.uniform(*self.alpha_range))
        return {"alpha": alpha}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        alpha = params["alpha"]

        spatial_dims = tuple(range(1, img.ndim))
        channel_mean = img.mean(dim=spatial_dims, keepdim=True)
        channel_std = img.std(dim=spatial_dims, keepdim=True)
        light = channel_mean + self.light_offset * channel_std

        return img * (1.0 - alpha) + light * alpha


class BleedingBlobTransform(ImageOnlyTransform, LocalTransform):
    """
    Localized red-tinted blob simulating bleeding in the surgical field:
    one shared spatial kernel (from LocalTransform._generate_kernel)
    across channels, with a per-channel bias so a patch shifts toward
    red rather than just brightening uniformly. channel_bias order must
    match the dataset's channel_names (video's dataset.json has
    {"0": "red", "1": "green", "2": "blue"}).
    """

    def __init__(
        self,
        scale: RandomScalar = (20, 80),
        loc: RandomScalar = (-0.2, 1.2),
        max_strength: RandomScalar = (0.3, 0.8),
        channel_bias: Tuple[float, float, float] = (1.0, -0.4, -0.4),
    ):
        ImageOnlyTransform.__init__(self)
        LocalTransform.__init__(self, scale, loc)

        self.max_strength = max_strength
        self.channel_bias = channel_bias

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C, *spatial = image.shape

        if C != len(self.channel_bias):
            raise ValueError(
                f"BleedingBlobTransform.channel_bias has {len(self.channel_bias)} entries "
                f"but the image has {C} channels -- channel_bias must match the RGB channel count."
            )

        kernel = self._generate_kernel(spatial)
        strength = sample_scalar(self.max_strength, image, kernel)

        return {"kernel": kernel, "strength": strength}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        kernel = params["kernel"]
        strength = params["strength"]

        kernel_tensor = torch.from_numpy(kernel).to(img.device, dtype=img.dtype)

        for c, bias in enumerate(self.channel_bias):
            img[c] = img[c] + kernel_tensor * bias * strength

        return img


class PickNTransforms(BasicTransform):
    """
    Always selects exactly n transforms (without replacement) from the
    given pool and applies all of them unconditionally -- unlike
    RandomTransform-wrapped independent per-op probabilities (nnU-Net's
    own supervised-training convention), there is no chance of a no-op
    draw, since selection itself is the only gate.

    That matters specifically for TrU's strong view in weak/strong
    consistency training: an independent-probability pipeline can (and,
    empirically, does at a non-trivial rate -- see the datamodule's
    verification scripts) produce a strong view identical to the weak
    view, wasting that training step's consistency signal entirely. This
    mirrors the strong-augmentation convention used by RandAugment/
    FixMatch and, more directly, SegMatch (arxiv 2308.05232,
    semi-supervised surgical instrument segmentation via FixMatch-style
    weak/strong consistency), which always applies exactly 3 selected
    photometric ops rather than gating each one by its own probability.
    """

    def __init__(self, transforms, n):
        super().__init__()
        self.transforms = list(transforms)
        self.n = int(n)

        if self.n < 1 or self.n > len(self.transforms):
            raise ValueError(f"n must be between 1 and {len(self.transforms)}, got {self.n}")

    def apply(self, data_dict, **params):
        chosen = np.random.choice(len(self.transforms), size=self.n, replace=False)

        for i in chosen:
            data_dict = self.transforms[i](**data_dict)

        return data_dict
