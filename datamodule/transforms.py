"""
Custom batchgeneratorsv2-style intensity transforms for the SSL dataloader.

RicianNoiseTransform is a local, import-fixed copy of
batchgeneratorsv2.transforms.noise.rician.RicianNoiseTransform (installed
v0.3.4): upstream's get_parameters() calls np.random.uniform but the
module never imports numpy at module scope, so calling it raises
NameError. Verified directly against the installed package. Algorithm is
copied verbatim from upstream -- only the missing import is fixed here,
so this stays a drop-in replacement if/when upstream fixes it.

PickNTransforms is a generic composition container (not task-specific)
used to build TrU's strong-view intensity pipeline -- see its own
docstring and transform_builders.py's _build_intensity_transforms_strong.
"""

from typing import Tuple

import numpy as np
import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform, ImageOnlyTransform


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
