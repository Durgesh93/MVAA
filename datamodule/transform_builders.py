"""
Geometric / intensity transform split, mixed into SSLnnUNetDataModule.

Geometric transforms (reused unchanged across all 3 tasks) and intensity
transforms (task-specific) are built separately so that:
  - TrU's MultiViewUnlabeledDataLoader can run geometric once and
    intensity independently K times per sample (see that class's
    docstring for why the geometric draw must be shared).
  - TrL (all 3 tasks, including CT) reuses the exact same per-task
    intensity pipeline as TrU, just with a single draw (K=1).

All three tasks -- CT included -- go through this split, not nnU-Net's
own bundled nnUNetTrainer.get_training_transforms(). One known,
accepted divergence from that bundled pipeline: it runs MirrorTransform
AFTER all 6 intensity transforms, while this split's geometric block
runs MirrorTransform BEFORE intensity. In practice this doesn't change
the augmented output *distribution* -- flipping is a coordinate
relabeling, and none of the intensity transforms here (noise, blur,
brightness/contrast, low-res, gamma) have any left/right-dependent
behavior, so applying them before or after a flip reaches the same set
of possible outputs, just via swapped left/right on any one draw.

TEE/video intensity design is grounded in real research, not guesses:
  - TEE: EAGT (arxiv 2605.16427) found strong intensity augmentation
    actively hurts echocardiography segmentation -- ranges/probabilities
    here are narrowed from CT's defaults accordingly, plus
    RicianNoiseTransform (signal-dependent, closer to ultrasound speckle
    than additive Gaussian) and a brightness-gradient transform for
    depth/gain falloff.
  - Video: SegSTRONG-C (arxiv 2407.11906) found smoke, bleeding, and low
    brightness are the dominant real-world surgical video corruptions --
    handled via a darker-biased brightness draw plus the custom
    SmokeHazeTransform/BleedingBlobTransform (see transforms.py).
"""

from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.local.brightness_gradient import BrightnessGradientAdditiveTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

from transforms import RicianNoiseTransform, SmokeHazeTransform, BleedingBlobTransform


class TransformBuilderMixin:
    """
    Mixed into SSLnnUNetDataModule. Relies on host attributes self.cm,
    self.ds_scales, self.prefix, and host method
    self._get_da_params_from_nnunet() -- see data_module.py.
    """

    def _build_geometric_transforms(self, use_spatial_transform: bool = True):
        """
        Task-agnostic geometric transform list, ported from nnU-Net's own
        get_training_transforms (crop-time Convert3DTo2D/back if dummy 2D
        DA, SpatialTransform for rotation+scaling, MirrorTransform,
        MaskImageTransform if configured, plus the deterministic
        bookkeeping transforms RemoveLabelTansform/DownsampleSegForDSTransform).

        use_spatial_transform: gates SpatialTransform's actual rotation
        and scaling randomness (p_rotation/p_scaling forced to 0 when
        False) -- crop/translation and MirrorTransform are always applied
        regardless (cheap, safe: CNNs are translation-equivariant by
        construction, so those don't carry the same risk). SpatialTransform
        itself always stays in the pipeline even when disabled: besides
        rotation/scaling, it's also what crops the oversized initial patch
        (init_ps, enlarged by _get_da_params_from_nnunet specifically to
        leave rotation margin) down to the real patch_size the network
        expects -- omitting the transform entirely leaves samples at the
        wrong (oversized) shape. Controlled per labeled/unlabeled branch
        via datamodule.use_spatial_transform_trl/_tru in the yaml -- see
        data_module.py's train_dataloader(). Rotation/scaling is a real
        generalization aid for TrL (target is always real ground truth,
        so a harder augmented example just makes training harder, never
        teaches something false). For TrU's self-training pseudo-labels
        it's a pure risk with no offsetting benefit here: this datamodule
        does single-view entropy minimization (no cross-view consistency
        check), so an unusual rotation/scale can push the network into a
        confidently-wrong guess that then gets reinforced as its own
        training target -- the classic self-training confirmation-bias
        failure mode.

        Cascade (is_cascaded) and region-based (self.lm.has_regions)
        branches of get_training_transforms are intentionally omitted:
        this datamodule raises in __init__ if cascaded, and
        NNUnetSetup.build_loss asserts not self.lm.has_regions.
        """

        rot, dummy_2d, init_ps, mirror = self._get_da_params_from_nnunet()

        patch_size_spatial = self.cm.patch_size[1:] if dummy_2d else self.cm.patch_size

        geometric = []

        if dummy_2d:
            geometric.append(Convert3DTo2DTransform())

        geometric.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.2 if use_spatial_transform else 0.0,
                rotation=rot,
                p_scaling=0.2 if use_spatial_transform else 0.0,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                border_mode_seg="constant",
                padding_value_seg=-1,
            )
        )

        if dummy_2d:
            geometric.append(Convert2DTo3DTransform())

        if mirror is not None and len(mirror) > 0:
            geometric.append(MirrorTransform(allowed_axes=mirror))

        if self.cm.use_mask_for_norm is not None and any(self.cm.use_mask_for_norm):
            geometric.append(
                MaskImageTransform(
                    apply_to_channels=[i for i in range(len(self.cm.use_mask_for_norm)) if self.cm.use_mask_for_norm[i]],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        geometric.append(RemoveLabelTansform(-1, 0))

        if self.ds_scales is not None:
            geometric.append(DownsampleSegForDSTransform(ds_scales=self.ds_scales))

        return ComposeTransforms(geometric)

    def _build_intensity_transforms_ct(self):
        """nnU-Net's default intensity transform list, unchanged."""

        return ComposeTransforms(
            [
                RandomTransform(
                    GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True),
                    apply_probability=0.1,
                ),
                RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.5, 1.0),
                        synchronize_channels=False,
                        synchronize_axes=False,
                        p_per_channel=0.5,
                        benchmark=True,
                    ),
                    apply_probability=0.2,
                ),
                RandomTransform(
                    MultiplicativeBrightnessTransform(
                        multiplier_range=BGContrast((0.75, 1.25)), synchronize_channels=False, p_per_channel=1
                    ),
                    apply_probability=0.15,
                ),
                RandomTransform(
                    ContrastTransform(
                        contrast_range=BGContrast((0.75, 1.25)),
                        preserve_range=True,
                        synchronize_channels=False,
                        p_per_channel=1,
                    ),
                    apply_probability=0.15,
                ),
                RandomTransform(
                    SimulateLowResolutionTransform(
                        scale=(0.5, 1),
                        synchronize_channels=False,
                        synchronize_axes=True,
                        ignore_axes=None,
                        allowed_channels=None,
                        p_per_channel=0.5,
                    ),
                    apply_probability=0.25,
                ),
                RandomTransform(
                    GammaTransform(
                        gamma=BGContrast((0.7, 1.5)),
                        p_invert_image=1,
                        synchronize_channels=False,
                        p_per_channel=1,
                        p_retain_stats=1,
                    ),
                    apply_probability=0.1,
                ),
                RandomTransform(
                    GammaTransform(
                        gamma=BGContrast((0.7, 1.5)),
                        p_invert_image=0,
                        synchronize_channels=False,
                        p_per_channel=1,
                        p_retain_stats=1,
                    ),
                    apply_probability=0.3,
                ),
            ]
        )

    def _build_intensity_transforms_tee(self):
        """
        Detuned version of the CT default, per EAGT (arxiv 2605.16427):
        strong brightness/contrast/gamma actively hurt echocardiography
        segmentation, so ranges/probabilities are narrowed here rather
        than left at CT's defaults. GaussianNoiseTransform is swapped for
        RicianNoiseTransform (signal-dependent, closer to ultrasound
        speckle statistics than additive Gaussian). A brightness-gradient
        transform is added to simulate depth/gain-dependent brightness
        falloff, a genuine ultrasound-specific artifact.
        """

        return ComposeTransforms(
            [
                RandomTransform(
                    RicianNoiseTransform(noise_variance=(0, 0.05)),
                    apply_probability=0.1,
                ),
                RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.5, 1.0),
                        synchronize_channels=False,
                        synchronize_axes=False,
                        p_per_channel=0.5,
                        benchmark=True,
                    ),
                    apply_probability=0.2,
                ),
                RandomTransform(
                    MultiplicativeBrightnessTransform(
                        multiplier_range=BGContrast((0.85, 1.15)), synchronize_channels=False, p_per_channel=1
                    ),
                    apply_probability=0.10,
                ),
                RandomTransform(
                    ContrastTransform(
                        contrast_range=BGContrast((0.85, 1.15)),
                        preserve_range=True,
                        synchronize_channels=False,
                        p_per_channel=1,
                    ),
                    apply_probability=0.10,
                ),
                RandomTransform(
                    SimulateLowResolutionTransform(
                        scale=(0.5, 1),
                        synchronize_channels=False,
                        synchronize_axes=True,
                        ignore_axes=None,
                        allowed_channels=None,
                        p_per_channel=0.5,
                    ),
                    apply_probability=0.25,
                ),
                RandomTransform(
                    GammaTransform(
                        gamma=BGContrast((0.85, 1.2)),
                        p_invert_image=1,
                        synchronize_channels=False,
                        p_per_channel=1,
                        p_retain_stats=1,
                    ),
                    apply_probability=0.05,
                ),
                RandomTransform(
                    GammaTransform(
                        gamma=BGContrast((0.85, 1.2)),
                        p_invert_image=0,
                        synchronize_channels=False,
                        p_per_channel=1,
                        p_retain_stats=1,
                    ),
                    apply_probability=0.15,
                ),
                RandomTransform(
                    BrightnessGradientAdditiveTransform(
                        scale=(20, 60), max_strength=(0.05, 0.15), same_for_all_channels=True, mean_centered=True
                    ),
                    apply_probability=0.15,
                ),
            ]
        )

    def _build_intensity_transforms_video(self):
        """
        CT's default intensity list (video is RGB camera footage, closer
        to natural-image domain than CT/TEE) plus transforms targeting
        the three corruption modes SegSTRONG-C (arxiv 2407.11906) found
        dominant in real surgical video: a darker-biased brightness draw
        for low-brightness, and SmokeHazeTransform/BleedingBlobTransform
        (custom, see transforms.py) for smoke and bleeding.
        """

        base = self._build_intensity_transforms_ct()

        extra = ComposeTransforms(
            [
                RandomTransform(
                    MultiplicativeBrightnessTransform(
                        multiplier_range=BGContrast((0.5, 0.85)), synchronize_channels=False, p_per_channel=1
                    ),
                    apply_probability=0.1,
                ),
                RandomTransform(SmokeHazeTransform(alpha_range=(0.1, 0.4), light_offset=2.0), apply_probability=0.1),
                RandomTransform(
                    BleedingBlobTransform(
                        scale=(20, 80), loc=(-0.2, 1.2), max_strength=(0.3, 0.8), channel_bias=(1.0, -0.4, -0.4)
                    ),
                    apply_probability=0.1,
                ),
            ]
        )

        return ComposeTransforms([base, extra])

    def _build_intensity_transforms(self):
        if self.prefix == "t1_ct":
            return self._build_intensity_transforms_ct()
        if self.prefix == "t2_tee":
            return self._build_intensity_transforms_tee()
        if self.prefix == "t3_vid":
            return self._build_intensity_transforms_video()

        raise ValueError(f"Unknown prefix '{self.prefix}' -- expected one of t1_ct, t2_tee, t3_vid.")
