"""
Geometric / intensity transform split, mixed into SSLnnUNetDataModule.

Geometric transforms and intensity transforms are built separately so
that:
  - TrU's MultiViewUnlabeledDataLoader can run geometric once and
    intensity independently K times per sample (see that class's
    docstring for why the geometric draw must be shared).
  - TrL uses the same intensity op pool as TrU, but a different
    composition: each op independently rolls its own apply_probability
    (_build_intensity_transforms). TrU's strong view instead always
    selects and applies exactly STRONG_AUG_NUM_OPS of them
    (_build_intensity_transforms_strong) -- see that method's docstring
    for why independent per-op probabilities are the wrong tool for a
    view that must diverge from another view, not just from the
    original.

This split is used instead of nnU-Net's own bundled
nnUNetTrainer.get_training_transforms(). One known, accepted divergence
from that bundled pipeline: it runs MirrorTransform AFTER all 6
intensity transforms, while this split's geometric block runs
MirrorTransform BEFORE intensity. In practice this doesn't change the
augmented output *distribution* -- flipping is a coordinate relabeling,
and none of the intensity transforms here (noise, blur,
brightness/contrast, low-res, gamma) have any left/right-dependent
behavior, so applying them before or after a flip reaches the same set
of possible outputs, just via swapped left/right on any one draw.

TEE's intensity recipe is detuned from a CT-style default, per EAGT
(arxiv 2605.16427): strong brightness/contrast/gamma actively hurt
echocardiography segmentation, so ranges/probabilities here are
narrowed accordingly. GaussianNoiseTransform is swapped for
RicianNoiseTransform (signal-dependent, closer to ultrasound speckle
statistics than additive Gaussian). A brightness-gradient transform is
added to simulate depth/gain-dependent brightness falloff, a genuine
ultrasound-specific artifact.
"""

from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
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

from .transforms import RicianNoiseTransform, PickNTransforms

# Number of ops always selected (without replacement) and applied for
# TrU's strong view -- see
# TransformBuilderMixin._build_intensity_transforms_strong. Matches
# SegMatch (arxiv 2308.05232, semi-supervised surgical instrument
# segmentation via FixMatch-style weak/strong consistency), the closest
# literature match to this codebase's own weak/strong TrU setup.
STRONG_AUG_NUM_OPS = 3


class TransformBuilderMixin:
    """
    Mixed into SSLnnUNetDataModule. Relies on host attributes self.cm,
    self.ds_scales, and host method self._get_da_params_from_nnunet()
    -- see data_module.py.
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
        wrong (oversized) shape. Controlled by the single
        datamodule.transform_geometric flag in the yaml, shared by both
        TrL and TrU -- see data_module.py's train_dataloader(). Rotation/
        scaling is a straightforward generalization aid for TrL (target
        is always real ground truth, so a harder augmented example just
        makes training harder, never teaches something false). For TrU it
        was previously forced off by a separate flag over concern that an
        unusual rotation/scale on a self-trained pseudo-label could push
        the network into a confidently-wrong guess that then gets
        reinforced as its own training target. That risk is specific to
        single-view self-training (now removed); under the weak/strong
        consistency scheme training always runs, the strong view's
        prediction is checked against the weak view's pseudo-label rather
        than blindly trusted, so the two branches were unified onto one
        flag, shipped default True (rotation/scaling on for both).

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

    def _intensity_recipe(self):
        """
        Detuned intensity op pool for TEE ultrasound, per EAGT (arxiv
        2605.16427): a list of (transform, apply_probability) pairs --
        the probability is only used by _build_intensity_transforms
        (TrL); _build_intensity_transforms_strong (TrU strong view) uses
        the same transform instances but ignores the stored probability
        entirely.
        """

        return [
            (RicianNoiseTransform(noise_variance=(0, 0.05)), 0.1),
            (
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.0),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5,
                    benchmark=True,
                ),
                0.2,
            ),
            (
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.85, 1.15)), synchronize_channels=False, p_per_channel=1
                ),
                0.10,
            ),
            (
                ContrastTransform(
                    contrast_range=BGContrast((0.85, 1.15)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                0.10,
            ),
            (
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=None,
                    allowed_channels=None,
                    p_per_channel=0.5,
                ),
                0.25,
            ),
            (
                GammaTransform(
                    gamma=BGContrast((0.85, 1.2)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                0.05,
            ),
            (
                GammaTransform(
                    gamma=BGContrast((0.85, 1.2)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                0.15,
            ),
            (
                BrightnessGradientAdditiveTransform(
                    scale=(20, 60), max_strength=(0.05, 0.15), same_for_all_channels=True, mean_centered=True
                ),
                0.15,
            ),
        ]

    def _build_intensity_transforms(self):
        """
        TrL's intensity pipeline: each op in the task's recipe
        independently rolls its own apply_probability (RandomTransform),
        so anywhere from 0 to all of them can fire on a given call. Fine
        for TrL -- it's real ground truth, so a harder (or occasionally
        unaugmented) example is never actively harmful, only more or
        less useful.
        """

        recipe = self._intensity_recipe()

        return ComposeTransforms([RandomTransform(transform, apply_probability=p) for transform, p in recipe])

    def _build_intensity_transforms_strong(self, n=STRONG_AUG_NUM_OPS):
        """
        TrU strong view's intensity pipeline: always selects and applies
        exactly n ops from the task's recipe (PickNTransforms), ignoring
        the recipe's stored per-op probabilities entirely.

        Independent per-op probabilities (as used by
        _build_intensity_transforms, above) are the wrong tool here: they
        were designed for supervised training, where an occasional
        unaugmented sample is harmless. TrU's strong view instead must
        diverge from the weak view for the consistency loss to mean
        anything -- an unlucky draw where every op happens to skip
        produces a strong view identical to the weak view, silently
        wasting that training step. See PickNTransforms' docstring
        (transforms.py) for the literature this mirrors.
        """

        recipe = self._intensity_recipe()

        return PickNTransforms([transform for transform, _ in recipe], n=n)
