"""
TrU (unlabeled) multi-view data loader.
"""

import numpy as np
import torch

from threadpoolctl import threadpool_limits
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class MultiViewUnlabeledDataLoader(nnUNetDataLoader):
    """
    Drop-in replacement for nnUNetDataLoader, used only for the unlabeled
    (TrU) loader.

    Produces num_views independently-augmented intensity views of every
    sample from ONE shared geometric draw (crop/rotation/scaling/mirror):
    the geometric transform runs once per sample, then intensity_transforms
    runs independently num_views times on clones of that single geometric
    result.

    Calling the geometric+intensity pipeline independently per view would
    NOT give aligned views -- crop position and SpatialTransform's
    rotation/scaling/mirror draw are resampled every call, so the outputs
    would be different patches, not different views of the same patch.
    Running the geometric transform once and branching afterward keeps all
    views voxel-aligned, differing only in intensity.

    weak_strong=True (requires num_views >= 2) forces view 0 to always be
    the weak view -- geometric-only, intensity_transforms skipped entirely
    -- while views 1..num_views-1 ("strong") independently draw from
    intensity_transforms. This is the FixMatch split: index 0 is meant to
    generate a pseudo-label/confidence mask, the rest are meant to be
    trained against it. weak_strong=False (default) keeps the legacy
    behavior where every view, including index 0, draws from
    intensity_transforms.
    """

    def __init__(self, *args, geometric_transforms, intensity_transforms, num_views, weak_strong=False, **kwargs):
        super().__init__(*args, transforms=None, **kwargs)

        self.geometric_transforms = geometric_transforms
        self.intensity_transforms = intensity_transforms
        self.num_views = int(num_views)
        self.weak_strong = bool(weak_strong)

        if self.num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {self.num_views}")

        if self.weak_strong and self.num_views < 2:
            raise ValueError(f"weak_strong requires num_views >= 2, got {self.num_views}")

    def generate_train_batch(self):
        selected_keys = self.get_indices()

        data_views_all = [None] * self.num_views
        seg_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)

                    data, seg, seg_prev, properties = self._data.load_case(i)
                    shape = data.shape[1:]
                    bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])
                    bbox = [[lb, ub] for lb, ub in zip(bbox_lbs, bbox_ubs)]
                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    weak = self.geometric_transforms(image=data_cropped, segmentation=seg_cropped)

                    seg_sample = weak["segmentation"]

                    for v in range(self.num_views):
                        if self.weak_strong and v == 0:
                            view_sample = weak["image"].clone()
                        else:
                            view = self.intensity_transforms(image=weak["image"].clone(), segmentation=seg_sample)
                            view_sample = view["image"]
                        if data_views_all[v] is None:
                            data_views_all[v] = torch.empty((self.batch_size, *view_sample.shape), dtype=torch.float32)
                        data_views_all[v][j] = view_sample

                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype) for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample

        return {
            "data_views": data_views_all,
            "target": seg_all,
            "keys": selected_keys,
        }
