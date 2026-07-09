"""
DDPMixin for SSL nnU-Net.

Rank-local output folder resolution, DDP barriers, and merging per-rank
outputs back into a single fold_all output tree.

All methods are prefixed with "ddp" so call sites (self._ddp_...) make
clear which mixin they come from.
"""

from utils import (
    get_rank_output_folder,
    merge_rank_folders,
    cleanup_rank_outputs,
)


class DDPMixin:

    def _ddp_rank_info(self):
        return (
            int(self.trainer.global_rank),
            int(self.trainer.world_size),
            bool(self.trainer.is_global_zero),
        )

    def _ddp_rank_output_folder(self):
        global_rank, _, _ = self._ddp_rank_info()

        return get_rank_output_folder(
            fold_output_folder=self.fold_output_folder,
            global_rank=global_rank,
        )

    def _ddp_barrier(self, name=None):
        if int(self.trainer.world_size) <= 1:
            return

        if name is None:
            self.trainer.strategy.barrier()
        else:
            self.trainer.strategy.barrier(name)

    def _ddp_merge_rank_outputs(self):
        """
        Simple DDP-safe merge.

        Order:
            1. all ranks finish writing
            2. barrier
            3. rank 0 merges rank folders
            4. barrier
            5. rank 0 deletes _rank_outputs

        We do not need done marker files here.
        """

        _, _, is_global_zero = self._ddp_rank_info()

        self._ddp_barrier(
            name="before_merge_rank_outputs",
        )

        if is_global_zero:
            merge_rank_folders(
                fold_output_folder=self.fold_output_folder,
                task_id=self.cfg.task_id,
                overwrite=True,
            )

        self._ddp_barrier(
            name="after_merge_rank_outputs",
        )

        if is_global_zero:
            cleanup_rank_outputs(
                fold_output_folder=self.fold_output_folder,
            )
