"""
DDPHelper for SSL nnU-Net.

Rank-local output folder resolution, DDP barriers, and merging per-rank
outputs back into a single experiment output tree.

Standalone component: holds no state of its own. `trainer` isn't known
until Lightning attaches it (unavailable at LightningModule.__init__
time), so every method takes `trainer` -- along with whatever
output-folder context it needs -- as an explicit argument rather
than storing it.
"""

from utils import get_rank_output_folder, merge_rank_folders, cleanup_rank_outputs


class DDPHelper:
    def _rank_info(self, trainer):
        return (int(trainer.global_rank), int(trainer.world_size), bool(trainer.is_global_zero))

    def rank_output_folder(self, trainer, output_folder):
        global_rank, _, _ = self._rank_info(trainer)

        return get_rank_output_folder(output_folder=output_folder, global_rank=global_rank)

    def barrier(self, trainer, name=None):
        if int(trainer.world_size) <= 1:
            return

        if name is None:
            trainer.strategy.barrier()
        else:
            trainer.strategy.barrier(name)

    def merge_rank_outputs(self, trainer, output_folder):
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

        _, _, is_global_zero = self._rank_info(trainer)

        self.barrier(trainer, name="before_merge_rank_outputs")

        if is_global_zero:
            merge_rank_folders(output_folder=output_folder, overwrite=True)

        self.barrier(trainer, name="after_merge_rank_outputs")

        if is_global_zero:
            cleanup_rank_outputs(output_folder=output_folder)
