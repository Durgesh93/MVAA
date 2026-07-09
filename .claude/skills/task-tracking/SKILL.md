---
name: task-tracking
description: Use when the user says "task video", "task ct", "task tee", or lists several together (e.g. "task video, ct, tee") to check MVAA nnU-Net training progress. Finds every experiment that has trained the named task(s), reads each one's training_progress.png, and reports a comparison table.
---

# MVAA task tracking

Triggers on phrases like "task video", "task ct", "task tee", or a
comma/space separated list of these. Each name maps to a dataset:

| task name | dataset_id                |
|-----------|----------------------------|
| video     | Dataset003_MVAA_VIDEO_SSL |
| ct        | Dataset001_MVAA_CT_SSL    |
| tee       | Dataset002_MVAA_TEE_SSL   |

## Steps

The directory layout is experiment name first, dataset name inside
that experiment folder — always walk it in that order, don't build a
combined path from a fixed task->dataset_id table:

`nnUNet_results/{experiment_name}/{dataset_id}/nnUNetPlans__2d/training_progress.png`

(no `fold_XXX` segment -- `_save_training_progress_plot` in
metrics_module.py writes directly to `actual_validation_output_base`,
not to `fold_output_folder`; `module.py`'s `progress_png_file`
attribute, which does include a `fold_XXX` segment, is unused dead
code. Verify against metrics_module.py's `_save_training_progress_plot`
if this ever seems to mismatch reality again -- don't trust
module.py's `progress_png_file` attribute name alone.)

1. For the requested task(s), resolve the dataset_id(s) from the
   table above (this only tells you which dataset folder *name* to
   look for in step 3, not the full path).

2. List every experiment folder (one level) under:
   `/cluster/work/projects/nn8104k/dsi014/experiment_storage/data/nnUNet/nnUNet_results/`

   Don't hardcode a fixed list of experiment names — new experiment
   folders get added over time (e.g. new branches/worktrees for
   follow-up ablations). Discover them dynamically each time this
   skill runs.

3. Within each experiment folder from step 2, look for a subfolder
   matching the requested dataset_id. If it exists, check whether
   this file exists inside it:
   `{experiment_folder}/{dataset_id}/nnUNetPlans__2d/training_progress.png`

   Skip experiments that don't have this dataset_id subfolder at all
   (they haven't trained that task), and skip ones that have the
   subfolder but not yet the PNG (haven't completed a validation
   epoch yet) — note the latter case explicitly in the report rather
   than silently omitting it, since a job that's running but stuck at
   zero validation epochs is itself worth flagging.

4. For every matching file, first check its size/mtime, then read it
   with the Read tool (it's a PNG and renders as an image). It's a
   7-panel grid (Train loss, Supervised loss, Pseudo loss, Dice, ASD
   mm, HD mm, HD95 mm) — panels that don't apply to a given trainer
   (e.g. no Pseudo loss on the supervised branch) are simply absent,
   not zero. Each panel's legend shows `last=` and `best=` values, and
   the x-axis shows the current epoch count.

5. Build a single markdown table across all matching experiments,
   with columns:

   `Experiment | Epochs | Dice (last/best) | ASD mm (last/best) | HD mm (last/best) | HD95 mm (last/best)`

   If multiple tasks were requested (e.g. "task video, ct, tee"),
   produce one table per task, clearly labeled.

6. Present the table(s) directly in the response. Note anything
   worth flagging: an experiment that looks stalled (unchanged epoch
   count vs. a prior check, if one is known from conversation
   context), a run that's clearly still very early relative to others,
   or a notable leader on Dice/HD95.

Do not launch, cancel, or modify any jobs/configs as part of this
skill — it only reads and reports.
