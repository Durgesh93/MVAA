---
name: mvaa-phase1-video-loss-experiments
description: "What \"Phase 1\" refers to for the MVAA video task loss ablation, and how to re-check its progress"
metadata: 
  node_type: memory
  type: project
  originSessionId: b303adc7-e9c9-4abb-9f35-e5cb82c2dee8
---

**Trigger phrase**: when the user says "phase_1 track video" (or similar, e.g. "check phase 1 video"), re-read `training_progress.png` for each of the 6 branches below and rebuild the comparison table -- same as was done on 2026-07-08.

Phase 1 is a loss-function head-to-head comparison on the MVAA video task (`Dataset003_MVAA_VIDEO_SSL`), 6 branches/worktrees created from the `supervised` branch in `/cluster/work/projects/nn8104k/dsi014/projects/MVAA`, each isolating one `loss_type` (see `losses.py`) with everything else identical (100 epochs, default `oversample_fg=0.33`, default loss hyperparameters):

| Branch / experiment_name | loss_type |
|---|---|
| `sup_dice_ce` | dice_ce (baseline) |
| `sup_dice_focal` | dice_focal |
| `sup_tversky_ce` | tversky_ce |
| `sup_focal_tversky` | focal_tversky |
| `sup_dice_topk` | dice_topk |
| `sup_unified_focal` | unified_focal |

**How to check**: use the [[task-tracking]] skill's mechanism (`~/.claude/skills/task-tracking/SKILL.md`) -- read
`/cluster/work/projects/nn8104k/dsi014/experiment_storage/data/nnUNet/nnUNet_results/{experiment_name}/Dataset003_MVAA_VIDEO_SSL/nnUNetPlans__2d/training_progress.png`
for each of the 6 branches above specifically (not a dynamic discovery pass over all experiments -- "phase_1 track video" means just these 6, whereas "task video" means the broader dynamic scan). Note: no `fold_XXX` segment in the path.

**Snapshot as of 2026-07-08 evening** (do not trust for long, all only 4-5 of 100 epochs in): `sup_dice_ce` led on Dice (0.350), followed by `sup_focal_tversky` (0.338), `sup_tversky_ce` (0.327), `sup_dice_focal` (0.282), `sup_dice_topk` (0.192), and `sup_unified_focal` clearly lagging (0.094, HD mm exactly flat across all 4 epochs -- worth checking if that's just an early plateau or genuinely stuck).

There's also a separate, unrelated experiment also training on the video dataset: `focal_ws_adaptive_warmup` (on branch `focal-pseudo-quality`, the weak/strong-augmentation + adaptive pseudo-label-threshold + warmup experiment, much further along at ~22 epochs, Dice 0.774). That one is NOT part of Phase 1 (different training paradigm, SSL not pure supervised) -- don't merge it into the Phase 1 table, but it's fair game for the general "task video" tracking skill.
