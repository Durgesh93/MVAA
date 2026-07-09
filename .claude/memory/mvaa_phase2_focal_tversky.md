---
name: mvaa-phase2-focal-tversky
description: Phase 1 loss-ablation conclusion and Phase 2 starting point for MVAA video task
metadata: 
  node_type: memory
  type: project
  originSessionId: 0ef411e3-93ac-4c1b-ab70-a4a9b6557795
---

**Phase 1 concluded (2026-07-09): `focal_tversky` (branch/experiment `sup_focal_tversky`) was chosen as the winning loss_type** for the MVAA video task (`Dataset003_MVAA_VIDEO_SSL`), out of the 6 candidates in [[mvaa_phase1_video_loss_experiments]]. At ~78 epochs it had Dice last/best 0.739/0.745, ASD 23.2/11.7mm, HD 166.2/73.0mm, HD95 85.3/39.6mm — competitive Dice with the best HD95 among fully-converged supervised runs (only `focal_ws_adaptive_warmup`, a separate SSL-paradigm experiment, scored higher on Dice/ASD).

**The other 5 Phase 1 branches/worktrees were deleted on 2026-07-09**: `sup_dice_ce`, `sup_dice_focal`, `sup_dice_topk`, `sup_tversky_ce`, `sup_unified_focal` — both the git branches and their worktree directories under `/cluster/work/projects/nn8104k/dsi014/projects/MVAA/`. Their nnUNet training result folders under `nnUNet_results/` were left in place (not part of that cleanup request).

**Why**: user's explicit call after comparing Phase 1 results via the [[task-tracking]] skill — focal_tversky offered the best balance, not the absolute best single metric.

**How to apply**: Phase 2 work builds on the `sup_focal_tversky` worktree/branch going forward. Don't reference the deleted branches as if they still exist — if asked to re-check "phase_1 track video", note that 5 of the 6 branches no longer exist (only `sup_focal_tversky` remains, plus the unrelated `focal-pseudo-quality` SSL branch and the base `supervised`/`main` branches).
