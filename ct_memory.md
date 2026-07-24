# CT (task1) training improvement — investigation notes

## Current state (SSL branch, `ssl/Dataset001_MVAA_CT_SSL`, ~95 epochs logged)

- Dice plateaus hard by epoch ~15-20 at ~0.83-0.84 and just noise-oscillates
  for the rest of training. `best=0.8463`, `last=0.8401` — no real gain past
  the early plateau.
- HD mm trends *worse* in later epochs (spikes to 9-11mm past epoch ~70 vs
  5-6mm around epoch 20-45), consistent with train loss also ticking back up
  slightly after its epoch-~45 minimum (best -0.8903 vs last -0.8713) — looks
  like mild late-training overfitting/instability on the boundary-sensitive
  metric specifically, even though bulk Dice stays flat.
- MVAA platform predict-all run (see earlier session): CT dice 0.8455-0.8463,
  well ahead of video's 73% at the time, but still plateaued internally.

## Data facts (measured directly, not assumed)

- **CORRECTED 2026-07-23**: `TrL=27` labeled cases (not 11 as originally
  written here — that number was stale/wrong, apparently from a different
  split's count, not the full labeled set). Verified directly two ways: (1)
  `dataset.json`'s `ssl_counts.TrL` = 27 and `ssl_case_ids.TrL` lists 27 ids
  (`0000`-`0026`); (2) loaded all 27 of those label files directly and
  confirmed every one has nonzero foreground voxels (mean ~1.9%, range
  1.19%-2.76% — consistent with the fg-fraction figure below, just the case
  *count* was wrong). `TrU=1070` (from `dataset.json`, also corrected from
  the earlier `535` figure — same stale-source issue). Still meaningfully
  scarcer than TEE (`TrL=50`, `TrU=10` — almost fully supervised) and video
  (`TrL=56`, `TrU=714+`), but not as dire as the original 11/535 made it
  look — weakens (doesn't eliminate) the "data scarcity is the ceiling"
  explanation for the plateau; the per-crop small-sample-variance argument
  for why balanced CE adds noise is unaffected (that's about voxels per crop,
  not cases in the dataset).
- class_1 (mitral valve) foreground fraction measured on all 27 real labeled
  cases: mean ~1.9% of voxels, range 1.19%-2.76%. (Note: most of the 1097
  files under `labelsTr/` are empty placeholders for TrU/val/test cases —
  only the 27 true TrL case ids have real annotations; don't naively average
  fg-fraction over the whole `labelsTr/` folder, it'll read ~0% and mislead.)
- `patch_size = [112, 128, 160]` (3d_fullres plans) vs. `shapes_after_crop`
  for the 27 TrL cases: **[UNVERIFIED after TrL correction — the "6 of 11"
  count below predates the TrL=27 fix and needs rechecking against all 27
  cases, not just the original 11]** patch is as
  big as or bigger than the whole cropped case — zero crop-position freedom,
  padding fills the rest). The other 5 have patch at 73-88% of case volume —
  some real slack, but bounded (even a perfect foreground-centered crop in
  the tightest case could only concentrate ~2% true foreground up to
  roughly ~2.7%, not a dramatic rebalancing).
- Net effect: essentially every training crop (labeled or unlabeled) is
  ~98% background, ~2% the class that actually matters — but see the
  correction below on *why*.

  **Correction (same session, after the above was first written):** initially
  attributed this ~98%/2% split partly to `oversample_foreground_percent=0.0`
  for TrU (`data_module.py:438`, unlabeled crops have no ground truth to bias
  sampling toward foreground with). That's true as a factual statement but
  was the wrong causal story — given patch_vol >= case_vol for most cases
  (see above), `oversample_foreground_percent` has **no room to do anything**
  regardless of its value: the whole case gets captured either way. Tried
  bumping TrL's `oversample_fg` to 0.70 as a fix on this premise, confirmed
  via the volume-ratio math above that it would have negligible effect for
  at least half the cases, and reverted back to `0.33` (all 3 tasks, config
  is now per-task in the yaml instead of hardcoded, but numerically a no-op
  vs. before). **The real cause of the ~98%/2% split is simpler: the valve
  genuinely is only ~2% of the volume, and the patch captures ~all of the
  volume regardless of positioning — anatomy + patch-size choice, not the
  sampling flag.** This does NOT change the underlying pseudo-loss/
  confident_frac dilution hypothesis below (crops really are background-
  dominated either way) — only the explanation for *why*, and it rules out
  oversampling as a viable fix for it.

## What's ruled out

- Augmentation pool is **not** the lever — `_intensity_recipe_ct()`'s own
  docstring confirms it's "nnU-Net's default intensity op pool, unchanged."
  Network/optimizer/deep-supervision are also stock nnU-Net via
  `NNUnetSetup`. The only non-default pieces are the boundary loss add-on
  and the whole SSL/pseudo-labeling apparatus.

## Live hypothesis: pseudo-label confidence gate may be background-dominated

- `train_pseudo_confident_frac` (aggregate) jumps to ~98-100% almost
  immediately after warmup (epoch ~5) and stays pinned there for the rest of
  training. Given ~98% of every crop's voxels are trivially-easy background,
  this aggregate metric is plausibly dominated by background confidence and
  says almost nothing about whether class_1's pseudo-labels are trustworthy.
- `WeakStrongPseudoLabelLoss` (losses.py) averages the masked CE per-*voxel*
  over `num_confident`, with no class balancing — contrast with the
  *supervised* `CompoundLoss`'s Dice term, which uses `do_bg=False` and
  averages per-*class*, immune to background swamping the count. If the
  pseudo loss's denominator really is ~98% background voxels, genuine
  foreground disagreement between weak/strong views could be diluted into
  near-invisibility in both the loss gradient and the confident_frac metric.
- This would explain the early plateau: 535 unlabeled cases are being
  consumed, but the consistency signal for the class that matters may be
  structurally drowned out rather than actually informative.

## What we changed to test this (2026-07-23 session)

Added per-class (not just aggregate) pseudo confident-fraction tracking,
covering **every** class — background and auxiliary classes included, not
just `tracked_labels` — so the plot can show whether class_1 behaves
differently from background instead of being hidden inside one aggregate
number:

- `losses.py`: `WeakStrongPseudoLabelLoss.forward` now also returns
  `per_class_confident_frac` (tensor indexed by class id, NaN where a class
  is absent from that batch — safe, torchmetrics' `MeanMetric` already
  silently drops NaN on `.update()`, verified directly).
- `lightning_module.py`: added `self.all_labels` (every class from
  `dataset_json["labels"]`, sorted by id), threaded the per-class tensor
  through `_pseudo_loss` → `training_step` → `MetricsTracker`.
- `metrics.py`: `MetricsTracker` now takes `all_labels` and creates one
  `train_pseudo_confident_frac_<name>` tracked key per class.
- `utils.py`: `save_training_progress_plot` gained
  `pseudo_confident_frac_classwise_keys`, mirroring the existing
  `dice_classwise_keys` mechanism — the "Pseudo confident pixel frac" panel
  now overlays one line per class (e.g. `background`, `class_1` for CT) in
  the *same* subplot, instead of a single aggregate line.

**Status: implemented, verified, uncommitted.** `git status` shows
`module/lightning_module.py`, `module/losses.py`, `module/metrics.py`,
`module/nnunet.py`, `utils.py` modified.

## Fix implemented and CT run launched (same session, follow-up)

User initially said "train first as-is, discuss the loss fix later" (see
git history of this file for the tabled-discussion version), then reversed
course before launching and asked for the per-class-balanced CE fix
immediately, in **both** loss branches, plus said "do this quickly so i can
start the training." Implemented and verified all of this before the run
started -- nothing here is still just proposed:

1. **Crash fix (unconditionally required, unrelated to the loss question):**
   `MetricsTracker.epoch_metrics`'s `CatMetric`s now use
   `nan_strategy="disable"` instead of the default `"warn"`. Root cause: with
   `pseudo_warmup_epochs=5`, epoch 0's `train_pseudo_confident_frac_<name>`
   is NaN for every step of the whole epoch (all classes, not just rare
   ones). `"warn"` *drops* NaN entries from CatMetric's internal list
   instead of keeping a placeholder -- first-ever call to `.compute()` on a
   never-successfully-updated CatMetric returns a bare empty Python `list`,
   not a tensor, which crashed `compute_epoch_history`'s `.detach()` call
   with `AttributeError: 'list' object has no attribute 'detach'` on the
   very first validation epoch. `"disable"` keeps every epoch's entry
   (NaN included), so this key's array stays the same length as `epoch`'s,
   and the plotting code's existing `~np.isnan(values)` masking handles the
   gaps correctly. Verified directly end-to-end with a script simulating
   the exact warmup-then-real-data sequence through `MetricsTracker`.

2. **Per-class-balanced CE, both branches (the actual fix, not just the
   diagnostic from before):**
   - `losses.py`: new `_class_balanced_ce(net_output, target, loss_mask)` --
     averages CE per class first (mean over that class's own voxels), then
     uniformly across classes present. Background is *included* as one
     class among equals (unlike Dice's `do_bg=False`, which excludes it
     entirely) -- keeps background's calibration signal without letting its
     voxel count dominate.
   - `CompoundLoss` (supervised): no longer wraps nnU-Net's own
     `DC_and_CE_loss` (which used plain per-voxel `RobustCrossEntropyLoss`).
     Now owns `MemoryEfficientSoftDiceLoss` (`do_bg=False`, unchanged)
     directly + `_class_balanced_ce` in place of the CE term. `DC_and_CE_loss`
     import removed from losses.py entirely.
   - `WeakStrongPseudoLabelLoss` (pseudo branch): the masked CE between
     weak pseudo-label and each strong view is now averaged the same way --
     per class within `confident_mask`, then uniformly across classes,
     instead of one flat average over every confident voxel regardless of
     class.
   - Both verified directly: gradients flow correctly, an extreme-imbalance
     dummy case (single foreground voxel in an otherwise-all-background
     volume) doesn't crash or NaN.

**User started the CT training run with all of this in place.** Deferred
discussion (whether it actually helps, watching for the variance tradeoff
on rare-class small-sample-count noise) to next time.

## Next step (when we look at this again / user says "ct training")

1. Check whether the run is progressing / has produced a
   `training_progress.png` yet for `ssl/Dataset001_MVAA_CT_SSL`.
2. Compare Dice/HD trajectory against the pre-fix baseline documented at the
   top of this file (plateau ~0.83-0.84 by epoch ~15-20, HD degrading after
   epoch ~70) -- does the per-class-balanced CE change that trajectory?
3. Look at the "Pseudo confident pixel frac" panel's per-class lines
   (`background` vs `class_1`) -- does `class_1` behave differently from
   `background` now that both the diagnostic *and* the fix are in place?
4. Watch for the variance tradeoff flagged when this was proposed: with only
   ~2% foreground and small per-crop confident-pixel counts for class_1, is
   the per-class-averaged loss/CE noisy step-to-step in a way that's visible
   in the training curves?

## 2026-07-25 session: run interruption + resume

The run (in the `ssl_pretrain` worktree, `MVAA.ssl_pretrain` id 10, command
`engine.py retrain ct`, resuming a checkpoint originally from `ssl_pretrain`
stage-2 chained training) got interrupted at some point after epoch 45.
Several `MVAA_ssl_pretrain` SLURM jobs over 2026-07-24 failed from an
unrelated, pre-existing bug: `RuntimeError: Cannot re-initialize CUDA in
forked subprocess` -- batchgenerators' `NonDetMultiThreadedAugmenter` forks
background augmentation workers (default Linux multiprocessing start method
is `fork`) after CUDA is already initialized in the main process. Hit at
epoch 0 in two quick failures (jobs 1631272, 1632303) and mid-training in a
third (job 1631283, 6.8h in) where a dead worker on one DDP rank caused the
other rank to hang on an NCCL collective for 30 min before the watchdog
aborted the whole 2-GPU job. **Not yet fixed** -- still a live risk for
future long CT runs (also affects the `train`/`pretrain` commands, not just
`retrain`, since it's in the shared augmenter path).

Separately, today's actual retrain attempt (job 1633749) failed instantly
(26s) with a *different*, one-off error: `device=1, num_gpus=1` internal
assert during `torch.cuda` init -- SLURM granted 2 GPUs on node `gpu-1-111`
but CUDA only enumerated 1 at process start. Confirmed via `sinfo`/`scontrol`
that the node itself was healthy (not drained, 4x H200, other jobs running
fine on it) -- this was transient allocation flakiness, not a code or node
problem. Fix was simply resubmitting: job **1634133** (node `gpu-1-79`)
launched successfully, resumed cleanly from the epoch-45 checkpoint, both
GPUs visible (`CUDA_VISIBLE_DEVICES:[0,1]`), training progressing normally.

Also verified (in case future sessions get confused by the log's printed
`TrL=11`): this is **not** a regression of the TrL=27 correction from
2026-07-23. `data_module.py`'s `fold == "all"` branch holds out
`min(5, len(trl))` = 5 of the 27 TrL cases for validation, leaving 22 for
training, then `split_by_rank` shards those across the 2 DDP ranks -> 11
each. `TrU`: 1070 splits evenly to 535/rank. Both match the log exactly --
dataset.json's TrL=27/TrU=1070 totals are still correct, the per-rank
training-loop numbers are just smaller because of the val holdout + DDP
sharding, not a data-loss bug.

**Next step, in addition to the list above**: watch whether job 1634133
completes cleanly or hits the forked-CUDA-subprocess crash again (it's not
fixed, just avoided so far by luck of not hitting a worker restart). If it
recurs, the actual fix is almost certainly forcing the multiprocessing start
method to `spawn` for the augmenter workers, or restructuring so CUDA isn't
touched in the main process before the augmenters spawn -- not yet
investigated in code.
