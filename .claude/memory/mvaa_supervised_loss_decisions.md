---
name: mvaa-supervised-loss-decisions
description: "Loss-function design decisions made on the MVAA `supervised` branch (losses.py)"
metadata: 
  node_type: memory
  type: project
  originSessionId: b303adc7-e9c9-4abb-9f35-e5cb82c2dee8
---

The MVAA `supervised` branch (`/cluster/work/projects/nn8104k/dsi014/projects/MVAA/supervised`, `losses.py`) implements 6 selectable `loss_type` options: `dice_ce`, `dice_focal`, `tversky_ce`, `focal_tversky`, `dice_topk`, `unified_focal` (see `litmodule.loss_type` in the experiment configs).

**Temperature scaling on logits before the compound loss (T<1 sharpening) was considered and explicitly rejected (2026-07-08)** — not implemented.

**Why**: doesn't change the final predicted class (argmax is temperature-invariant), only training dynamics. For CE/Focal it's a rough, differently-mechanized analog to what Focal's `(1-p)^gamma` term already does explicitly. For Dice/Tversky's soft overlap, sharpening forces early-training voxels (many near 50/50) toward a hard 0/1 mask before the network has earned that confidence, risking noisier/unstable gradients — especially compounding with Tversky's already-asymmetric FP/FN weighting. User judged it not worth the added complexity/instability risk for this project.

**How to apply**: don't re-propose plain training-time logit temperature scaling for this codebase's compound losses unless something material changes (e.g. a specific instability problem it might address). Post-hoc calibration (Guo et al. 2017 style, fit T after training on a frozen model) was not the thing rejected — that's a different technique and wasn't discussed in depth.
