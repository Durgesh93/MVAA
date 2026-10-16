"""Print a Hydra-safe tag slug for the sweep this agent belongs to."""
import os, re, sys

sweep_id = os.environ.get("WANDB_SWEEP_ID", "").strip()

if not sweep_id:
    sys.exit(0)                      # not a sweep run: leave the config tag alone

name = sweep_id                      # fallback: the id is always better than nothing

try:
    import yaml, wandb

    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")

    if not (entity and project):
        cfg = yaml.safe_load(open("config/experiment_TEE.yaml"))["wandb"]
        entity, project = entity or cfg["entity"], project or cfg["project"]

    name = wandb.Api(timeout=20).sweep(f"{entity}/{project}/{sweep_id}").name or sweep_id
except Exception:
    pass                             # offline, revoked key, deleted sweep -- use the id

# Hydra splits on commas/brackets and W&B caps tag length, so reduce to a slug.
slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")[:64]
print(slug or sweep_id)
