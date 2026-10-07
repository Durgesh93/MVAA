#!/bin/bash
# Entry point for `wandb agent`, which cannot call engine.py directly.
#
# Three things have to happen before training starts:
#
#  1. `py` is a shell FUNCTION from the LUMI workspace env, not an
#     executable, so the agent's child shell has to source main.sh to get it
#     (and with it the venv, ROCm config, the user-site overlay and
#     PYTHONPATH).
#
#  2. Every run needs its OWN experiment_name. engine.py's clear_results()
#     deletes ${paths.nnunet_results}/${experiment_name} at startup, so two
#     agents sharing a name would delete each other's results mid-run.
#     WANDB_RUN_ID is unique per run and is exported by the agent
#     (wandb_agent.py sets wandb.env.RUN_ID before launching this command),
#     which also makes the results folder on disk match the W&B run. The
#     swept num_labeled is prefixed onto it so the folders are navigable
#     without cross-referencing W&B.
#
#  3. The agent overwrites WANDB_DIR with its own cwd (wandb_agent.py:255),
#     which would scatter wandb/ directories through the checkout. Sourcing
#     main.sh AFTER that restores the scratch path.
#
# Sweep parameters arrive as Hydra-style key=value pairs via ${args_no_hyphens}
# and are forwarded untouched.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Sourced with `|| true` and output discarded: this is defensive, not a
# known failure. main.sh used to exit non-zero here on a stray
# `set_ps: command not found`, which under `set -e` aborted the run
# silently; that bug is fixed, but a login script is not something a
# training run should die on. Strict mode goes on afterwards.
# shellcheck disable=SC1090
source "${ENV_STORAGE_BASE:-/project/project_465003462/durgeshk/envs/workspace}/platforms/lumi/main.sh" >/dev/null 2>&1 || true

# Only pipefail: `set -u` breaks the workspace's own `py` function, which
# reads unset variables (helper.sh: REPO_DIR), and `set -e` is unusable
# for the same reason main.sh is sourced defensively above.
set -o pipefail

if ! command -v py >/dev/null 2>&1 && ! declare -F py >/dev/null 2>&1; then
  echo "ERROR: 'py' is not available after sourcing the workspace env." >&2
  exit 1
fi

if ! python -c 'import nnunetv2' >/dev/null 2>&1; then
  echo "ERROR: nnunetv2 is not importable. Install it into the user-site" >&2
  echo "       overlay with: pip install nnunetv2" >&2
  exit 1
fi

RUN_TAG="${WANDB_RUN_ID:-manual_$(date +%Y%m%d_%H%M%S)_$$}"

# Readable prefix from the swept parameter, so nnUNet_results/ sorts by
# supervision level instead of by opaque run id.
NUM_LABELED=""
for arg in "$@"; do
  case "$arg" in
    datamodule.num_labeled=*) NUM_LABELED="${arg#*=}" ;;
  esac
done

if [ -n "$NUM_LABELED" ]; then
  EXP_NAME="$(printf 'fewshot_n%03d_%s' "$NUM_LABELED" "$RUN_TAG")"
else
  EXP_NAME="fewshot_${RUN_TAG}"
fi

echo "============================================================"
echo "sweep run"
echo "  repo            : $REPO_ROOT"
echo "  experiment_name : $EXP_NAME"
echo "  num_labeled     : ${NUM_LABELED:-<not swept>}"
echo "  WANDB_DIR       : ${WANDB_DIR:-<unset>}"
echo "  overrides       : $*"
echo "============================================================"

# Not `exec`: `py` is a shell function, and exec can only replace the
# shell with an external executable ("exec: py: not found").
py engine.py train "experiment_name=$EXP_NAME" "$@"
exit $?
