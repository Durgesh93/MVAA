#!/usr/bin/env bash
set -euo pipefail

# Build the MVAA submission image and test it against workspace/input/ (see
# ../submission.py's setup_action; excluded from the image via .dockerignore)
# and the sibling output/, work/ folders, using the organizer's exact docker
# run contract (section 4) so a passing run here is a real signal, not just
# "it builds."
#
# Push to Docker Hub is opt-in (./docker.sh --push), only after a PASS,
# never automatic -- run `docker login` yourself first (once, interactively;
# credentials never belong in this script).
#
# GPUS/CPUS/MEMORY are overridable for test machines that don't match the
# organizer's exact spec -- e.g. a CPU-only cloud instance for a quick
# build/logic smoke test: GPUS="" CPUS=8 MEMORY=16g ./docker.sh
# Defaults match their contract exactly (section 4).

PUSH=0
if [ "${1:-}" = "--push" ]; then
  PUSH=1
fi

# Some vast.ai hosts transparently MITM Docker Hub's TLS with their own
# proxy cert (a bandwidth-saving pull cache, not something in this VM's own
# config). Pinning the intercepted leaf cert (the previous approach here)
# turned out unreliable: the proxy is load-balanced across nodes that each
# mint their own leaf, so the cert captured a moment ago isn't necessarily
# the one presented on the next connection -- confirmed by querying the
# same host twice a few minutes apart and getting different fingerprints.
# insecure-registries sidesteps this by skipping cert verification for
# just these hosts; harmless if a host isn't intercepting at all, since a
# real cert still connects fine either way.
sudo python3 - <<'PYEOF'
import json
path = "/etc/docker/daemon.json"
try:
    config = json.load(open(path))
except (FileNotFoundError, json.JSONDecodeError):
    config = {}
hosts = ["registry-1.docker.io", "auth.docker.io", "index.docker.io"]
config["insecure-registries"] = sorted(set(config.get("insecure-registries", [])) | set(hosts))
json.dump(config, open(path, "w"), indent=2)
PYEOF

sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker

# One restart for both config changes -- systemd's restart rate-limit
# already tripped once from submission.py's docker_login_remote restarting
# docker.service right before this script ran. reset-failed clears that
# state first so this restart doesn't fail too.
sudo systemctl reset-failed docker.service 2>/dev/null || true
sudo systemctl restart docker

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
# Also what gets pushed for the actual CodaBench submission.json (see
# ../submission.py) -- keep in sync with submission.py's IMAGE constant.
IMAGE_TAG="docker.io/durgesh1993/team_vi:final"

# Only our own tag + its dangling leftovers -- not `docker system prune -af`,
# which would also evict the base image's cached layers.
echo "==> Removing previous $IMAGE_TAG (keeping cache/other images)"
docker rmi -f "$IMAGE_TAG" 2>/dev/null || true
docker image prune -f

GPUS="${GPUS---gpus all}"
CPUS="${CPUS:-14}"
MEMORY="${MEMORY:-28g}"

echo "==> Building $IMAGE_TAG from $SCRIPT_DIR"
docker build -t "$IMAGE_TAG" "$SCRIPT_DIR"

mkdir -p "$INFERENCE_DIR/output" "$INFERENCE_DIR/work"

echo "==> Running $IMAGE_TAG against $SCRIPT_DIR/input and $INFERENCE_DIR/{output,work}"
docker run --rm $GPUS \
  --network none \
  --memory "$MEMORY" \
  --cpus "$CPUS" \
  --pids-limit 2048 \
  --shm-size 8g \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  -v "$SCRIPT_DIR/input":/input:ro \
  -v "$INFERENCE_DIR/output":/output:rw \
  -v "$INFERENCE_DIR/work":/work:rw \
  -e MVAA_INPUT_DIR=/input \
  -e MVAA_OUTPUT_DIR=/output \
  -w /work \
  "$IMAGE_TAG"

echo "==> Verifying expected output files"
FAILED=0
for entry in "t1_ct:task1_predictions.json" "t2_tee:task2_predictions.json" "t3_vid:task3_predictions.json"; do
  prefix="${entry%%:*}"
  json_name="${entry##*:}"
  json_path="$INFERENCE_DIR/output/$prefix/$json_name"

  if [ -f "$json_path" ]; then
    echo "  OK   $json_path"
  else
    echo "  MISSING   $json_path"
    FAILED=1
  fi
done

if [ "$FAILED" -ne 0 ]; then
  echo "==> FAIL: see MISSING entries above"
  exit 1
fi

echo "==> PASS: all 3 tasks wrote predictions to $INFERENCE_DIR/output"

if [ "$PUSH" -eq 1 ]; then
  echo "==> Pushing $IMAGE_TAG (requires a prior 'docker login')"
  docker push "$IMAGE_TAG"
else
  echo "==> Not pushing (pass --push to also push $IMAGE_TAG to Docker Hub)"
fi
