#!/usr/bin/env bash
set -euo pipefail

# Build the MVAA submission image and test it against workspace/input/ (see
# ../submission.py's setup_action for how that gets populated -- excluded
# from the image itself via .dockerignore) and the sibling output/, work/
# folders, using the organizer's exact docker run contract (section 4:
# network none, resource limits, -v .../input:ro etc., -e MVAA_*_DIR,
# -w /work) so a passing run here is a real signal for the actual
# submission, not just "it builds."
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

# Some vast.ai hosts run a transparent TLS-intercepting proxy on Docker
# Hub's hostnames (presumably a bandwidth-saving pull-through cache --
# confirmed via `getent hosts registry-1.docker.io` returning a private
# 192.168.x.x address even when queried against 8.8.8.8, i.e. the redirect
# is at the host's network level, not something in this VM's own config).
# It terminates TLS with its own cert instead of Docker Hub's real one.
# `insecure-registries` does NOT work around this -- confirmed by direct
# testing, Docker still enforces strict TLS for the official docker.io
# registry regardless of that setting. The proxy also never sends its own
# root CA in the handshake (only its per-host leaf cert), so instead of
# trying to trust a root we don't have, just capture whatever leaf cert
# each hostname currently presents and add it directly as a trusted
# anchor -- an exact-match trust entry doesn't need to be self-signed. If
# a host ISN'T doing this interception, this just captures the real Docker
# Hub cert instead, which is already trusted anyway -- harmless either way.
for h in registry-1.docker.io auth.docker.io index.docker.io; do
  echo | openssl s_client -connect "$h:443" -servername "$h" 2>/dev/null \
    | openssl x509 | sudo tee "/usr/local/share/ca-certificates/vast-proxy-$h.crt" >/dev/null
done
sudo update-ca-certificates

sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker

# One restart for both config changes above -- not one each. systemd rate-
# limits how many times a unit can restart within a short window
# (StartLimitBurst), and restarting docker.service back-to-back (once here,
# once already in submission.py's docker_login_remote before this script
# even ran) was enough to trip it ("start-limit-hit"), which then made this
# restart fail even though the config itself was fine. reset-failed clears
# that rate-limit state defensively before we ask for the one restart we
# actually need.
sudo systemctl reset-failed docker.service 2>/dev/null || true
sudo systemctl restart docker

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
# Docker Hub, username durgesh1993 -- this is also what gets pushed for
# the actual CodaBench submission.json (see ../submission.py), not just a
# local test tag. Keep this in sync with submission.py's IMAGE constant.
IMAGE_TAG="docker.io/durgesh1993/team_vi:final"

# Only drop our own tag + dangling leftovers from previous builds of it --
# NOT `docker system prune -af`, which would also evict the base image's
# cached layers and make every build re-pull/re-run from scratch. Scoped
# like this, unchanged layers (base image, unchanged COPY steps) stay
# cached, so a second build after a small code change is fast.
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
