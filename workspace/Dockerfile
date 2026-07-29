# MVAA final-test submission image. Runs run_inference.py for all 3 tasks
# (ct/tee/video) under the organizer's contract: no network, reads
# MVAA_INPUT_DIR, writes MVAA_OUTPUT_DIR (both always passed via -e per
# their contract; config/*.yaml also defaults each to /input, /output for
# local testing without Docker). Code lives at /workspace -- same
# convention as the organizer's own reference Dockerfile -- deliberately
# NOT /work: that path is their writable scratch mount (-v .../job/work:
# /work:rw, -w /work), which would overlay and hide our code at container
# start if we put it there.
#
# Build context is this workspace/ directory itself: docker build -t
# mvaa-infer . from inside workspace/.
#
# Base pinned to CUDA 12.4 to match the final-test GPU (V100/sm_70, driver
# 550.x) -- same base the organizers' own reference image uses. torch/
# torchvision come from this image and are left alone by requirements.txt.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0"

WORKDIR /workspace

COPY requirements.txt requirements.txt
# --trusted-host: some build hosts (vast.ai instances behind their own
# TLS-intercepting Docker Hub cache, see docker.sh) intercept *all*
# outbound HTTPS on the network, not just Docker Hub -- including PyPI.
# The RUN below executes inside this fresh container, which has no idea
# about whatever custom cert the host machine was made to trust (see
# docker.sh/submission.py's trust_remote_dockerhub_proxy_cert -- that only
# updates the host's own system trust store, not this build stage's).
# pip also mostly ignores the system trust store anyway (it verifies
# against its own vendored certifi bundle by default), so a host-level
# cert fix wouldn't reach it even if COPYed in. --trusted-host is pip's
# own standard, scoped mechanism for exactly this "known TLS-intercepting
# proxy" situation -- skips verification only for these specific hosts,
# not globally. Harmless on a host with no interception: these hosts just
# verify normally there too, --trusted-host only matters when verification
# would otherwise fail.
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    -r requirements.txt

COPY . .


# Absolute path, not relative -- the organizer's contract always passes
# -w /work at runtime, which overrides this image's own WORKDIR. A
# relative "run_inference.py" would then be looked up in /work (empty,
# not where the code lives) and fail. run_inference.py itself is already
# cwd-independent (BASE_DIR = Path(__file__).resolve().parent); this just
# needs to match.
ENTRYPOINT ["python", "/workspace/run_inference.py"]
