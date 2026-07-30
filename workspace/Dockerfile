# MVAA final-test submission image. Runs run_inference.py for all 3 tasks
# under the organizer's contract (no network, MVAA_INPUT_DIR/MVAA_OUTPUT_DIR
# via -e). Code lives at /workspace, not /work -- /work is their writable
# scratch mount (-v .../job/work:/work:rw, -w /work) and would overlay our
# code at container start otherwise.
#
# Build context is workspace/ itself: docker build -t mvaa-infer . from
# inside workspace/.
#
# Base pinned to CUDA 12.4 to match the final-test GPU (V100/sm_70, driver
# 550.x) -- same base the organizers' own reference image uses.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0"

WORKDIR /workspace

COPY requirements.txt requirements.txt
# --trusted-host: some build hosts intercept all outbound HTTPS, not just
# Docker Hub -- including PyPI (see docker.sh). This container has no idea
# about the host's trust-store fix (docker.sh's trust_remote_dockerhub_proxy_cert
# only touches the host, not this build stage), and pip verifies against its
# own vendored certifi bundle anyway, so --trusted-host is the right scoped
# fix here. Harmless on a host with no interception.
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    -r requirements.txt

COPY . .


# Absolute path -- the organizer's -w /work at runtime overrides this
# image's WORKDIR, so a relative path would look for run_inference.py in
# empty /work instead. run_inference.py is already cwd-independent
# (BASE_DIR = Path(__file__).resolve().parent); this just needs to match.
ENTRYPOINT ["python", "/workspace/run_inference.py"]
