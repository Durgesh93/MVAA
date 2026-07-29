"""
Typer-based menu tool for building the MVAA CodaBench submission.zip / docker
build archive and shipping it to config.json's machine -- either a fixed
machine, or (if config.json has vast_api_key set) a rented vast.ai instance.

Run with no arguments for an interactive numbered menu, or invoke each step
directly as its own subcommand:

    python3 submission.py                    # interactive menu
    python3 submission.py create-vast         # 1. rent/reuse a vast.ai instance
    python3 submission.py delete-vast         # 2. destroy the instance on record
    python3 submission.py push-docker         # 3. build zips + rsync the archive to the remote
    python3 submission.py trigger-build       # 4. run workspace/docker.sh on the remote
    python3 submission.py trigger-build --push  #    ...and also push the image to Docker Hub
    python3 submission.py pull-output         # 5. pull the remote output/ back as a local zip

The 5 steps are independent -- run them in order for a full build/test/push
cycle, or re-run just one (e.g. trigger-build again after fixing a bug,
without re-pushing the archive). Connection details for whichever vast.ai
instance is "current" are remembered in .submission_state.json between runs
(gitignored, like config.json).

Two zip artifacts, both written under zip/ by push-docker:
    zip/submission.zip         -- what actually gets uploaded to CodaBench
                                   (submission.json pointing at IMAGE below).
                                   IMAGE must match workspace/docker.sh's
                                   IMAGE_TAG exactly -- whatever gets pushed
                                   there is what CodaBench will actually try
                                   to pull.
    zip/docker_<timestamp>.zip -- self-extracting archive of input/+workspace/
                                   for building/testing the image elsewhere.

A copy of submission.zip is also written to workspace/submission.zip *before*
the docker archive is built, so it travels inside the archive and lands on
the remote machine too.

The output is a normal zip archive with a shell script prepended. A zip
reader locates the central directory by scanning backward from EOF, so
prepended bytes don't break it -- `unzip docker_<timestamp>.zip` still
works exactly as before. But the file is also directly runnable: the
shebang lets you just execute it, and it extracts into the current
directory and hands off to workspace/docker.sh.

This node has no direct route to the outside world (only the site's HTTP
CONNECT proxy, when config.json's proxy_host is set), so rsync/ssh here
are tunneled through this same script re-invoked as their own
ProxyCommand (see --tunnel below). vast.ai instances have their own direct
internet access, but the proxy requirement is about where *this script*
runs, not the target, so the same proxy_host/proxy_port still apply.

config.json fields:
    host, port, user, key_file          -- required for a fixed machine (ignored if vast_api_key is set,
                                            since host/port/user come from the rented vast.ai instance instead)
    remote_dir                          -- optional, default "~/"
    proxy_host, proxy_port              -- optional, omit/null for a machine with direct internet access
    docker_username, docker_token       -- optional, enables automatic `docker login` on the remote
                                            machine before --push (a Docker Hub access token, not your password)
    docker_password                     -- optional, enables deleting the old Docker Hub tag before
                                            pushing (see delete_dockerhub_tag -- needs a real password,
                                            a token can't authorize this)
    vast_api_key                        -- optional, switches to renting a vast.ai instance via create-vast
    vast_query, vast_image, vast_disk_gb, vast_min_cuda -- optional vast.ai tuning, see find_vastai_offer
                                            /create_vastai_instance
"""

import concurrent.futures
import io
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import typer

INFERENCE_DIR = Path(__file__).resolve().parent
ZIP_DIR = INFERENCE_DIR / "zip"
WORKSPACE_DIR = INFERENCE_DIR / "workspace"
CONFIG_PATH = INFERENCE_DIR / "config.json"
STATE_PATH = INFERENCE_DIR / ".submission_state.json"

IMAGE = "docker.io/durgesh1993/team_vi:final"
TIMEOUT_SECONDS = 21600

INCLUDE_DIRS = ["input", "workspace"]

EXCLUDE_DIR_NAMES = {"__pycache__", ".temp"}
EXCLUDE_SUFFIXES = {".pyc"}

# Checkpoints, gzipped imaging data, images -- already high-entropy/
# pre-compressed, so DEFLATE burns CPU for near-zero size savings. Store
# these raw; small text/config/code files still get DEFLATE since it's
# cheap and actually shrinks them.
STORE_ONLY_SUFFIXES = (".ckpt", ".pt", ".pth", ".nii.gz", ".gz", ".png", ".jpg", ".jpeg", ".zip")

DEFAULT_PROXY_PORT = 3128

SELF_EXTRACT_HEADER = """#!/bin/sh
set -e
ARCHIVE="$0"
DEST="$(pwd)"

if command -v unzip >/dev/null 2>&1; then
    # unzip exits non-zero when extra bytes precede the archive (this
    # self-extracting header itself) -- expected here, not a real
    # failure, so don't let `set -e` abort on it.
    unzip -o "$ARCHIVE" -d "$DEST" >/dev/null || true
elif command -v python3 >/dev/null 2>&1; then
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ARCHIVE" "$DEST"
else
    echo "Need 'unzip' or 'python3' on PATH to extract this archive." >&2
    exit 1
fi

if [ ! -f "$DEST/workspace/docker.sh" ]; then
    echo "Extraction failed: $DEST/workspace/docker.sh not found." >&2
    exit 1
fi

exec bash "$DEST/workspace/docker.sh" "$@"
"""


# ============================================================
# Local zip/ cleanup (opt-in via --clean-zip)
# ============================================================
def clean_local_zip_dir() -> None:
    if not ZIP_DIR.is_dir():
        return

    print(f"==> Cleaning local {ZIP_DIR}")
    for item in ZIP_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


# ============================================================
# submission.zip (CodaBench artifact)
# ============================================================
def write_submission_zip() -> Path:
    ZIP_DIR.mkdir(exist_ok=True)

    payload = json.dumps({"image": IMAGE, "timeout_seconds": TIMEOUT_SECONDS}, indent=2)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", payload)
    data = buffer.getvalue()

    zip_copy = ZIP_DIR / "submission.zip"
    zip_copy.write_bytes(data)

    workspace_copy = WORKSPACE_DIR / "submission.zip"
    workspace_copy.write_bytes(data)

    print(f"==> Wrote {zip_copy} and {workspace_copy}:\n{payload}")
    return zip_copy


# ============================================================
# Docker archive build (parallel read, mixed STORED/DEFLATED)
# ============================================================
def should_skip(path: Path) -> bool:
    return any(part in EXCLUDE_DIR_NAMES for part in path.parts) or path.suffix in EXCLUDE_SUFFIXES


def compress_type_for(path: Path) -> int:
    name = path.name.lower()
    if name.endswith(STORE_ONLY_SUFFIXES):
        return zipfile.ZIP_STORED
    return zipfile.ZIP_DEFLATED


def collect_files() -> list[Path]:
    files = []
    for dir_name in INCLUDE_DIRS:
        base = INFERENCE_DIR / dir_name
        if not base.is_dir():
            raise FileNotFoundError(f"Expected directory not found: {base}")
        for file_path in base.rglob("*"):
            if file_path.is_dir() or should_skip(file_path):
                continue
            files.append(file_path)
    return files


def _read_file(file_path: Path):
    arcname = str(file_path.relative_to(INFERENCE_DIR))
    data = file_path.read_bytes()
    mtime = time.localtime(file_path.stat().st_mtime)[:6]
    return arcname, data, compress_type_for(file_path), mtime


def build_docker_archive() -> Path:
    ZIP_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_archive = ZIP_DIR / f"docker_{timestamp}.zip"

    files = collect_files()
    workers = min(16, len(files)) or 1
    print(f"==> Reading {len(files)} files with {workers} threads")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_read_file, files))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for arcname, data, ctype, mtime in results:
            zinfo = zipfile.ZipInfo(arcname, date_time=mtime)
            zf.writestr(zinfo, data, compress_type=ctype)

    with open(output_archive, "wb") as f:
        f.write(SELF_EXTRACT_HEADER.encode("utf-8"))
        f.write(zip_buffer.getvalue())

    output_archive.chmod(0o755)

    size_mb = output_archive.stat().st_size / (1024 * 1024)
    print(f"==> Wrote {output_archive} ({size_mb:.1f} MB)")
    return output_archive


# ============================================================
# HTTP CONNECT proxy tunnel -- this script re-invokes itself as its own
# SSH ProxyCommand (`python3 submission.py --tunnel <host> <port>`), since
# this node has no direct route out except through the site's proxy.
# ============================================================
def _pump(read, write) -> None:
    try:
        while True:
            data = read(65536)
            if not data:
                break
            write(data)
    except (OSError, ValueError):
        pass


def run_tunnel(target_host: str, target_port: str, proxy_host: str, proxy_port: int) -> None:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=15)
    request = (
        f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
        f"Host: {target_host}:{target_port}\r\n"
        f"Connection: keep-alive\r\n\r\n"
    ).encode()
    sock.sendall(request)

    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(1)
        if not chunk:
            sys.stderr.write("proxy closed connection during CONNECT\n")
            sys.exit(1)
        response += chunk

    status_line = response.split(b"\r\n", 1)[0].decode(errors="replace")
    if " 200 " not in status_line:
        sys.stderr.write(f"proxy CONNECT failed: {status_line}\n")
        sys.exit(1)

    # The 15s timeout above was only meant for establishing the proxy
    # connection/reading the CONNECT response -- it stays set on the socket
    # otherwise, so any relay-side recv() during a quiet stretch (e.g. a
    # checkpoint load with no output for >15s) would raise socket.timeout,
    # get silently swallowed by _pump's except clause, and leave the tunnel
    # half-broken. The relay itself is long-lived and should block
    # indefinitely instead.
    sock.settimeout(None)

    stdin_fd = sys.stdin.fileno()
    stdout_fd = sys.stdout.fileno()

    t1 = threading.Thread(target=_pump, args=(lambda n: os.read(stdin_fd, n), sock.sendall), daemon=True)
    t2 = threading.Thread(target=_pump, args=(sock.recv, lambda data: os.write(stdout_fd, data)), daemon=True)
    t1.start()
    t2.start()
    while t1.is_alive() and t2.is_alive():
        t1.join(timeout=0.5)
        t2.join(timeout=0.5)


# ============================================================
# Machine config + local run state
# ============================================================
def load_machine_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {CONFIG_PATH} -- create it with host/port/user/key_file "
            f"(or vast_api_key to rent a vast.ai instance instead)."
        )
    config = json.loads(CONFIG_PATH.read_text())

    required = ["key_file"] if config.get("vast_api_key") else ["host", "port", "user", "key_file"]
    for field in required:
        if field not in config:
            raise ValueError(f"config.json missing required field: {field}")
    return config


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(**updates) -> None:
    state = load_state()
    state.update(updates)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def resolve_connection_config(config: dict) -> dict:
    # Fixed machine: host/port/user already live in config.json itself.
    # vast.ai machine: those instead come from whatever create-vast last
    # recorded in .submission_state.json, since the instance (and its
    # IP/port) only exists after that step runs.
    if not config.get("vast_api_key"):
        return config

    vast = load_state().get("vast")
    if not vast:
        raise RuntimeError("No vast.ai instance on record -- run create-vast first.")
    return {**config, **vast}


# ============================================================
# Docker Hub tag deletion + remote docker login
# ============================================================
def _dockerhub_repo_and_tag() -> tuple[str, str]:
    ref = IMAGE[len("docker.io/") :] if IMAGE.startswith("docker.io/") else IMAGE
    repo, _, tag = ref.rpartition(":")
    return repo, tag or "latest"


def delete_dockerhub_tag(config: dict) -> None:
    # Deleting a tag needs a JWT from POST /v2/users/login/ -- Docker Hub
    # rejects that login (and so any deletion) when the "password" given is
    # a Personal Access Token: "access is forbidden with a JWT issued from
    # a personal access token". Only a real account password works here.
    # Kept as a separate docker_password field (not docker_token, which
    # `docker login` on the remote machine DOES accept) and entirely
    # optional -- pushing to the same tag already replaces what anyone
    # pulling gets regardless, this only tidies up Docker Hub's own tag
    # list/storage, so there's no need to put your raw password here
    # unless you specifically want that.
    username = config.get("docker_username")
    password = config.get("docker_password")
    if not username or not password:
        print(
            "==> No docker_password in config.json -- skipping Docker Hub tag deletion "
            "(pushing will still replace the tag's contents either way)"
        )
        return

    repo, tag = _dockerhub_repo_and_tag()

    login_req = urllib.request.Request(
        "https://hub.docker.com/v2/users/login/",
        data=json.dumps({"username": username, "password": password}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(login_req) as resp:
            jwt = json.loads(resp.read())["token"]
    except urllib.error.HTTPError as e:
        print(f"==> Docker Hub login failed ({e.code}), skipping tag deletion: {e.read().decode(errors='replace')}")
        return

    delete_req = urllib.request.Request(
        f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}/",
        headers={"Authorization": f"Bearer {jwt}"},
        method="DELETE",
    )
    print(f"==> Deleting old Docker Hub tag {repo}:{tag}")
    try:
        urllib.request.urlopen(delete_req)
        print(f"==> Deleted {repo}:{tag}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"==> {repo}:{tag} didn't exist yet, nothing to delete")
        else:
            print(f"==> Failed to delete {repo}:{tag} ({e.code}): {e.read().decode(errors='replace')}")


DOCKERHUB_TLS_HOSTS = ["registry-1.docker.io", "auth.docker.io", "index.docker.io"]


def trust_remote_dockerhub_proxy_cert(config: dict) -> None:
    # Some vast.ai hosts run a transparent TLS-intercepting proxy on Docker
    # Hub's hostnames (confirmed by directly inspecting the cert chain --
    # DNS for registry-1.docker.io resolves to a private 192.168.x.x
    # address even when queried against 8.8.8.8, i.e. the redirect happens
    # at the host's network level, not local resolver config -- almost
    # certainly a bandwidth-saving pull-through cache, not something
    # malicious). It terminates TLS with its own cert signed by
    # "My Internal Proxy Root CA" instead of Docker Hub's real one.
    #
    # Docker's `insecure-registries` setting does NOT fix this -- tried it,
    # Docker still enforces strict TLS for the official docker.io registry
    # regardless of that setting. And we don't have the proxy's actual root
    # CA cert to install (it only ever presents its leaf cert during the
    # handshake, never sends its own root). So instead: capture the exact
    # leaf cert each hostname currently presents and add it directly as a
    # trusted anchor -- a non-self-signed cert can still be trusted this
    # way, since verification just needs an exact match in the trust store,
    # not a full chain to a self-signed root. If a host ISN'T doing this
    # interception at all, this captures the real Docker Hub cert instead,
    # which is already trusted anyway -- harmless either way.
    capture_cmds = " && ".join(
        f"(echo | openssl s_client -connect {host}:443 -servername {host} 2>/dev/null "
        f"| openssl x509 | sudo tee /usr/local/share/ca-certificates/vast-proxy-{host}.crt >/dev/null)"
        for host in DOCKERHUB_TLS_HOSTS
    )
    # reset-failed defensively clears systemd's restart rate-limit state --
    # docker.sh restarts docker.service again later (for nvidia-ctk), and
    # two restarts landing close together across these separate SSH calls
    # was enough to trip "start-limit-hit" and fail a later restart even
    # though its config was fine.
    remote_cmd = (
        f"{capture_cmds} && sudo update-ca-certificates && "
        "sudo systemctl reset-failed docker.service 2>/dev/null; "
        "sudo systemctl restart docker"
    )

    print(f"==> Trusting Docker Hub's (possibly proxied) TLS cert on {config['host']}")
    subprocess.run(
        ["ssh", *_ssh_base_args(config), f"{config['user']}@{config['host']}", remote_cmd],
        check=True,
    )


def docker_login_remote(config: dict) -> None:
    # docker_username/docker_token in config.json are both optional -- if
    # either is missing, this is just skipped and you fall back to running
    # `docker login` yourself on the remote machine.
    username = config.get("docker_username")
    token = config.get("docker_token")
    if not username or not token:
        return

    trust_remote_dockerhub_proxy_cert(config)

    remote_cmd = f"docker login --username {shlex.quote(username)} --password-stdin"
    print(f"==> docker login on {config['host']} as {username}")
    # Token goes over the SSH session's stdin, not embedded in remote_cmd --
    # the remote command line (and thus this token) would otherwise be
    # visible to anyone else on that machine via `ps`.
    subprocess.run(
        ["ssh", *_ssh_base_args(config), f"{config['user']}@{config['host']}", remote_cmd],
        input=token.encode(),
        check=True,
    )


# ============================================================
# SSH/rsync plumbing
# ============================================================
def _resolved_key_path(config: dict) -> Path:
    key_file = Path(config["key_file"])
    if not key_file.is_absolute():
        key_file = INFERENCE_DIR / key_file
    key_file.chmod(0o600)
    return key_file


def _proxy_command(config: dict) -> str | None:
    # Only tunnel through the site proxy if the config explicitly asks for
    # it -- no default fallback, so a config with proxy_host missing, null,
    # or "" connects directly (e.g. a different cluster with no proxy).
    # Proxy host/port are passed as positional args (not inline shell
    # env-assignment) so this string works whether it ends up word-split by
    # rsync's naive -e parser or handed to ssh directly.
    proxy_host = config.get("proxy_host")
    if not proxy_host:
        return None
    proxy_port = config.get("proxy_port") or DEFAULT_PROXY_PORT
    this_file = Path(__file__).resolve()
    return f"python3 {this_file} --tunnel %h %p {proxy_host} {proxy_port}"


def _ssh_base_args(config: dict) -> list[str]:
    args = [
        "-p", str(config["port"]),
        "-i", str(_resolved_key_path(config)),
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    proxy_command = _proxy_command(config)
    if proxy_command is not None:
        args += ["-o", f"ProxyCommand={proxy_command}"]
    return args


def _quote_remote_path(path: str) -> str:
    # shlex.quote would wrap a leading "~" in literal quotes, which disables
    # the remote shell's tilde expansion entirely -- leave a leading ~
    # unquoted (safe: remote_dir is trusted local config, not untrusted
    # input).
    return path if path.startswith("~") else shlex.quote(path)


def clean_remote(config: dict) -> None:
    remote_dir = config.get("remote_dir", "~/")
    remote_dir_clean = remote_dir.rstrip("/")

    # Guard against a misconfigured remote_dir ("", "/", ".") turning
    # `rm -rf <dir>/*` into something catastrophic on the remote machine.
    if remote_dir_clean in ("", "/", ".", ".."):
        raise ValueError(f"Refusing to clean unsafe remote_dir: {remote_dir!r}")

    remote_cmd = f"rm -rf {_quote_remote_path(remote_dir_clean)}/*"
    print(f"==> Cleaning {config['host']}:{remote_dir_clean}/*")
    subprocess.run(
        ["ssh", *_ssh_base_args(config), f"{config['user']}@{config['host']}", remote_cmd],
        check=True,
    )


def push_archive(archive_path: Path, config: dict) -> str:
    remote_dir = config.get("remote_dir", "~/")
    # rsync's -e value is a single string it re-splits itself (unlike
    # subprocess's list-based argv) -- shlex.quote each piece so the
    # ProxyCommand value (which contains spaces) survives that re-split
    # intact instead of being torn into garbage tokens.
    ssh_cmd = "ssh " + " ".join(shlex.quote(arg) for arg in _ssh_base_args(config))
    dest = f"{config['user']}@{config['host']}:{remote_dir}"

    print(f"==> Pushing {archive_path.name} to {dest}")
    subprocess.run(
        ["rsync", "-avP", "--partial", "-e", ssh_cmd, str(archive_path), dest],
        check=True,
    )

    remote_dir_clean = remote_dir.rstrip("/") or "."
    return f"{remote_dir_clean}/{archive_path.name}"


def trigger_remote_build(remote_path: str, config: dict, push: bool) -> None:
    remote_cmd = f"bash {remote_path}" + (" --push" if push else "")
    print(f"==> Running on {config['host']}: {remote_cmd}")
    # -tt forces a remote pseudo-terminal -- without one, docker build/run's
    # progress output detects a non-interactive pipe and mostly stops
    # streaming live, only flushing in large chunks. Only used here, never
    # for push_archive's rsync transport, which needs a plain pty-less
    # channel.
    subprocess.run(
        ["ssh", "-tt", *_ssh_base_args(config), f"{config['user']}@{config['host']}", remote_cmd],
        check=True,
    )


def pull_remote_output(timestamp: str, config: dict) -> Path | None:
    remote_dir = config.get("remote_dir", "~/")
    remote_dir_clean = remote_dir.rstrip("/") or "."
    output_name = f"output_docker_{timestamp}.zip"

    zip_snippet = (
        "import zipfile, pathlib; "
        "out = pathlib.Path('output'); "
        f"zf = zipfile.ZipFile('{output_name}', 'w', zipfile.ZIP_DEFLATED); "
        "[zf.write(p, str(p)) for p in out.rglob('*') if p.is_file()] if out.is_dir() else None; "
        "zf.close()"
    )
    remote_cmd = f"cd {_quote_remote_path(remote_dir_clean)} && python3 -c \"{zip_snippet}\""

    print(f"==> Zipping remote output/ into {output_name}")
    try:
        subprocess.run(
            ["ssh", *_ssh_base_args(config), f"{config['user']}@{config['host']}", remote_cmd],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"==> Could not zip remote output/ ({e}); skipping pull-back")
        return None

    ZIP_DIR.mkdir(exist_ok=True)
    local_path = ZIP_DIR / output_name
    ssh_cmd = "ssh " + " ".join(shlex.quote(arg) for arg in _ssh_base_args(config))
    src = f"{config['user']}@{config['host']}:{remote_dir_clean}/{output_name}"

    print(f"==> Pulling {output_name} to {local_path}")
    try:
        subprocess.run(
            ["rsync", "-avP", "--partial", "-e", ssh_cmd, src, str(local_path)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"==> Could not pull {output_name} back ({e})")
        return None

    print(f"==> Wrote {local_path}")
    return local_path


# ============================================================
# vast.ai instance provisioning (used when config.json has vast_api_key)
# ============================================================
VASTAI_DEFAULT_MIN_CUDA = "12.4"
# vast.ai's standard container-type instances have Docker-in-Docker
# disabled ("disabled for security", per their own FAQ) -- a plain image
# like pytorch/pytorch has no docker binary at all, and even installing
# one wouldn't get a working daemon since the container itself lacks the
# privileges. vastai/kvm is their official KVM-*virtualized* template --
# a real VM with full kernel/cgroup privileges, not a shared-kernel
# container, so Docker just runs normally inside it. Confirmed working by
# direct testing (not just docs) before this default was set. CLI variant
# (no desktop GUI, ~4.8GB vs ubuntu_desktop's ~9.67GB) -- we only need
# SSH+Docker, not a display.
VASTAI_DEFAULT_IMAGE = "docker.io/vastai/kvm:ubuntu_cli_22.04-2025-11-21"
VASTAI_DEFAULT_DISK_GB = 40
# Tags instances this script creates so find_existing_vastai_instance can
# tell them apart from any unrelated instances on the same vast.ai account
# -- reusing/destroying an instance we didn't create would be bad. Not
# confirmed against current `vastai create instance --help` output at
# write time; if instance creation errors on --label, drop it (and reuse
# detection will just always create a new instance instead).
VASTAI_LABEL = "mvaa-submission"


def _vastai_env(config: dict) -> dict:
    env = os.environ.copy()
    api_key = config.get("vast_api_key")
    if api_key:
        env["VAST_API_KEY"] = api_key
    return env


def _vastai_run(args: list[str], config: dict, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["vastai", *args], env=_vastai_env(config), text=True, **kwargs)


def find_vastai_offer(config: dict) -> dict:
    # cuda_max_good is vast.ai's query field for the highest CUDA version a
    # host's driver supports -- this is the one thing here not confirmed
    # against current `vastai search offers --help` output at write time,
    # so double check it if this comes back empty/errors. No upper bound on
    # it (cuda_max_good=13.x hosts are fine -- driver backward compat means
    # a newer driver still runs the pinned cu124 image; confirmed earlier
    # in this project on an RTX 4090). It only constrains the *driver*
    # though, not the GPU's compute capability (sm_XX) -- a host could
    # still be e.g. Blackwell/sm_120, which the pinned cu124 image has no
    # kernels for at all regardless of driver version (see the "no kernel
    # image available" investigation earlier in this project).
    #
    # gpu_ram>=24 is a practical stand-in for "4090-class or better" --
    # vast.ai has no direct GPU-tier comparison field, but 24GB matches the
    # 4090's own VRAM and rules out weaker cards. Adjust vast_query
    # yourself for a stricter match (e.g. an explicit gpu_name list).
    #
    # vms_enabled=true is required -- the vastai/kvm VM image (see
    # VASTAI_DEFAULT_IMAGE) only boots on hosts that support VM rental at
    # all; without this filter, offer selection could land on a
    # container-only host where Docker-in-Docker is disabled again.
    #
    # reliability>0.95 and inet_down>100 rule out the flaky/slow hosts that
    # otherwise tend to be the absolute cheapest -- "cheapest" below is
    # cheapest among these, not cheapest overall. Without this, the
    # ~4.8GB VM image can take a very long time to pull on first boot on a
    # slow/unreliable host, which is what "cheapest instance takes forever
    # to start" was.
    #
    # dlperf>=100 is vast.ai's own single-number deep-learning performance
    # score (roughly comparable across different GPU models/counts) -- adds
    # a floor on actual compute, not just VRAM/driver/network, since a host
    # can pass every other filter here and still be a weak or oversubscribed
    # card.
    min_cuda = config.get("vast_min_cuda", VASTAI_DEFAULT_MIN_CUDA)
    min_reliability = config.get("vast_min_reliability", 0.95)
    min_inet_down = config.get("vast_min_inet_down", 100)
    min_dlperf = config.get("vast_min_dlperf", 100)
    query = config.get(
        "vast_query",
        f"cuda_max_good>={min_cuda} gpu_ram>=24 num_gpus=1 rentable=true verified=true "
        f"vms_enabled=true reliability>{min_reliability} inet_down>{min_inet_down} "
        f"dlperf>={min_dlperf}",
    )
    print(f"==> Searching vast.ai offers: {query}")
    result = _vastai_run(["search", "offers", query, "--raw"], config, capture_output=True, check=True)
    offers = json.loads(result.stdout)
    # Sort explicitly by price rather than relying on the CLI's -o sort-flag
    # semantics ("select the offer with min price" must hold regardless of
    # which way that flag's +/- turns out to sort).
    offers.sort(key=lambda o: o.get("dph_total", float("inf")))
    if not offers:
        raise RuntimeError(f"No vast.ai offers matched: {query}")

    offer = offers[0]
    print(
        f"==> Selected offer {offer.get('id')}: {offer.get('gpu_name')} "
        f"cuda_max_good={offer.get('cuda_max_good')} ${offer.get('dph_total')}/hr"
    )
    return offer


def find_existing_vastai_instance(config: dict) -> str | None:
    # Avoids paying for + creating a second instance when one of ours is
    # already up -- reuse it instead.
    result = _vastai_run(["show", "instances", "--raw"], config, capture_output=True, check=True)
    instances = json.loads(result.stdout)

    for inst in instances:
        if inst.get("label") != VASTAI_LABEL:
            continue
        status = inst.get("actual_status") or inst.get("status")
        if status in ("running", "loading"):
            instance_id = str(inst["id"])
            print(f"==> Reusing existing vast.ai instance {instance_id} (status={status})")
            return instance_id

    return None


def create_vastai_instance(offer: dict, config: dict) -> str:
    image = config.get("vast_image", VASTAI_DEFAULT_IMAGE)
    disk_gb = config.get("vast_disk_gb", VASTAI_DEFAULT_DISK_GB)

    print(f"==> Renting vast.ai offer {offer['id']} (image={image}, disk={disk_gb}GB)")
    result = _vastai_run(
        [
            "create", "instance", str(offer["id"]),
            "--image", image,
            "--disk", str(disk_gb),
            "--ssh", "--direct",
            "--label", VASTAI_LABEL,
            "--raw",
        ],
        config,
        capture_output=True,
        check=True,
    )
    data = json.loads(result.stdout)
    instance_id = data.get("new_contract") or data.get("id")
    if instance_id is None:
        raise RuntimeError(f"Could not find new instance id in vastai response: {data}")

    instance_id = str(instance_id)
    print(f"==> Created instance {instance_id}")
    return instance_id


def wait_for_vastai_instance(instance_id: str, config: dict, timeout_s: int = 900) -> None:
    print(f"==> Waiting for instance {instance_id} to reach 'running'")
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        result = _vastai_run(["show", "instance", instance_id, "--raw"], config, capture_output=True, check=True)
        info = json.loads(result.stdout)
        status = info.get("actual_status") or info.get("status")

        if status == "running":
            print(f"==> Instance {instance_id} is running")
            return

        print(f"==> Instance {instance_id} status: {status}, waiting...")
        time.sleep(10)

    raise TimeoutError(f"vast.ai instance {instance_id} did not reach 'running' within {timeout_s}s")


def vastai_ssh_target(instance_id: str, config: dict) -> dict:
    result = _vastai_run(["ssh-url", instance_id], config, capture_output=True, check=True)
    ssh_url = result.stdout.strip()
    parsed = urllib.parse.urlsplit(ssh_url)

    if not parsed.hostname:
        raise RuntimeError(f"Could not parse vastai ssh-url output: {ssh_url!r}")

    return {"host": parsed.hostname, "port": parsed.port or 22, "user": parsed.username or "root"}


def wait_for_ssh_ready(config: dict, timeout_s: int = 180, interval_s: int = 5) -> None:
    # vast.ai's "running" status can land slightly before sshd inside the
    # container is actually accepting connections -- probing with a real
    # SSH connection (rather than trusting the status field alone) avoids
    # the "Connection closed by UNKNOWN port 65535" failure that shows up
    # when clean_remote/push_archive hit that gap.
    ssh_args = ["ssh", "-o", "ConnectTimeout=10", *_ssh_base_args(config), f"{config['user']}@{config['host']}", "true"]
    using_proxy = _proxy_command(config) is not None
    print(f"==> Waiting for SSH on {config['host']}:{config['port']} (proxy={'on' if using_proxy else 'off'})")
    print(f"==> Probe command: {' '.join(shlex.quote(a) for a in ssh_args)}")

    deadline = time.time() + timeout_s
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        result = subprocess.run(ssh_args, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"==> SSH is ready on {config['host']}:{config['port']} (attempt {attempt})")
            return

        # Printed every attempt, not just at the end -- this is the actual
        # diagnostic (proxy CONNECT failure vs auth vs connection refused
        # vs something else), not a silent retry loop.
        print(f"==> Attempt {attempt} failed (rc={result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
        time.sleep(interval_s)

    raise TimeoutError(f"SSH never became ready on {config['host']}:{config['port']} within {timeout_s}s")


def destroy_vastai_instance(instance_id: str, config: dict) -> None:
    print(f"==> Destroying vast.ai instance {instance_id}")
    # `vastai destroy instance` prompts "Are you sure? [y/N]" interactively
    # with no confirmed --yes/-y flag to skip it (not in current docs at
    # write time) -- answer it over stdin instead.
    _vastai_run(["destroy", "instance", instance_id], config, input="y\n", check=False)


# ============================================================
# The 5 actions, callable directly (menu) or via typer (CLI)
# ============================================================
def create_vast_action(config: dict) -> dict:
    if not config.get("vast_api_key"):
        raise RuntimeError("config.json has no vast_api_key -- nothing to rent (this is for a fixed machine).")

    instance_id = find_existing_vastai_instance(config)
    if instance_id is None:
        offer = find_vastai_offer(config)
        instance_id = create_vastai_instance(offer, config)

    wait_for_vastai_instance(instance_id, config)
    ssh_target = vastai_ssh_target(instance_id, config)
    conn = {**config, **ssh_target}
    wait_for_ssh_ready(conn)

    save_state(vast={"instance_id": instance_id, **ssh_target})
    print(f"==> Ready: {ssh_target['user']}@{ssh_target['host']}:{ssh_target['port']} (instance {instance_id})")
    return conn


def delete_vast_action(config: dict) -> None:
    state = load_state()
    vast = state.get("vast")
    instance_id = vast["instance_id"] if vast else find_existing_vastai_instance(config)

    if instance_id is None:
        print("==> No vast.ai instance on record -- nothing to delete.")
        return

    destroy_vastai_instance(instance_id, config)
    if vast:
        state.pop("vast", None)
        STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"==> Destroyed instance {instance_id}")


def push_docker_action(config: dict, clean_zip: bool = False) -> str:
    if clean_zip:
        clean_local_zip_dir()

    write_submission_zip()
    archive_path = build_docker_archive()

    conn = resolve_connection_config(config)
    clean_remote(conn)
    remote_path = push_archive(archive_path, conn)

    timestamp = archive_path.stem.removeprefix("docker_")
    save_state(last_remote_path=remote_path, last_archive_timestamp=timestamp)
    print(f"==> {remote_path} is on {conn['host']}, ready for trigger-build")
    return remote_path


def trigger_build_action(config: dict, push: bool = False) -> None:
    conn = resolve_connection_config(config)
    remote_path = load_state().get("last_remote_path")
    if not remote_path:
        raise RuntimeError("No pushed archive on record -- run push-docker first.")

    if push:
        delete_dockerhub_tag(conn)
        docker_login_remote(conn)

    trigger_remote_build(remote_path, conn, push=push)


def pull_output_action(config: dict) -> Path | None:
    conn = resolve_connection_config(config)
    timestamp = load_state().get("last_archive_timestamp") or time.strftime("%Y%m%d_%H%M%S")
    return pull_remote_output(timestamp, conn)


# ============================================================
# Interactive menu
# ============================================================
def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    answer = input(f"{prompt} [{suffix}] ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def interactive_menu() -> None:
    menu = [
        ("1", "Create vast.ai instance", lambda: create_vast_action(load_machine_config())),
        ("2", "Delete vast.ai instance", lambda: delete_vast_action(load_machine_config())),
        (
            "3",
            "Push docker (build zips + rsync archive to remote)",
            lambda: push_docker_action(
                load_machine_config(), clean_zip=_ask_yes_no("Clean local zip/ first?")
            ),
        ),
        (
            "4",
            "Trigger build (run workspace/docker.sh on remote)",
            lambda: trigger_build_action(
                load_machine_config(), push=_ask_yes_no("Also push image to Docker Hub?")
            ),
        ),
        ("5", "Pull output (fetch remote output/ as a local zip)", lambda: pull_output_action(load_machine_config())),
    ]

    while True:
        print("\n=== MVAA submission tool ===")
        for key, label, _ in menu:
            print(f"  {key}. {label}")
        print("  0. Exit")

        choice = input("Select option: ").strip().lower()
        if choice in ("0", "q", "quit", "exit"):
            return

        match = next((m for m in menu if m[0] == choice), None)
        if match is None:
            print("Invalid option, try again.")
            continue

        try:
            match[2]()
        except Exception as e:
            print(f"==> Error: {e}")


# ============================================================
# Typer CLI
# ============================================================
app = typer.Typer(add_completion=False, help="MVAA submission build/push/test tool.")


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        interactive_menu()


@app.command("create-vast")
def create_vast_cmd() -> None:
    """1. Rent (or reuse) a vast.ai GPU instance and wait until SSH is reachable."""
    create_vast_action(load_machine_config())


@app.command("delete-vast")
def delete_vast_cmd() -> None:
    """2. Destroy the vast.ai instance currently on record."""
    delete_vast_action(load_machine_config())


@app.command("push-docker")
def push_docker_cmd(
    clean_zip: bool = typer.Option(False, "--clean-zip", help="Delete old zip/ contents first."),
) -> None:
    """3. Build submission.zip + the docker build archive, and rsync it to the remote machine."""
    push_docker_action(load_machine_config(), clean_zip=clean_zip)


@app.command("trigger-build")
def trigger_build_cmd(
    push: bool = typer.Option(False, "--push", help="Also push the built image to Docker Hub."),
) -> None:
    """4. Run workspace/docker.sh on the remote machine against the last pushed archive."""
    trigger_build_action(load_machine_config(), push=push)


@app.command("pull-output")
def pull_output_cmd() -> None:
    """5. Zip the remote output/ dir and pull it back to zip/output_docker_<timestamp>.zip."""
    pull_output_action(load_machine_config())


if __name__ == "__main__":
    # `--tunnel` is this script re-invoking itself as an SSH ProxyCommand
    # (see _proxy_command) -- intercepted before typer ever sees argv, since
    # it isn't one of the 5 subcommands above.
    if len(sys.argv) > 1 and sys.argv[1] == "--tunnel":
        _, target_host, target_port, proxy_host, proxy_port = sys.argv[1:6]
        run_tunnel(target_host, target_port, proxy_host, int(proxy_port))
    else:
        app()
