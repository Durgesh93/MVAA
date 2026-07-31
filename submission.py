"""
Typer CLI for building the MVAA CodaBench submission and shipping it to a
remote machine -- a fixed one, or (if config.json has vast_api_key) a rented
vast.ai instance.

Run with no args for an interactive menu, or invoke each step directly:

    python3 submission.py                    # interactive menu
    python3 submission.py create-vast         # 1. rent/reuse a vast.ai instance
    python3 submission.py delete-vast         # 2. destroy the instance on record
    python3 submission.py setup supervised [--version best|last|swa]  # 3. stage input/ckpts/plans for a branch
    python3 submission.py push-docker         # 4. build zips + rsync to remote
    python3 submission.py trigger-build [--push]  # 5. run workspace/docker.sh remotely
    python3 submission.py pull-output         # 6. pull the remote output/ back as a zip

Steps are independent -- rerun just one as needed (workspace/{ckpts,plans,
input}/ and zip/ all persist between runs). Connection details for the
current vast.ai instance are remembered in both config.json's own "vast"
field (durable, survives a stale/rotated instance since it's what
resolve_connection_config reads first) and .submission_state.json
(gitignored, auto-managed "last known" bookkeeping used as a fallback).
create-vast checks config.json's recorded instance id first and reuses it
(re-fetching host/port fresh, in case those rotated) if it's still
running -- only searching/renting a new one if it isn't. No separate
command or instance-id flag needed to reconnect.

Artifacts, written under zip/ by push-docker:
    zip/submission.zip         -- uploaded to CodaBench (points at IMAGE;
                                   must match workspace/docker.sh's IMAGE_TAG)
    zip/docker_<timestamp>.zip -- self-extracting archive of workspace/
                                   (input/ckpts/plans included), also
                                   copied to workspace/submission.zip so
                                   it travels inside the archive

The archive is a normal zip with a shell script prepended -- still
`unzip`-able, but also directly executable (extracts and hands off to
workspace/docker.sh).

This node has no direct internet route except the site's HTTP proxy, so
rsync/ssh are tunneled through this script re-invoking itself as its own
ProxyCommand (see --tunnel). vast.ai instances still need the proxy too,
since it's about where *this script* runs, not the target.

config.json fields:
    host, port, user, key_file    -- fixed machine (ignored if vast_api_key set)
    remote_dir                    -- optional, default "~/"
    proxy_host, proxy_port        -- optional; when unset, falls back to this shell's own
                                      HTTPS_PROXY/HTTP_PROXY env vars, or direct if neither is set
    docker_username, docker_token -- optional, enables remote `docker login` on --push
    docker_password               -- optional, enables deleting the old Docker Hub tag first
    vast_api_key                  -- switches to renting a vast.ai instance
    vast_query, vast_image, vast_disk_gb, vast_min_cuda -- optional vast.ai tuning
    vast                          -- {instance_id, host, port, user} of the current
                                      instance; written by create-vast, but can also
                                      be hand-edited/pasted in directly -- create-vast
                                      reuses instance_id if it's still running
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
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

INFERENCE_DIR = Path(__file__).resolve().parent
ZIP_DIR = INFERENCE_DIR / "zip"
WORKSPACE_DIR = INFERENCE_DIR / "workspace"
CONFIG_PATH = INFERENCE_DIR / "config.json"
STATE_PATH = INFERENCE_DIR / ".submission_state.json"

console = Console()


def info(message: str, emoji: str = "➜") -> None:
    console.print(f"[bold cyan]{emoji} [/] {message}")


def success(message: str, emoji: str = "✅") -> None:
    console.print(f"[bold green]{emoji} [/] {message}")


def warn(message: str, emoji: str = "⚠️ ") -> None:
    console.print(f"[bold yellow]{emoji}[/] {message}")


def error(message: str, emoji: str = "❌") -> None:
    console.print(f"[bold red]{emoji} [/] {message}")

IMAGE = "docker.io/durgesh1993/team_vi:final"
TIMEOUT_SECONDS = 21600

INCLUDE_DIRS = ["workspace"]

EXCLUDE_DIR_NAMES = {"__pycache__", ".temp"}
EXCLUDE_SUFFIXES = {".pyc"}

# Already high-entropy/compressed -- DEFLATE would burn CPU for no size win.
STORE_ONLY_SUFFIXES = (".ckpt", ".pt", ".pth", ".nii.gz", ".gz", ".png", ".jpg", ".jpeg", ".zip")

DEFAULT_PROXY_PORT = 3128

SELF_EXTRACT_HEADER = """#!/bin/sh
set -e
ARCHIVE="$0"
DEST="$(pwd)"

if command -v unzip >/dev/null 2>&1; then
    # Non-zero exit here is expected (prepended header bytes), not a real failure.
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

    info(f"Cleaning local {ZIP_DIR}", "🧹")
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

    success(f"Wrote {zip_copy} and {workspace_copy}:", "📦")
    console.print_json(payload)
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
    info(f"Reading {len(files)} files with {workers} threads", "📖")

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
    success(f"Wrote {output_archive} ({size_mb:.1f} MB)", "🐳")
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

    # Clear the connect-only timeout -- otherwise a quiet stretch (e.g. a
    # slow checkpoint load) trips socket.timeout mid-relay and half-breaks the tunnel.
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


def save_config_vast(vast: dict | None) -> None:
    # Mirrors the resolved instance into config.json itself (durable,
    # user-editable) rather than only .submission_state.json (gitignored,
    # auto-managed "last known" bookkeeping) -- so a specific instance can
    # be pinned by hand-editing config.json too, and survives independent
    # of whatever push-docker/trigger-build last recorded. vast=None clears it.
    config = json.loads(CONFIG_PATH.read_text())
    if vast is None:
        config.pop("vast", None)
    else:
        config["vast"] = vast
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def resolve_connection_config(config: dict) -> dict:
    # vast.ai: host/port/user come from config.json's own "vast" block if
    # present (set by create-vast), else .submission_state.json's last
    # recorded instance.
    if not config.get("vast_api_key"):
        return config

    vast = config.get("vast") or load_state().get("vast")
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
    # Tag deletion needs a JWT from a real password login -- access tokens
    # (docker_token) are rejected here, so this is a separate optional field.
    # Pushing already overwrites the tag's contents either way; this just
    # tidies up Docker Hub's own tag list.
    username = config.get("docker_username")
    password = config.get("docker_password")
    if not username or not password:
        warn(
            "No docker_password in config.json -- skipping Docker Hub tag deletion "
            "(pushing will still replace the tag's contents either way)",
            "🏷️ ",
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
        error(f"Docker Hub login failed ({e.code}), skipping tag deletion: {e.read().decode(errors='replace')}", "🔐")
        return

    delete_req = urllib.request.Request(
        f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}/",
        headers={"Authorization": f"Bearer {jwt}"},
        method="DELETE",
    )
    info(f"Deleting old Docker Hub tag {repo}:{tag}", "🗑️ ")
    try:
        urllib.request.urlopen(delete_req)
        success(f"Deleted {repo}:{tag}", "🗑️ ")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            info(f"{repo}:{tag} didn't exist yet, nothing to delete", "ℹ️ ")
        else:
            error(f"Failed to delete {repo}:{tag} ({e.code}): {e.read().decode(errors='replace')}")


DOCKERHUB_TLS_HOSTS = ["registry-1.docker.io", "auth.docker.io", "index.docker.io"]


def trust_remote_dockerhub_proxy_cert(config: dict) -> None:
    # Some vast.ai hosts transparently MITM Docker Hub's TLS with their own
    # cert (a pull cache, not malicious). `insecure-registries` doesn't
    # bypass this, and we never see the proxy's root CA, only its leaf --
    # so trust each hostname's current leaf cert directly (exact-match
    # trust needs no root chain). Harmless if a host isn't intercepting.
    capture_cmds = " && ".join(
        f"(echo | openssl s_client -connect {host}:443 -servername {host} 2>/dev/null "
        f"| openssl x509 | sudo tee /usr/local/share/ca-certificates/vast-proxy-{host}.crt >/dev/null)"
        for host in DOCKERHUB_TLS_HOSTS
    )
    # reset-failed clears systemd's restart rate-limit -- docker.sh restarts
    # docker again later and back-to-back restarts tripped "start-limit-hit".
    remote_cmd = (
        f"{capture_cmds} && sudo update-ca-certificates && "
        "sudo systemctl reset-failed docker.service 2>/dev/null; "
        "sudo systemctl restart docker"
    )

    info(f"Trusting Docker Hub's (possibly proxied) TLS cert on {config['host']}", "🔒")
    subprocess.run(
        ["ssh", *_ssh_base_args(config), f"{config['user']}@{config['host']}", remote_cmd],
        check=True,
    )


def docker_login_remote(config: dict) -> None:
    # Optional -- skip and fall back to running `docker login` yourself remotely.
    username = config.get("docker_username")
    token = config.get("docker_token")
    if not username or not token:
        return

    trust_remote_dockerhub_proxy_cert(config)

    remote_cmd = f"docker login --username {shlex.quote(username)} --password-stdin"
    info(f"docker login on {config['host']} as {username}", "🔑")
    # Token goes over stdin, not the command line, so `ps` can't leak it.
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


def _env_proxy_host_port() -> tuple[str, int] | None:
    # Falls back to this shell's own HTTPS_PROXY/HTTP_PROXY (Olivia's
    # compute nodes export these already) when config.json doesn't pin an
    # explicit proxy_host -- so ssh/rsync automatically match whatever
    # this machine actually needs instead of a stale value left over from
    # a different machine (e.g. one with direct internet access).
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(var)
        if value:
            parsed = urllib.parse.urlsplit(value)
            if parsed.hostname:
                return parsed.hostname, parsed.port or DEFAULT_PROXY_PORT
    return None


def _proxy_command(config: dict) -> str | None:
    # Positional args (not env-assignment) so this survives rsync's -e word-split.
    proxy_host = config.get("proxy_host")
    proxy_port = config.get("proxy_port")

    if not proxy_host:
        env_proxy = _env_proxy_host_port()
        if env_proxy is None:
            return None
        proxy_host, proxy_port = env_proxy

    proxy_port = proxy_port or DEFAULT_PROXY_PORT
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
    # shlex.quote would block tilde expansion -- leave a leading ~ unquoted
    # (safe: remote_dir is trusted config, not untrusted input).
    return path if path.startswith("~") else shlex.quote(path)


def clean_remote(config: dict) -> None:
    remote_dir = config.get("remote_dir", "~/")
    remote_dir_clean = remote_dir.rstrip("/")

    # Guard against a misconfigured remote_dir turning this into `rm -rf /*`.
    if remote_dir_clean in ("", "/", ".", ".."):
        raise ValueError(f"Refusing to clean unsafe remote_dir: {remote_dir!r}")

    remote_cmd = f"rm -rf {_quote_remote_path(remote_dir_clean)}/*"
    info(f"Cleaning {config['host']}:{remote_dir_clean}/*", "🧹")
    subprocess.run(
        ["ssh", *_ssh_base_args(config), f"{config['user']}@{config['host']}", remote_cmd],
        check=True,
    )


def push_archive(archive_path: Path, config: dict) -> str:
    remote_dir = config.get("remote_dir", "~/")
    # rsync re-splits -e itself -- quote each piece so ProxyCommand's spaces survive.
    ssh_cmd = "ssh " + " ".join(shlex.quote(arg) for arg in _ssh_base_args(config))
    dest = f"{config['user']}@{config['host']}:{remote_dir}"

    info(f"Pushing {archive_path.name} to {dest}", "⬆️ ")
    subprocess.run(
        ["rsync", "-avP", "--partial", "-e", ssh_cmd, str(archive_path), dest],
        check=True,
    )

    remote_dir_clean = remote_dir.rstrip("/") or "."
    return f"{remote_dir_clean}/{archive_path.name}"


def trigger_remote_build(remote_path: str, config: dict, push: bool) -> None:
    remote_cmd = f"bash {remote_path}" + (" --push" if push else "")
    info(f"Running on {config['host']}: {remote_cmd}", "▶️ ")
    # -tt forces a pty so docker's progress output streams live instead of
    # buffering in chunks -- not used for push_archive's rsync, which needs no pty.
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

    info(f"Zipping remote output/ into {output_name}", "🗜️ ")
    try:
        subprocess.run(
            ["ssh", *_ssh_base_args(config), f"{config['user']}@{config['host']}", remote_cmd],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        warn(f"Could not zip remote output/ ({e}); skipping pull-back")
        return None

    ZIP_DIR.mkdir(exist_ok=True)
    local_path = ZIP_DIR / output_name
    ssh_cmd = "ssh " + " ".join(shlex.quote(arg) for arg in _ssh_base_args(config))
    src = f"{config['user']}@{config['host']}:{remote_dir_clean}/{output_name}"

    info(f"Pulling {output_name} to {local_path}", "⬇️ ")
    try:
        subprocess.run(
            ["rsync", "-avP", "--partial", "-e", ssh_cmd, src, str(local_path)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        warn(f"Could not pull {output_name} back ({e})")
        return None

    success(f"Wrote {local_path}", "💾")
    return local_path


# ============================================================
# vast.ai instance provisioning (used when config.json has vast_api_key)
# ============================================================
VASTAI_DEFAULT_MIN_CUDA = "12.4"
# Standard vast.ai container instances have Docker-in-Docker disabled, so
# plain images can't run Docker at all. vastai/kvm is a real KVM VM instead
# (confirmed working directly) -- CLI variant since we only need SSH+Docker.
VASTAI_DEFAULT_IMAGE = "docker.io/vastai/kvm:ubuntu_cli_22.04-2025-11-21"
VASTAI_DEFAULT_DISK_GB = 40
# Tags instances this script creates so we never reuse/destroy someone
# else's instance. Drop --label if instance creation errors on it.
VASTAI_LABEL = "mvaa-submission"


def _vastai_env(config: dict) -> dict:
    env = os.environ.copy()
    api_key = config.get("vast_api_key")
    if api_key:
        env["VAST_API_KEY"] = api_key
    return env


def _vastai_run(args: list[str], config: dict, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["vastai", *args], env=_vastai_env(config), text=True, **kwargs)


def _default_vastai_queries(config: dict) -> list[str]:
    # cuda_max_good: highest CUDA a host's driver supports -- no upper bound
    # needed (driver backward compat covers the pinned cu124 image), but it
    # doesn't constrain GPU compute capability (a Blackwell/sm_120 host would
    # still have no kernels for cu124, regardless of driver version).
    # gpu_ram>=24: stand-in for "4090-class or better" (no direct GPU-tier field).
    # vms_enabled=true: required for vastai/kvm's VM image to boot at all --
    # never relaxed, unlike every other filter below.
    min_cuda = config.get("vast_min_cuda", VASTAI_DEFAULT_MIN_CUDA)
    min_reliability = config.get("vast_min_reliability", 0.95)
    min_inet_down = config.get("vast_min_inet_down", 100)
    min_dlperf = config.get("vast_min_dlperf", 100)
    base = f"cuda_max_good>={min_cuda} gpu_ram>=24 num_gpus=1 rentable=true vms_enabled=true"

    # Tried strictest-first, each step dropping one filter -- on the live
    # market a hard dlperf>=100 alone excludes most single RTX 4090 offers
    # (they benchmark ~97-99 on vast.ai's own scale), so combined with
    # verified/reliability/inet_down the strict query routinely narrows
    # down to 0-1 candidates. Falling through to a looser step is always
    # logged, never silent.
    return [
        f"{base} verified=true reliability>{min_reliability} inet_down>{min_inet_down} dlperf>={min_dlperf}",
        f"{base} reliability>{min_reliability} inet_down>{min_inet_down} dlperf>={min_dlperf / 2}",
        f"{base} reliability>0.9",
        base,
    ]


def find_vastai_offer(config: dict) -> dict:
    # An explicit vast_query in config.json is taken as-is, no broadening --
    # the user already said exactly what they want.
    queries = [config["vast_query"]] if "vast_query" in config else _default_vastai_queries(config)

    offers = []
    for step, query in enumerate(queries):
        if step > 0:
            warn(f"No offers matched -- broadening search (step {step + 1}/{len(queries)})", "🔍")
        info(f"Searching vast.ai offers: {query}", "🔍")
        result = _vastai_run(["search", "offers", query, "--raw"], config, capture_output=True, check=True)
        offers = json.loads(result.stdout)
        if offers:
            break

    if not offers:
        raise RuntimeError(f"No vast.ai offers matched any query, last tried: {queries[-1]}")

    # Sort explicitly by price rather than trusting the CLI's -o sort-flag semantics.
    offers.sort(key=lambda o: o.get("dph_total", float("inf")))

    offer = offers[0]
    success(
        f"Selected offer {offer.get('id')}: {offer.get('gpu_name')} "
        f"cuda_max_good={offer.get('cuda_max_good')} ${offer.get('dph_total')}/hr",
        "🎯",
    )
    return offer


def find_existing_vastai_instance(config: dict) -> str | None:
    # Reuse an already-running instance instead of paying for a second one.
    result = _vastai_run(["show", "instances", "--raw"], config, capture_output=True, check=True)
    instances = json.loads(result.stdout)

    for inst in instances:
        if inst.get("label") != VASTAI_LABEL:
            continue
        # cur_state is what's actually populated for an already-settled
        # instance on this vastai CLI version -- actual_status/status come
        # back null once an instance isn't fresh from creation, even
        # though it's genuinely running (confirmed empirically).
        status = inst.get("cur_state") or inst.get("actual_status") or inst.get("status")
        if status in ("running", "loading"):
            instance_id = str(inst["id"])
            info(f"Reusing existing vast.ai instance {instance_id} (status={status})", "♻️ ")
            return instance_id

    return None


def create_vastai_instance(offer: dict, config: dict) -> str:
    image = config.get("vast_image", VASTAI_DEFAULT_IMAGE)
    disk_gb = config.get("vast_disk_gb", VASTAI_DEFAULT_DISK_GB)

    info(f"Renting vast.ai offer {offer['id']} (image={image}, disk={disk_gb}GB)", "🚀")
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
    success(f"Created instance {instance_id}", "🆕")
    return instance_id


def wait_for_vastai_instance(instance_id: str, config: dict, timeout_s: int = 900) -> None:
    info(f"Waiting for instance {instance_id} to reach 'running'", "⏳")
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        # `show instance <id>` (singular) returns actual_status/status/
        # cur_state all null on this vastai CLI version -- `show instances`
        # (plural) reliably populates cur_state instead, so poll that and
        # filter client-side (same approach as find_existing_vastai_instance).
        result = _vastai_run(["show", "instances", "--raw"], config, capture_output=True, check=True)
        instances = json.loads(result.stdout)
        instance_info = next((inst for inst in instances if str(inst.get("id")) == str(instance_id)), None)
        status = None
        if instance_info is not None:
            status = instance_info.get("cur_state") or instance_info.get("actual_status") or instance_info.get("status")

        if status == "running":
            success(f"Instance {instance_id} is running", "🟢")
            return

        info(f"Instance {instance_id} status: {status}, waiting...", "⏳")
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
    # "running" status can land before sshd actually accepts connections --
    # probe with a real SSH attempt instead of trusting the status field alone.
    ssh_args = ["ssh", "-o", "ConnectTimeout=10", *_ssh_base_args(config), f"{config['user']}@{config['host']}", "true"]
    using_proxy = _proxy_command(config) is not None
    info(f"Waiting for SSH on {config['host']}:{config['port']} (proxy={'on' if using_proxy else 'off'})", "📡")
    console.print(f"[dim]   probe: {' '.join(shlex.quote(a) for a in ssh_args)}[/]")

    deadline = time.time() + timeout_s
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        result = subprocess.run(ssh_args, capture_output=True, text=True)

        if result.returncode == 0:
            success(f"SSH is ready on {config['host']}:{config['port']} (attempt {attempt})")
            return

        # Printed every attempt -- this is the real diagnostic, not a silent retry.
        warn(f"Attempt {attempt} failed (rc={result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
        time.sleep(interval_s)

    raise TimeoutError(f"SSH never became ready on {config['host']}:{config['port']} within {timeout_s}s")


def destroy_vastai_instance(instance_id: str, config: dict) -> None:
    warn(f"Destroying vast.ai instance {instance_id}", "💥")
    # No confirmed --yes flag to skip the interactive "Are you sure?" prompt --
    # answer over stdin instead.
    _vastai_run(["destroy", "instance", instance_id], config, input="y\n", check=False)


# ============================================================
# Setup (workspace/{ckpts,plans,input}) -- one branch determines all three:
# the checkpoint, its matching plans/dataset.json (can legitimately differ
# from nnUNet_preprocessed's current copy, e.g. a class added mid-project,
# with no snapshot saved alongside the checkpoint to catch that), and a
# small local input sample. Every branch worktree symlinks dirs/data_storage
# at the same shared tree, keyed by branch name, so one name covers all three.
# ============================================================
if "EXP_STORAGE_BASE" not in os.environ:
    raise EnvironmentError("EXP_STORAGE_BASE is not set -- source envs/workspace/platforms/<platform>/main.sh first.")
NNUNET_DATA_DIR = Path(os.environ["EXP_STORAGE_BASE"]) / "data" / "nnUNet"
NNUNET_RESULTS_DIR = NNUNET_DATA_DIR / "nnUNet_results"
NNUNET_PREPROCESSED_DIR = NNUNET_DATA_DIR / "nnUNet_preprocessed"
REFERENCE_DATA_DIR = NNUNET_DATA_DIR / "MVAA_nnUNET" / "reference_data"
CKPTS_DIR = WORKSPACE_DIR / "ckpts"
PLANS_DIR = WORKSPACE_DIR / "plans"
INPUT_DIR = WORKSPACE_DIR / "input"

CKPT_PLANS_IDENTIFIER = "nnUNetPlans"
CKPT_FOLD = "all"
SAMPLE_VAL_CASES = 3

# task -> (dataset_id, configuration, prefix), same across every branch.
CKPT_TASKS = {
    "ct": ("Dataset001_MVAA_CT_SSL", "3d_fullres", "t1_ct"),
    "tee": ("Dataset002_MVAA_TEE_SSL", "3d_fullres", "t2_tee"),
    "video": ("Dataset003_MVAA_VIDEO_SSL", "2d", "t3_vid"),
}


def list_available_branches() -> list[str]:
    return sorted(p.name for p in NNUNET_RESULTS_DIR.iterdir() if p.is_dir())


def list_git_branches() -> list[str]:
    # Worktrees share one .git, so this lists every branch across all of
    # them (main, ssl, ssl_pretrain, ...), not just this worktree's own --
    # a branch can exist here before it has any staged nnUNet_results.
    result = subprocess.run(
        ["git", "-C", str(INFERENCE_DIR), "branch", "--list", "--format=%(refname:short)"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _resolve_ckpt_file(checkpoint_dir: Path, version: str) -> Path:
    if version == "last":
        path = checkpoint_dir / "last.ckpt"
        if not path.is_file():
            raise FileNotFoundError(f"No last.ckpt in {checkpoint_dir}")
        return path

    if version == "swa":
        path = checkpoint_dir / "swa.ckpt"
        if not path.is_file():
            raise FileNotFoundError(f"No swa.ckpt in {checkpoint_dir} (branch may not train with SWA)")
        return path

    # best: newest by mtime, matching <branch>/utils.py's resolve_prediction_ckpt.
    best_ckpts = sorted(checkpoint_dir.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not best_ckpts:
        raise FileNotFoundError(f"No best-*.ckpt in {checkpoint_dir}")
    return best_ckpts[0]


def _stage_plans(dataset_id: str) -> None:
    src_dir = NNUNET_PREPROCESSED_DIR / dataset_id
    dest_dir = PLANS_DIR / dataset_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    for name in ("nnUNetPlans.json", "dataset.json"):
        src = src_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing {src} -- can't stage plans for {dataset_id}")
        dest = dest_dir / name
        info(f"[{dataset_id}] {src} -> {dest}", "📐")
        shutil.copy2(src, dest)


def _write_test_cases_manifest(dest_dir: Path, cases: list[dict]) -> None:
    # Official contract: {"cases": [{"case_id": ..., "image": "relative/path"}, ...]}
    # "image" is relative to /input itself (dest_dir's parent), not to
    # dest_dir -- confirmed against a real failed hidden-test run whose
    # error showed a doubled .../t1_ct/t1_ct/... path when the runtime
    # code (workspace/datamodule/inference_dataset.py, video_source.py)
    # joined "image" onto dest_dir directly. Matching that convention here
    # too, so a local docker.sh run exercises the same path resolution the
    # real hidden test set does, instead of a fixture that happens to dodge
    # this exact bug class.
    manifest_path = dest_dir / "test_cases.json"
    info(f"[{dest_dir.name}] writing {manifest_path} ({len(cases)} case(s))", "📝")
    payload = {"cases": sorted(cases, key=lambda c: c["case_id"])}
    manifest_path.write_text(json.dumps(payload, indent=2))


def _stage_sample_volumes(task_dir: str, prefix: str) -> None:
    # Copied under their own original filename -- discover_cases (see
    # workspace/datamodule/inference_dataset.py) reads test_cases.json for
    # the case_id -> image mapping, no renaming needed.
    src_dir = REFERENCE_DATA_DIR / task_dir / "val" / "images"
    cases = sorted(src_dir.glob("*.nii.gz"))[:SAMPLE_VAL_CASES]
    if not cases:
        raise FileNotFoundError(f"No validation cases found in {src_dir}")

    dest_dir = INPUT_DIR / prefix
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    for case in cases:
        dest = dest_dir / case.name
        info(f"[{prefix}] {case} -> {dest}", "🧪")
        shutil.copy2(case, dest)
        manifest_cases.append({"case_id": case.name.removesuffix(".nii.gz"), "image": f"{prefix}/{case.name}"})

    _write_test_cases_manifest(dest_dir, manifest_cases)


def _stage_sample_video() -> None:
    src_dir = REFERENCE_DATA_DIR / "t3_vid" / "val" / "images"
    recordings = sorted(p for p in src_dir.iterdir() if p.is_dir())
    if not recordings:
        raise FileNotFoundError(f"No validation recordings found in {src_dir}")

    recording = recordings[0]
    frames = sorted(recording.glob("*.png"))[:SAMPLE_VAL_CASES]
    if not frames:
        raise FileNotFoundError(f"No frames found in {recording}")

    dest_dir = INPUT_DIR / "t3_vid" / recording.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    for frame in frames:
        dest = dest_dir / frame.name
        info(f"[t3_vid] {frame} -> {dest}", "🧪")
        shutil.copy2(frame, dest)
        stem = frame.stem
        case_id = stem if stem.startswith(recording.name) else f"{recording.name}_{stem}"
        manifest_cases.append({"case_id": case_id, "image": f"t3_vid/{recording.name}/{frame.name}"})

    _write_test_cases_manifest(INPUT_DIR / "t3_vid", manifest_cases)


def setup_action(branch: str, version: str = "best") -> None:
    branch_dir = NNUNET_RESULTS_DIR / branch
    if not branch_dir.is_dir():
        available = ", ".join(list_available_branches())
        raise FileNotFoundError(f"Unknown branch '{branch}' under {NNUNET_RESULTS_DIR}. Available: {available}")

    for task, (dataset_id, configuration, prefix) in CKPT_TASKS.items():
        checkpoint_dir = (
            branch_dir / dataset_id / f"{CKPT_PLANS_IDENTIFIER}__{configuration}" / f"fold_{CKPT_FOLD}" / "checkpoints"
        )
        ckpt_path = _resolve_ckpt_file(checkpoint_dir, version)

        dest_dir = CKPTS_DIR / prefix
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / "model.ckpt"

        info(f"[{task}] {ckpt_path} -> {dest_path}", "💾")
        shutil.copy2(ckpt_path, dest_path)

        _stage_plans(dataset_id)

    _stage_sample_volumes("t1_ct", "t1_ct")
    _stage_sample_volumes("t2_tee", "t2_tee")
    _stage_sample_video()

    success(f"'{branch}' ({version}) staged: workspace/{{ckpts,plans,input}}/ ready", "🧪")


# ============================================================
# The 6 actions, callable directly (menu) or via typer (CLI)
# ============================================================
def _resolve_configured_vast_instance(config: dict) -> str | None:
    """
    If config.json already names an instance (config["vast"]["instance_id"]),
    check whether it's still actually running and reuse its id instead of
    searching/creating -- avoids renting a second instance just because a
    previous run's id/port rotated out from under .submission_state.json.
    Host/port are always re-fetched fresh via vastai_ssh_target() by the
    caller, since those can change even for the same still-alive instance.
    """
    instance_id = (config.get("vast") or {}).get("instance_id")
    if instance_id is None:
        return None

    result = _vastai_run(["show", "instances", "--raw"], config, capture_output=True, check=True)
    instances = json.loads(result.stdout)
    instance_info = next((inst for inst in instances if str(inst.get("id")) == str(instance_id)), None)

    if instance_info is None:
        info(f"config.json's recorded instance {instance_id} no longer exists -- searching/creating instead", "ℹ️ ")
        return None

    status = instance_info.get("cur_state") or instance_info.get("actual_status") or instance_info.get("status")
    if status not in ("running", "loading"):
        info(
            f"config.json's recorded instance {instance_id} is not running (status={status}) "
            "-- searching/creating instead",
            "ℹ️ ",
        )
        return None

    info(f"Reusing config.json's recorded instance {instance_id}", "♻️ ")
    return str(instance_id)


def create_vast_action(config: dict) -> dict:
    if not config.get("vast_api_key"):
        raise RuntimeError("config.json has no vast_api_key -- nothing to rent (this is for a fixed machine).")

    instance_id = (
        _resolve_configured_vast_instance(config)
        or find_existing_vastai_instance(config)
    )
    if instance_id is None:
        offer = find_vastai_offer(config)
        instance_id = create_vastai_instance(offer, config)

    wait_for_vastai_instance(instance_id, config)
    ssh_target = vastai_ssh_target(instance_id, config)
    conn = {**config, **ssh_target}
    wait_for_ssh_ready(conn)

    vast = {"instance_id": instance_id, **ssh_target}
    save_state(vast=vast)
    save_config_vast(vast)
    success(f"Ready: {ssh_target['user']}@{ssh_target['host']}:{ssh_target['port']} (instance {instance_id})", "🎉")
    return conn


def delete_vast_action(config: dict) -> None:
    state = load_state()
    vast = config.get("vast") or state.get("vast")
    instance_id = vast["instance_id"] if vast else find_existing_vastai_instance(config)

    if instance_id is None:
        info("No vast.ai instance on record -- nothing to delete.", "ℹ️ ")
        return

    destroy_vastai_instance(instance_id, config)
    if state.get("vast"):
        state.pop("vast", None)
        STATE_PATH.write_text(json.dumps(state, indent=2))
    if config.get("vast"):
        save_config_vast(None)
    success(f"Destroyed instance {instance_id}", "🗑️ ")


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
    success(f"{remote_path} is on {conn['host']}, ready for trigger-build", "🚀")
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
    answer = console.input(f"[bold cyan]?[/] {prompt} [dim]\\[{suffix}][/] ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def _prompt_branch() -> str:
    # Every local git branch is shown (not just NNUNET_RESULTS_DIR's dirs),
    # so a branch that exists but hasn't trained/staged results yet is
    # still visible -- annotated so it's clear which ones setup can
    # actually stage right now.
    git_branches = list_git_branches()
    staged = set(list_available_branches())

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1, 0, 1), expand=False)
    table.add_column("branch", style="bold cyan")
    table.add_column("status")
    for b in git_branches:
        status = "[green]results ready[/]" if b in staged else "[dim]no nnUNet_results yet[/]"
        table.add_row(b, status)
    console.print(Panel(table, title="[bold]Local git branches[/]", border_style="cyan", box=box.ROUNDED))

    return console.input("[bold cyan]?[/] Branch name: ").strip()


def interactive_menu() -> None:
    menu = [
        ("1", "🚀", "Create vast.ai instance", lambda: create_vast_action(load_machine_config())),
        ("2", "💥", "Delete vast.ai instance", lambda: delete_vast_action(load_machine_config())),
        (
            "3",
            "🧪",
            "Setup (stage a branch's checkpoints, plans, and sample input)",
            lambda: setup_action(
                _prompt_branch(),
                console.input("[bold cyan]?[/] Version [dim]\\[best/last/swa, default best][/]: ").strip() or "best",
            ),
        ),
        (
            "4",
            "📦",
            "Push docker (build zips + rsync archive to remote)",
            lambda: push_docker_action(
                load_machine_config(), clean_zip=_ask_yes_no("Clean local zip/ first?")
            ),
        ),
        (
            "5",
            "▶️ ",
            "Trigger build (run workspace/docker.sh on remote)",
            lambda: trigger_build_action(
                load_machine_config(), push=_ask_yes_no("Also push image to Docker Hub?")
            ),
        ),
        ("6", "📥", "Pull output (fetch remote output/ as a local zip)", lambda: pull_output_action(load_machine_config())),
    ]

    while True:
        table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1, 0, 1), expand=False)
        table.add_column("key", style="bold cyan", justify="right", no_wrap=True)
        table.add_column("emoji", no_wrap=True)
        table.add_column("label")
        for key, emoji, label, _ in menu:
            table.add_row(key, emoji, label)
        table.add_row("0", "🚪", "Exit")

        console.print()
        console.print(Panel(table, title="[bold]🛰️  MVAA Submission Tool[/]", border_style="cyan", box=box.ROUNDED))

        choice = console.input("[bold cyan]➜ Select option:[/] ").strip().lower()
        if choice in ("0", "q", "quit", "exit"):
            return

        match = next((m for m in menu if m[0] == choice), None)
        if match is None:
            warn("Invalid option, try again.")
            continue

        try:
            match[3]()
        except Exception as e:
            console.print(Panel(f"[bold red]{e}[/]", title="❌ Error", border_style="red", box=box.ROUNDED))


# ============================================================
# Typer CLI
# ============================================================
app = typer.Typer(add_completion=False, help="🛰️  MVAA submission build/push/test tool.")


def _guarded(fn, *args, **kwargs):
    # Routes failures through the same error panel as the interactive menu,
    # instead of a raw traceback.
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        console.print(Panel(f"[bold red]{e}[/]", title="❌ Error", border_style="red", box=box.ROUNDED))
        raise typer.Exit(code=1) from None


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        interactive_menu()


@app.command("create-vast")
def create_vast_cmd() -> None:
    """🚀 1. Rent (or reuse) a vast.ai GPU instance and wait until SSH is reachable."""
    _guarded(create_vast_action, load_machine_config())


@app.command("delete-vast")
def delete_vast_cmd() -> None:
    """💥 2. Destroy the vast.ai instance currently on record."""
    _guarded(delete_vast_action, load_machine_config())


@app.command("setup")
def setup_cmd(
    branch: str = typer.Argument(None, help="nnUNet_results subfolder name -- omit to pick from a live list."),
    version: str = typer.Option("best", "--version", help="Which checkpoint to use: best, last, or swa."),
) -> None:
    """🧪 3. Stage a branch's checkpoints, plans, and sample input (workspace/{ckpts,plans,input}/)."""
    _guarded(setup_action, branch or _prompt_branch(), version)


@app.command("push-docker")
def push_docker_cmd(
    clean_zip: bool = typer.Option(False, "--clean-zip", help="Delete old zip/ contents first."),
) -> None:
    """📦 4. Build submission.zip + the docker build archive, and rsync it to the remote machine."""
    _guarded(push_docker_action, load_machine_config(), clean_zip=clean_zip)


@app.command("trigger-build")
def trigger_build_cmd(
    push: bool = typer.Option(False, "--push", help="Also push the built image to Docker Hub."),
) -> None:
    """▶️  5. Run workspace/docker.sh on the remote machine against the last pushed archive."""
    _guarded(trigger_build_action, load_machine_config(), push=push)


@app.command("pull-output")
def pull_output_cmd() -> None:
    """📥 6. Zip the remote output/ dir and pull it back to zip/output_docker_<timestamp>.zip."""
    _guarded(pull_output_action, load_machine_config())


if __name__ == "__main__":
    # `--tunnel` is this script re-invoking itself as an SSH ProxyCommand --
    # intercept before typer ever sees argv (not one of the 6 subcommands).
    if len(sys.argv) > 1 and sys.argv[1] == "--tunnel":
        _, target_host, target_port, proxy_host, proxy_port = sys.argv[1:6]
        run_tunnel(target_host, target_port, proxy_host, int(proxy_port))
    else:
        app()
