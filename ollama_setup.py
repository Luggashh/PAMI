"""
Bring up Ollama without root privileges and fan it out across the GPUs.

Responsibilities:
    1. Locate an `ollama` binary, or install one into the user's home directory
       (no sudo: the official tarball is unpacked under ~/.local/ollama).
    2. Launch one server per GPU, each pinned via CUDA_VISIBLE_DEVICES and
       bound to its own port.
    3. Pull the model once (all servers share the same models directory).
    4. Shut the servers down again on exit.
"""

import atexit
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

from config import (
    MODEL_NAME,
    NUM_PARALLEL_PER_SERVER,
    NUM_SERVERS,
    OLLAMA_BASE_PORT,
    OLLAMA_DOWNLOAD_URL,
    OLLAMA_HOST,
    OLLAMA_INSTALL_DIR,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODELS_DIR,
)

_PROCESSES: list[subprocess.Popen] = []


# ── Binary discovery / installation ──────────────────────────────


def _local_binary() -> str | None:
    """Path to a previously home-installed ollama binary, if present."""
    candidate = os.path.join(OLLAMA_INSTALL_DIR, "bin", "ollama")
    return candidate if os.path.isfile(candidate) else None


def _download(url: str, dest: str) -> None:
    """Download `url` to `dest` with a coarse progress indicator."""
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        last_pct = -1
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if total:
                pct = int(read * 100 / total)
                if pct != last_pct and pct % 5 == 0:
                    print(f"   {pct:3d}%  ({read / 1e9:.2f} / {total / 1e9:.2f} GB)")
                    last_pct = pct
            elif read % (256 << 20) < (1 << 20):
                print(f"   {read / 1e9:.2f} GB")
    print("   download complete.")


def install_ollama_userspace() -> str:
    """
    Install Ollama into the user's home directory. Requires no root access.

    Returns the path to the installed binary.
    """
    print(f"📦 Installing Ollama into {OLLAMA_INSTALL_DIR} (no sudo required)...")
    os.makedirs(OLLAMA_INSTALL_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "ollama-linux-amd64.tgz")
        print(f"   Downloading {OLLAMA_DOWNLOAD_URL}")
        _download(OLLAMA_DOWNLOAD_URL, archive)

        print("   Extracting (this includes the bundled CUDA runtime)...")
        with tarfile.open(archive, "r:gz") as tar:
            try:
                tar.extractall(OLLAMA_INSTALL_DIR, filter="tar")
            except TypeError:  # Python < 3.11.4 has no extraction filters
                tar.extractall(OLLAMA_INSTALL_DIR)

    binary = os.path.join(OLLAMA_INSTALL_DIR, "bin", "ollama")
    if not os.path.isfile(binary):
        raise RuntimeError(
            f"Ollama tarball extracted but no binary at {binary}. "
            f"Contents: {os.listdir(OLLAMA_INSTALL_DIR)}"
        )
    os.chmod(binary, 0o755)
    print(f"✅ Ollama installed at {binary}\n")
    return binary


def resolve_ollama_binary() -> str:
    """Find an ollama binary, installing one into $HOME if necessary."""
    on_path = shutil.which("ollama")
    if on_path:
        print(f"✅ Using existing Ollama binary: {on_path}")
        return on_path

    local = _local_binary()
    if local:
        print(f"✅ Using home-installed Ollama binary: {local}")
        return local

    return install_ollama_userspace()


# ── GPU detection ────────────────────────────────────────────────


def detect_gpus() -> list[str]:
    """
    Device IDs to pin servers to, as they must be written into
    CUDA_VISIBLE_DEVICES.

    If the job scheduler already restricted us to a subset of cards, those
    exact IDs are reused — re-indexing them from 0 would silently grab GPUs
    that belong to somebody else.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [d.strip() for d in visible.split(",") if d.strip()]

    try:
        out = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return [str(i) for i, line in enumerate(out.stdout.splitlines()) if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


# ── Server lifecycle ─────────────────────────────────────────────


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((OLLAMA_HOST, port)) != 0


def _pick_ports(count: int) -> list[int]:
    """Pick `count` free ports starting from OLLAMA_BASE_PORT."""
    ports: list[int] = []
    port = OLLAMA_BASE_PORT
    while len(ports) < count and port < OLLAMA_BASE_PORT + 200:
        if _port_free(port):
            ports.append(port)
        else:
            print(f"   Port {port} already in use — skipping.")
        port += 1
    if len(ports) < count:
        raise RuntimeError("Could not find enough free ports for the Ollama servers.")
    return ports


def _server_env(binary: str, port: int, gpu_id: str | None) -> dict:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"{OLLAMA_HOST}:{port}"
    env["OLLAMA_MODELS"] = OLLAMA_MODELS_DIR
    env["OLLAMA_KEEP_ALIVE"] = OLLAMA_KEEP_ALIVE
    env["OLLAMA_NUM_PARALLEL"] = str(NUM_PARALLEL_PER_SERVER)
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    # The home-installed tarball ships its CUDA runtime next to the binary.
    lib_dir = os.path.join(os.path.dirname(os.path.dirname(binary)), "lib", "ollama")
    if os.path.isdir(lib_dir):
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    return env


def _wait_until_ready(base_url: str, timeout: float = 180.0) -> bool:
    """Poll /api/tags until the server answers, instead of guessing with sleep."""
    import requests  # local import: keeps this module importable without deps

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/api/tags", timeout=5).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def shutdown_servers() -> None:
    """Terminate every server this process started."""
    for proc in _PROCESSES:
        if proc.poll() is None:
            proc.terminate()
    for proc in _PROCESSES:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    _PROCESSES.clear()


def start_servers(log_dir: str) -> list[str]:
    """
    Install (if needed) and launch the Ollama servers, then pull the model.

    Returns the list of base URLs, one per server.
    """
    binary = resolve_ollama_binary()
    os.makedirs(OLLAMA_MODELS_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    gpus = detect_gpus()
    n_servers = NUM_SERVERS or max(len(gpus), 1)
    if gpus:
        print(f"🖥  Detected {len(gpus)} GPU(s) {gpus} → launching {n_servers} Ollama server(s).")
    else:
        print("⚠️  No NVIDIA GPU detected — running on CPU with a single server.")

    ports = _pick_ports(n_servers)
    base_urls: list[str] = []

    for i, port in enumerate(ports):
        # More servers than GPUs simply wraps around and shares cards.
        gpu_id = gpus[i % len(gpus)] if gpus else None
        log_path = os.path.join(log_dir, f"ollama_server{i}_gpu{gpu_id or 'cpu'}.log")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            [binary, "serve"],
            env=_server_env(binary, port, gpu_id),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        _PROCESSES.append(proc)
        base_urls.append(f"http://{OLLAMA_HOST}:{port}")
        print(f"   ▶ server {i} → GPU {gpu_id}, port {port}, log {log_path}")

    atexit.register(shutdown_servers)

    print("\n⏳ Waiting for servers to accept connections...")
    for i, url in enumerate(base_urls):
        if not _wait_until_ready(url):
            print(f"❌ Server {i} at {url} never became ready. Check {log_dir}.")
            shutdown_servers()
            sys.exit(1)
        print(f"   ✅ {url}")

    _pull_model(binary, base_urls[0])
    _warm_up(base_urls)
    return base_urls


def _pull_model(binary: str, base_url: str) -> None:
    """Pull the model once — all servers share OLLAMA_MODELS_DIR."""
    print(f"\n📥 Ensuring model '{MODEL_NAME}' is available...")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = base_url
    env["OLLAMA_MODELS"] = OLLAMA_MODELS_DIR
    result = subprocess.run([binary, "pull", MODEL_NAME], env=env)
    if result.returncode != 0:
        raise RuntimeError(f"`ollama pull {MODEL_NAME}` failed (exit {result.returncode}).")
    print("✅ Model ready.")


def _warm_up(base_urls: list[str]) -> None:
    """Force each server to load the model onto its GPU before timing starts."""
    import requests

    print("\n🔥 Warming up each server (loading the model onto its GPU)...")
    for url in base_urls:
        try:
            requests.post(
                f"{url}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": "1+1=",
                    "stream": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                    "options": {"num_predict": 1},
                },
                timeout=600,
            ).raise_for_status()
            print(f"   ✅ {url}")
        except requests.exceptions.RequestException as exc:
            print(f"   ⚠️  {url} warm-up failed: {exc}")
    print()
