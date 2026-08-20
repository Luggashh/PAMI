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
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time

import requests

from config import (
    MODEL_NAME,
    NUM_PARALLEL_PER_SERVER,
    NUM_SERVERS,
    OLLAMA_BASE_PORT,
    OLLAMA_DOWNLOAD_BASE,
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


def _archive_name() -> str:
    """Release asset base name for this machine's architecture."""
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"
    return f"ollama-linux-{arch}"


def resolve_download_url() -> str:
    """
    Pick the release archive to download.

    Ollama moved from `.tgz` to `.tar.zst`; probe both so the pipeline keeps
    working across that change in either direction.
    """
    if OLLAMA_DOWNLOAD_URL:
        return OLLAMA_DOWNLOAD_URL

    base = _archive_name()
    candidates = [
        f"{OLLAMA_DOWNLOAD_BASE}/{base}.tar.zst",
        f"{OLLAMA_DOWNLOAD_BASE}/{base}.tgz",
    ]
    for url in candidates:
        try:
            response = requests.head(url, allow_redirects=True, timeout=30)
            if response.status_code == 200:
                return url
        except requests.exceptions.RequestException:
            continue

    raise RuntimeError(
        "Could not find a downloadable Ollama archive. Tried:\n  "
        + "\n  ".join(candidates)
        + "\nSet PAMI_OLLAMA_URL to a working archive URL to override."
    )


def _download(url: str, dest: str) -> None:
    """Stream `url` to `dest` with a coarse progress indicator."""
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        last_pct = -1
        with open(dest, "wb") as out:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                out.write(chunk)
                read += len(chunk)
                if total:
                    pct = int(read * 100 / total)
                    if pct != last_pct and pct % 5 == 0:
                        print(f"   {pct:3d}%  ({read / 1e9:.2f} / {total / 1e9:.2f} GB)", flush=True)
                        last_pct = pct
                elif read % (256 << 20) < (1 << 20):
                    print(f"   {read / 1e9:.2f} GB", flush=True)
    print("   Download complete.")


def _extract(archive: str, dest_dir: str) -> None:
    """
    Unpack a `.tgz` or `.tar.zst` release archive.

    The official install.sh requires the `zstd` CLI, which needs root to
    install on most distros. Python 3.14+ decompresses zstd natively, and the
    pip-installable `zstandard` module covers older interpreters — so no root
    is needed either way.
    """
    print("   Extracting (this includes the bundled CUDA runtime)...", flush=True)

    def _unpack(fileobj_or_path, mode):
        with tarfile.open(**fileobj_or_path, mode=mode) as tar:
            try:
                tar.extractall(dest_dir, filter="tar")
            except TypeError:  # Python < 3.11.4 has no extraction filters
                tar.extractall(dest_dir)

    if archive.endswith(".tgz") or archive.endswith(".tar.gz"):
        _unpack({"name": archive}, "r:gz")
        return

    # 1. Native zstd support (Python 3.14+, if built against libzstd).
    try:
        _unpack({"name": archive}, "r:zst")
        return
    except (tarfile.CompressionError, ImportError, ValueError):
        pass

    # 2. The `zstandard` pip module.
    try:
        import zstandard

        with open(archive, "rb") as raw:
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            _unpack({"fileobj": reader}, "r|")
        return
    except ImportError:
        pass

    # 3. A `zstd` binary on PATH (conda envs usually ship one).
    zstd_bin = shutil.which("zstd") or shutil.which("unzstd")
    if zstd_bin:
        decompressed = archive[: -len(".zst")]
        subprocess.run([zstd_bin, "-d", "-f", archive, "-o", decompressed], check=True)
        _unpack({"name": decompressed}, "r:")
        os.remove(decompressed)
        return

    raise RuntimeError(
        "Cannot decompress a .tar.zst archive: this Python has no zstd support, "
        "the `zstandard` module is not installed, and no `zstd` binary is on PATH.\n"
        "Fix with:  pip install zstandard"
    )


def install_ollama_userspace() -> str:
    """
    Install Ollama into the user's home directory. Requires no root access.

    Returns the path to the installed binary.
    """
    print(f"📦 Installing Ollama into {OLLAMA_INSTALL_DIR} (no sudo required)...")
    os.makedirs(OLLAMA_INSTALL_DIR, exist_ok=True)

    url = resolve_download_url()

    # Stage the ~1.5 GB download next to the install dir, not in /tmp, which is
    # often small or quota-limited on shared cluster nodes.
    staging = os.path.dirname(os.path.abspath(OLLAMA_INSTALL_DIR))
    os.makedirs(staging, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=staging) as tmp:
        archive = os.path.join(tmp, os.path.basename(url))
        print(f"   Downloading {url}", flush=True)
        _download(url, archive)
        _extract(archive, OLLAMA_INSTALL_DIR)

    binary = os.path.join(OLLAMA_INSTALL_DIR, "bin", "ollama")
    if not os.path.isfile(binary):
        raise RuntimeError(
            f"Ollama archive extracted but no binary at {binary}. "
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
