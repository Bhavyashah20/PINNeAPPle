"""Run CalculiX (``ccx``): native binary if on PATH, else the bundled Docker image.

The image is built from ``Dockerfile`` next to this module (Debian's official ``calculix-ccx``
package on the official Debian image, pulled from the AWS mirror of the Docker library to avoid Docker Hub's
anonymous rate limit), tag ``pinneapple/calculix:local``: no third-party container is pulled.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass

IMAGE = "pinneapple/calculix:local"
_HERE = os.path.dirname(os.path.abspath(__file__))


def _docker_env():
    env = dict(os.environ)
    helper = "/Applications/Docker.app/Contents/Resources/bin"  # Docker Desktop credential helper (macOS)
    if os.path.isdir(helper) and helper not in env.get("PATH", ""):
        env["PATH"] = helper + os.pathsep + env.get("PATH", "")
    return env


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, env=_docker_env(), timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


#: The same official Debian image from three registries, tried in turn: a registry that rate limits anonymous pulls
#: (public.ecr.aws answers "429 Too Many Requests / Data limit exceeded" on busy shared CI runners, Docker Hub has
#: its own limit) must not make the build fail while another one serves the image. ``PINNEAPPLE_CALCULIX_BASE``
#: puts a base of your choice first.
BASE_IMAGES = ("public.ecr.aws/docker/library/debian:bookworm-slim",
               "mirror.gcr.io/library/debian:bookworm-slim",
               "debian:bookworm-slim")


def _bases() -> list[str]:
    first = os.environ.get("PINNEAPPLE_CALCULIX_BASE")
    return ([first] if first else []) + [b for b in BASE_IMAGES if b != first]


def ensure_image(image: str = IMAGE, attempts: int = 3) -> str:
    """Build the image if it is missing. The build downloads the base image and the Debian package, so a failed
    attempt is retried, each time from the next registry of :data:`BASE_IMAGES` (registry or mirror hiccups and
    rate limits); the last failure is raised with Docker's own output."""
    env = _docker_env()
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True, env=env).returncode == 0:
        return image
    bases = _bases()
    for k in range(max(1, attempts)):
        base = bases[k % len(bases)]
        p = subprocess.run(["docker", "build", "-q", "--build-arg", f"BASE={base}", "-t", image, _HERE],
                           capture_output=True, text=True, env=env)
        if p.returncode == 0:
            return image
        if k + 1 < attempts:
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"docker build of {image} failed {attempts} time(s) (bases tried: "
                       f"{', '.join(bases[:max(1, attempts)])}); last output:\n{(p.stdout + p.stderr)[-4000:]}")


def ccx_backend() -> str | None:
    """'native', 'docker' or None."""
    if shutil.which("ccx"):
        return "native"
    if docker_available():
        return "docker"
    return None


@dataclass
class CcxRun:
    returncode: int
    stdout: str
    frd: str
    dat: str


def run_ccx(workdir: str, job: str, *, threads: int = 1, timeout: float = 3600.0, backend: str | None = None) -> CcxRun:
    """Run ``ccx -i <job>`` in ``workdir`` (which must contain ``<job>.inp``)."""
    backend = backend or ccx_backend()
    env = dict(_docker_env(), OMP_NUM_THREADS=str(threads))
    if backend == "native":
        cmd = ["ccx", "-i", job]
    elif backend == "docker":
        ensure_image()
        cmd = ["docker", "run", "--rm", "-e", f"OMP_NUM_THREADS={threads}", "-v",
               f"{os.path.abspath(workdir)}:/work", "-w", "/work", IMAGE, "-i", job]
    else:
        raise RuntimeError("CalculiX not available: install ccx or Docker")
    p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout, env=env)
    return CcxRun(p.returncode, p.stdout + p.stderr, os.path.join(workdir, job + ".frd"),
                  os.path.join(workdir, job + ".dat"))
