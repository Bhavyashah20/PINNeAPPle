"""DeepMind ``meshgraphnets`` datasets: partial download, TFRecord reader and conversion to tensors.

The datasets of Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks", ICLR 2021
(``cylinder_flow``, ``airfoil``, ``flag_simple``, ...), are published as one ``meta.json`` plus one TFRecord file per
split under https://storage.googleapis.com/dm-meshgraphnets/<dataset>/. Reading the records needs
``pip install tfrecord`` (no TensorFlow).

    >>> from pinneapple_data.datasets.deepmind_mgn import download_prefix, read_trajectories
    >>> download_prefix("cylinder_flow", "valid", Path("data/cylinder_flow"), nbytes=50_000_000)  # doctest: +SKIP
    >>> trajs = read_trajectories(Path("data/cylinder_flow"), "valid", max_traj=4)              # doctest: +SKIP
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from pinneapple_neural.architectures.graphnn.mgn_dynamics import MeshGraph

__all__ = ["DM_URL", "CYLINDER_TYPE_MAP", "download_prefix", "read_trajectories", "cylinder_flow_to_tensors"]

DM_URL = "https://storage.googleapis.com/dm-meshgraphnets"

#: Raw DeepMind node-type ids (0 NORMAL, 4 INFLOW, 5 OUTFLOW, 6 WALL_BOUNDARY) to the four contiguous
#: classes used by PhysicsNeMo's ``vortex_shedding_mgn``: NORMAL=0, INFLOW=1, OUTFLOW=2, WALL=3.
CYLINDER_TYPE_MAP: dict[int, int] = {0: 0, 4: 1, 5: 2, 6: 3}

_TIMEOUT = 60


def _fetch(url: str, dest: Path, nbytes: int | None = None, timeout: float = _TIMEOUT) -> None:
    """Download ``url`` (its first ``nbytes`` bytes when given, via an HTTP Range request) to ``dest``."""
    headers = {"Range": f"bytes=0-{nbytes - 1}"} if nbytes else {}
    req = urllib.request.Request(url, headers=headers)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        if nbytes and getattr(resp, "status", 206) == 200:   # server ignored Range: copy only the prefix
            fh.write(resp.read(nbytes))
        else:
            shutil.copyfileobj(resp, fh)
    os.replace(tmp, dest)


def download_prefix(dataset: str, split: str, dest: Path, nbytes: int) -> Path:
    """Download ``meta.json`` and the first ``nbytes`` of ``<split>.tfrecord`` of a DeepMind dataset into ``dest``.

    TFRecord is a sequential format, so a byte prefix holds whole leading trajectories (the truncated last record is
    dropped by :func:`read_trajectories`). Files already present are not downloaded again. Returns the record path.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    meta = dest / "meta.json"
    if not meta.exists():
        _fetch(f"{DM_URL}/{dataset}/meta.json", meta)
    rec = dest / f"{split}.tfrecord"
    if not rec.exists():
        _fetch(f"{DM_URL}/{dataset}/{split}.tfrecord", rec, nbytes=int(nbytes))
    return rec


def read_trajectories(data_dir: Path, split: str, max_traj: int) -> list[dict[str, np.ndarray]]:
    """Read up to ``max_traj`` trajectories of ``<data_dir>/<split>.tfrecord`` as dicts of numpy arrays.

    Field shapes follow ``meta.json``: static fields (``mesh_pos``, ``cells``, ``node_type``) lose the leading time
    axis, dynamic ones (``velocity``, ``pressure``, ...) are ``(T, N, C)``. A truncated last record (prefix download)
    ends the read.
    """
    from tfrecord.reader import tfrecord_loader

    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    desc = {k: "byte" for k in meta["field_names"]}
    out: list[dict[str, np.ndarray]] = []
    it = tfrecord_loader(str(data_dir / f"{split}.tfrecord"), None, description=desc)
    while len(out) < max_traj:
        try:
            rec = next(it)
        except StopIteration:
            break
        except Exception:  # truncated tail of a prefix download
            break
        traj = {}
        for name, spec in meta["features"].items():
            arr = np.frombuffer(bytes(rec[name]), dtype=np.dtype(spec["dtype"])).copy()
            arr = arr.reshape(spec["shape"][0], -1, spec["shape"][-1])
            traj[name] = arr[0] if spec["type"] == "static" else arr
        out.append(traj)
    return out


def cylinder_flow_to_tensors(
    traj: dict[str, np.ndarray], node_type_map: dict[int, int] = CYLINDER_TYPE_MAP,
) -> tuple[MeshGraph, torch.Tensor, torch.Tensor]:
    """``cylinder_flow`` trajectory -> ``(MeshGraph, velocity (T, N, 2), pressure (T, N, 1))``.

    ``node_type_map`` collapses the raw ids to contiguous classes (default :data:`CYLINDER_TYPE_MAP`).
    """
    import torch

    from pinneapple_neural.architectures.graphnn.mgn_dynamics import MeshGraph

    pos = torch.from_numpy(traj["mesh_pos"]).float()
    cells = torch.from_numpy(traj["cells"]).long()
    raw = traj["node_type"][:, 0]
    nt = torch.from_numpy(np.vectorize(node_type_map.get)(raw)).long()
    g = MeshGraph.from_mesh(pos, cells, nt)
    return g, torch.from_numpy(traj["velocity"]).float(), torch.from_numpy(traj["pressure"]).float()
