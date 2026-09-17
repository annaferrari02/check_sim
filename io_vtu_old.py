"""
Reading SimVascular / svSolver output.

Results here are VOLUME .vtu files that do NOT carry GlobalNodeID, so we map
each mesh-surface cap node to its coincident volume node by COORDINATES using a
cKDTree. Cap nodes are the same physical nodes as the volume boundary nodes, so
the match distance is ~0. The point ordering is identical across all
result_*.vtu of the same mesh/procs run, so the tree and the resulting indices
are built ONCE (from the first timestep) and reused for every step.

Pressure per face  : area-weighted mean over the cap triangles.
Flow per face      : Q = sum_tri (v_bar . areaVector), areaVector magnitude =
                     triangle area, direction = outward normal (SimVascular
                     winding). Inlet flow is NEGATIVE, as in setup_bcs.py.
"""

import os
import re
import glob
import time
import warnings
import numpy as np

try:
    import pyvista as pv
except ImportError as e:  # pragma: no cover
    raise ImportError("pyvista is required (pip install pyvista).") from e

try:
    from scipy.spatial import cKDTree
except ImportError as e:  # pragma: no cover
    raise ImportError("scipy is required (pip install scipy).") from e


# match distance above this (in mesh units, cm) is suspicious / fatal
_MATCH_WARN = 1e-6
_MATCH_FATAL = 1e-2
_READ_RETRIES = 4
_READ_RETRY_DELAY = 1.0


def _read_result(path):
    """Read a result file, retrying if the solver is still writing it."""
    last_error = None
    for attempt in range(_READ_RETRIES + 1):
        try:
            size_before = os.path.getsize(path)
            result = pv.read(path)
            size_after = os.path.getsize(path)
            if size_before == size_after:
                return result
            last_error = RuntimeError("file size changed while reading")
        except Exception as exc:
            last_error = exc
        if attempt < _READ_RETRIES:
            time.sleep(_READ_RETRY_DELAY)
    raise RuntimeError(
        f"Could not read stable result file after {_READ_RETRIES + 1} attempts: {path}"
    ) from last_error


class Face:
    """Geometry + result-index mapping for one named surface cap."""

    def __init__(self, name, kind, points, tris):
        self.name = name          # filename stem
        self.kind = kind          # 'inlet' | 'wall' | 'outlet'
        self.points = points      # (n_pts, 3) cap node coordinates
        self.tris = tris          # (n_tri, 3) local point indices
        a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
        self.area_vec = 0.5 * np.cross(b - a, c - a)     # (n_tri,3) outward
        self.tri_area = np.linalg.norm(self.area_vec, axis=1)
        self.area = float(self.tri_area.sum())
        self.radius = float(np.sqrt(self.area / np.pi))  # equiv. circular r
        self.ridx = None          # (n_pts,) indices into the result arrays


def list_patients(cfg):
    return sorted(glob.glob(os.path.join(cfg.patients_root, cfg.patient_glob)))


def _classify(stem, cfg):
    s = stem.lower()
    if s in [x.lower() for x in cfg.inlet_names]:
        return "inlet"
    if any(s.startswith(p.lower()) for p in cfg.wall_prefixes):
        return "wall"
    return "outlet"


def read_faces(patient_dir, cfg):
    d = os.path.join(patient_dir, cfg.mesh_surfaces_subdir)
    paths = sorted(glob.glob(os.path.join(d, "*.vtp")))
    if not paths:
        raise FileNotFoundError(f"No .vtp faces in {d}")
    faces = {}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        m = pv.read(path)
        conn = m.faces.reshape(-1, 4)          # assumes pure-triangle caps
        if not np.all(conn[:, 0] == 3):
            raise ValueError(f"{path} has non-triangular cells")
        tris = conn[:, 1:]
        faces[stem] = Face(stem, _classify(stem, cfg), np.asarray(m.points), tris)
    return faces


def discover_steps(patient_dir, cfg):
    """Return sorted list of (step_number, path)."""
    d = os.path.join(patient_dir, cfg.results_subdir)
    out = []
    for p in glob.glob(os.path.join(d, cfg.results_glob)):
        mobj = re.search(r"(\d+)(?=\.\w+$)", os.path.basename(p))
        if mobj:
            out.append((int(mobj.group(1)), p))
    out.sort()
    return out


def _build_result_map(first_path, faces, cfg):
    """Map every cap node to its nearest volume node (coincident) by coords."""
    res = _read_result(first_path)
    vol_pts = np.asarray(res.points)
    tree = cKDTree(vol_pts)
    max_d = 0.0
    for f in faces.values():
        d, idx = tree.query(f.points, k=1)
        f.ridx = idx.astype(np.int64)
        max_d = max(max_d, float(d.max()))
    if max_d > _MATCH_FATAL:
        raise RuntimeError(
            f"Max cap->volume match distance = {max_d:.3e} (mesh units). "
            "Surface and volume meshes don't coincide -- check that "
            "mesh-surfaces and the result .vtu are the same case / same units."
        )
    if max_d > _MATCH_WARN:
        warnings.warn(f"Max cap->volume match distance = {max_d:.3e} "
                      "(expected ~1e-9). Proceeding, but verify the mesh.")
    return max_d


def extract_timeseries(patient_dir, faces, cfg):
    """
    Returns
    -------
    times    : (n_t,) float   physical time of each saved step
    P        : dict name -> (n_t,)  area-weighted mean pressure [dyn/cm^2]
    Q        : dict name -> (n_t,)  flux (v.A) [cm^3/s], signed (inlet < 0)
    match_d  : float          max cap->volume node distance (diagnostic)
    """
    steps = discover_steps(patient_dir, cfg)
    if not steps:
        raise FileNotFoundError(
            f"No results matching '{cfg.results_glob}' in "
            f"{os.path.join(patient_dir, cfg.results_subdir)}"
        )
    match_d = _build_result_map(steps[0][1], faces, cfg)

    times = np.array([s * cfg.dt for s, _ in steps])
    P = {name: np.zeros(len(steps)) for name in faces}
    Q = {name: np.zeros(len(steps)) for name in faces}

    for k, (_step, path) in enumerate(steps):
        if k == 0 or (k + 1) % 10 == 0 or k == len(steps) - 1:
            print(f"      reading result {k + 1}/{len(steps)}: "
                  f"{os.path.basename(path)}", flush=True)
        res = _read_result(path)
        p = np.asarray(res.point_data[cfg.pressure_array])
        v = np.asarray(res.point_data[cfg.velocity_array])
        for f in faces.values():
            pf = p[f.ridx]
            vf = v[f.ridx]
            p_tri = pf[f.tris].mean(axis=1)                 # (n_tri,)
            P[f.name][k] = (p_tri * f.tri_area).sum() / f.area
            v_tri = vf[f.tris].mean(axis=1)                 # (n_tri,3)
            Q[f.name][k] = float((v_tri * f.area_vec).sum())

    return times, P, Q, match_d
