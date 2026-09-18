"""
Central configuration. Edit ONLY this file to point the pipeline at your data.

Physical units are CGS (SimVascular / svSolver convention):
  pressure  -> dyn/cm^2   (MAP 119989.8 dyn/cm^2 ~= 90 mmHg)
  velocity  -> cm/s
  length    -> cm
  flow    -> cm^3/s = mL/s   (matches Qref in setup_bcs.py)
"""

from dataclasses import dataclass, field
from typing import Tuple

DYN_PER_MMHG = 1333.22  # 1 mmHg = 1333.22 dyn/cm^2


@dataclass
class Config:
    # --- simulation timing -------------------------------------------------
    T: float = 0.8            # cardiac period [s]
    dt: float = 0.0025        # solver timestep [s]  -> 320 steps / cycle
    n_cycles: int = 3         # number of cycles simulated

    # --- where the patients live ------------------------------------------
    # Each patient folder (pz***) is expected to contain:
    #   <mesh_surfaces_subdir>/*.vtp   face geometry + GlobalNodeID
    #   <results_subdir>/<results_glob>  per-timestep results (VTP or VTU)
    patients_root: str = "E:/database_sim" #da sistemare sulla base di dove stanno i dati
    patient_glob: str = "pz*"
    mesh_surfaces_subdir: str = "mesh-complete/mesh-surfaces"
    results_subdir: str = "72-procs"
    results_glob: str = "result_*.vtu"  
    # --- array names inside the results files ------------------------------
    pressure_array: str = "Pressure"
    velocity_array: str = "Velocity"
    # results have NO GlobalNodeID -> faces are mapped to the volume by
    # nearest-neighbour on coordinates (cKDTree), see io_vtu.py

    # --- face classification by filename stem (case-insensitive) -----------
    # inlet: stem is exactly one of these
    inlet_names: Tuple[str, ...] = ("inlet",)
    # wall: stem STARTS WITH one of these
    wall_prefixes: Tuple[str, ...] = ("wall",)
    # everything else in mesh-surfaces is treated as a terminal outlet

    # --- physics reference -------------------------------------------------
    MAP_dyn_cm2: float = 119989.8   # imposed MAP from setup_bcs.py

    inlet_sys_band_mmHg: tuple = (110.0, 160.0)
    inlet_dia_band_mmHg: tuple = (70.0, 100.0)
    pulse_pressure_band_mmHg: tuple = (40.0, 80.0)
    # 3D-domain resistance considered negligible if |Δp|/MAP below this:
    resistance_frac_MAP_ok: float = 0.05      # 5%
    # convergence: inlet pressure shape considered settled if RMS/PP below:
    shape_rms_over_pp_ok: float = 0.05        # 5%
    # Murray adherence considered good if per-sim RMS relative error below:
    murray_rms_ok: float = 0.10               # 10%

    # --- output ------------------------------------------------------------
    out_dir: str = "report"         # written under patients_root
    figure_dpi: int = 130
