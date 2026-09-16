"""
Entry point. Run:  python3 main.py
Edit config.py first to point at your data.
"""

import os
import traceback

from config import Config
from cycles import sanity_check_timing
from io_vtu import list_patients, read_faces, extract_timeseries
import checks
import report


def run_patient(patient_dir, cfg):
    faces = read_faces(patient_dir, cfg)
    times, P, Q, match_d = extract_timeseries(patient_dir, faces, cfg)
    print(f"    cap->volume match distance: {match_d:.2e} | {len(times)} steps")
    return {
        "check1": checks.check1_inlet_outlet_pressure(times, P, Q, faces, cfg),
        "check2": checks.check2_pressure_shape(times, P, faces, cfg),
        "check3": checks.check3_outlet_flow_shape(times, Q, faces, cfg),
        "check4": checks.check4_murray(times, Q, faces, cfg),
    }


def main(cfg=None):
    cfg = cfg or Config()
    sanity_check_timing(cfg)

    out_dir = os.path.join(cfg.patients_root, cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    patients = list_patients(cfg)
    if not patients:
        raise SystemExit(f"No patients matching '{cfg.patient_glob}' "
                         f"under {cfg.patients_root}")

    out_md = os.path.join(out_dir, "report.md")
    all_results = {}
    body_sections = []
    for pdir in patients:
        name = os.path.basename(pdir.rstrip("/"))
        print(f"[{name}] ...", flush=True)
        try:
            all_results[name] = run_patient(pdir, cfg)
            print(f"[{name}] done")
        except Exception as e:  # keep going on the other sims
            all_results[name] = {"error": str(e)}
            print(f"[{name}] FAILED: {e}")
            traceback.print_exc()
        # rewrite the single batch report after each sim (crash-safe)
        md = report.render_cohort(all_results, cfg, out_dir)
        with open(out_md, "w", encoding="utf-8") as fh:
            fh.write(md)

    print(f"\nBatch report written to {out_md}")


if __name__ == "__main__":
    main()
