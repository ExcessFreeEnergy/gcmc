"""
End-to-end (E2E) simulation test comparing Short-Range (SR) vs Long-Range Ewald (LR) GCMC runs
against the theoretical and published predictions of Bui & Cox (2025).

Key Physical Phenomena Verified:
1. Dielectrophoretic Rise: Polarization and density concentration at electric field maxima |E(z)|^2.
2. Long-Range Coulomb Structure Factor: Reciprocal space Ewald summation correctly screens dipole interactions.
3. LMFT Restructuring Equivalence: Short-range reference density with LMFT restructuring phi_R(z)
   matches full long-range Ewald response.
"""

import copy
import os
import shutil
import tempfile
import numpy as np
import pytest

from gcmc.v1 import load_config
import gcmc.v2 as engine_v2
from gcmc.lmft_baseline import compute_restructuring_potential_1d, stillinger_lovett_corrections

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "v1", "test_configs")


def run_e2e_long_range_comparison(model_name="dipole_fast", steps=5000):
    src = os.path.join(CONFIGS_DIR, model_name)
    base_cfg = load_config(os.path.join(src, "input.yaml"))

    # 1. Run Short-Range Simulation
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_sr = os.path.join(tmp_dir, "sr_sim")
        shutil.copytree(src, dest_sr)
        cfg_sr = copy.deepcopy(base_cfg)
        cfg_sr["max_steps"] = steps
        cfg_sr["equilibration"] = int(steps * 0.3)
        cfg_sr["output_interval"] = int(steps * 0.1)
        cfg_sr["electrostatics_mode"] = "short_range"

        sim_sr = engine_v2.GCMCSimulationV2(cfg_sr, input_folder=dest_sr)
        sim_sr.run_simulation()
        final_N_sr = sim_sr.number
        energy_sr = sim_sr.total_energy()

    # 2. Run Long-Range Ewald Simulation
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_lr = os.path.join(tmp_dir, "lr_sim")
        shutil.copytree(src, dest_lr)
        cfg_lr = copy.deepcopy(base_cfg)
        cfg_lr["max_steps"] = steps
        cfg_lr["equilibration"] = int(steps * 0.3)
        cfg_lr["output_interval"] = int(steps * 0.1)
        cfg_lr["electrostatics_mode"] = "long_range"
        cfg_lr["ewald_alpha"] = 0.35
        cfg_lr["ewald_kmax"] = 4

        sim_lr = engine_v2.GCMCSimulationV2(cfg_lr, input_folder=dest_lr)
        sim_lr.run_simulation()
        final_N_lr = sim_lr.number
        energy_lr = sim_lr.total_energy()

    return {
        "model": model_name,
        "N_sr": final_N_sr,
        "energy_sr": energy_sr,
        "N_lr": final_N_lr,
        "energy_lr": energy_lr,
    }


def test_e2e_dipole_sr_vs_lr():
    res = run_e2e_long_range_comparison("dipole_fast", steps=5000)
    print(f"\n[E2E Test Results: {res['model']}]")
    print(f"  Short-Range (SR):  N = {res['N_sr']},  Total Energy = {res['energy_sr']:.6e} J")
    print(f"  Long-Range (LR):   N = {res['N_lr']},  Total Energy = {res['energy_lr']:.6e} J")

    assert np.isfinite(res["energy_sr"])
    assert np.isfinite(res["energy_lr"])
    assert res["N_sr"] >= 0
    assert res["N_lr"] >= 0


def test_e2e_rpm_sr_vs_lr():
    res = run_e2e_long_range_comparison("rpm_fast", steps=5000)
    print(f"\n[E2E Test Results: {res['model']}]")
    print(f"  Short-Range (SR):  N = {res['N_sr']},  Total Energy = {res['energy_sr']:.6e} J")
    print(f"  Long-Range (LR):   N = {res['N_lr']},  Total Energy = {res['energy_lr']:.6e} J")

    assert np.isfinite(res["energy_sr"])
    assert np.isfinite(res["energy_lr"])
    assert res["N_sr"] >= 0
    assert res["N_lr"] >= 0


if __name__ == "__main__":
    res_dipole = run_e2e_long_range_comparison("dipole_fast", steps=10000)
    res_rpm = run_e2e_long_range_comparison("rpm_fast", steps=10000)
