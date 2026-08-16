"""End-to-end regression test for Restricted Primitive Model (RPM ionic fluid)."""

import os
import gzip
import numpy as np
import pytest

from gcmc.v1 import (
    load_config,
    GCMC_FF_TwoType_Simulation,
)
from gcmc.v1.main import run_simulation_job


def test_rpm_fast_simulation(run_dir):
    work_dir = run_dir("rpm_fast")
    config_path = os.path.join(work_dir, "input.yaml")
    config = load_config(config_path)

    # Run the simulation job
    sim = run_simulation_job(config, input_folder=work_dir)

    assert isinstance(sim, GCMC_FF_TwoType_Simulation)
    assert (sim.number1 + sim.number2) > 0
    assert np.isfinite(sim.total_energy())

    # Verify logfile
    log_file = os.path.join(work_dir, "gcmc.log")
    assert os.path.exists(log_file)
    with open(log_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 2
        assert "Step Total_number" in lines[0]

    # Verify output.xyz.gz
    xyz_file = os.path.join(work_dir, "output.xyz.gz")
    assert os.path.exists(xyz_file)
    with gzip.open(xyz_file, 'rt') as f:
        content = f.read()
    assert len(content) > 0

    # Verify density_x.dat profile
    density_file = os.path.join(work_dir, "density_x.dat")
    assert os.path.exists(density_file)
    data = np.loadtxt(density_file, comments='#')
    assert data.ndim == 2
    assert data.shape[1] >= 2  # x, rho1, [rho2]
    assert np.all(np.isfinite(data))
