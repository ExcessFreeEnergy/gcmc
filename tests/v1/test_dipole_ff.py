"""End-to-end regression test for linear dipolar Stockmayer fluid (ABC molecule)."""

import os
import gzip
import numpy as np
import pytest

from gcmc.v1 import (
    load_config,
    initialize_potentials,
    initialize_external_potentials,
    GCMC_FF_ABC_Simulation,
)
from gcmc.v1.main import run_simulation_job


def test_dipole_fast_simulation(run_dir):
    work_dir = run_dir("dipole_fast")
    config_path = os.path.join(work_dir, "input.yaml")
    config = load_config(config_path)

    # Run the simulation job
    sim = run_simulation_job(config, input_folder=work_dir)

    assert isinstance(sim, GCMC_FF_ABC_Simulation)
    assert sim.number > 0
    assert np.isfinite(sim.total_energy())

    # Verify logfile
    log_file = os.path.join(work_dir, "gcmc.log")
    assert os.path.exists(log_file)
    with open(log_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) > 2  # Header + multiple step logs
    assert lines[0].strip() == "Step Total_number Energy"

    # Verify output.xyz.gz
    xyz_file = os.path.join(work_dir, "output.xyz.gz")
    assert os.path.exists(xyz_file)
    with gzip.open(xyz_file, 'rt') as f:
        content = f.read()
    assert len(content) > 0
    assert "Properties=species:S:1:pos:R:3" in content
    assert "A " in content and "B " in content and "C " in content
