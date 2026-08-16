"""End-to-end regression test for SPC/E water fluid."""

import os
import gzip
import numpy as np
import pytest

from gcmc.v1 import (
    load_config,
    GCMC_FF_H2O_Simulation,
)
from gcmc.v1.main import run_simulation_job


def test_h2o_fast_simulation(run_dir):
    work_dir = run_dir("h2o_fast")
    config_path = os.path.join(work_dir, "input.yaml")
    config = load_config(config_path)

    # Run the simulation job
    sim = run_simulation_job(config, input_folder=work_dir)

    assert isinstance(sim, GCMC_FF_H2O_Simulation)
    assert sim.number > 0
    assert np.isfinite(sim.total_energy())

    # Verify logfile
    log_file = os.path.join(work_dir, "gcmc.log")
    assert os.path.exists(log_file)
    with open(log_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) > 2
    assert "Step Total_number Energy" in lines[0]

    # Verify output.xyz.gz
    xyz_file = os.path.join(work_dir, "output.xyz.gz")
    assert os.path.exists(xyz_file)
    with gzip.open(xyz_file, 'rt') as f:
        content = f.read()
    assert len(content) > 0
    assert "O " in content and "H1 " in content and "H2 " in content
