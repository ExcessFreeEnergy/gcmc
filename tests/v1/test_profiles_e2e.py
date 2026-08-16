"""End-to-end test for density profile calculation and post-processing tools."""

import os
import numpy as np
import pytest

from gcmc.v1 import load_config
from gcmc.v1.main import run_simulation_job
from gcmc.v1.utils.get_density_profile import (
    read_extended_xyz,
    average_density_profiles,
)


def test_density_profile_extraction(run_dir):
    work_dir = run_dir("dipole_fast")
    config_path = os.path.join(work_dir, "input.yaml")
    config = load_config(config_path)

    # Run simulation
    run_simulation_job(config, input_folder=work_dir)

    xyz_file = os.path.join(work_dir, "output.xyz.gz")
    assert os.path.exists(xyz_file)

    # Parse trajectory
    positions_list, lattice_vectors_list = read_extended_xyz(xyz_file)
    assert len(positions_list) > 0
    assert len(lattice_vectors_list) > 0

    # Compute density profile
    bin_centers, avg_density = average_density_profiles(
        positions_list, lattice_vectors_list, bins=50
    )

    assert len(bin_centers) == 50
    assert 'A' in avg_density
    assert 'B' in avg_density
    assert 'C' in avg_density

    density_A = avg_density['A']
    assert len(density_A) == 50
    assert np.all(np.isfinite(density_A))
    assert np.any(density_A > 0.0)
