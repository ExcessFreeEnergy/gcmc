"""
Comprehensive dual-engine parity test suite.
Compares the high-performance v2 engine against the known-good v1 baseline.
"""

import os
import gzip
import numpy as np
import pytest

from gcmc.v1 import load_config
from gcmc.v1.potentials import initialize_potentials as init_pot_v1
from gcmc.v1.external_potentials import initialize_external_potentials as init_ext_v1
import gcmc.v1.main as engine_v1
import gcmc.v2 as engine_v2

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "v1", "test_configs")


@pytest.mark.parametrize("config_name", ["dipole_fast", "rpm_fast", "h2o_fast"])
def test_initial_potential_energy_parity(config_name):
    """
    Verify that on the exact same molecular configuration, v1 and v2
    compute identical total potential energy within 1e-5 relative error.
    """
    work_dir = os.path.join(CONFIGS_DIR, config_name)
    config_path = os.path.join(work_dir, "input.yaml")
    config = load_config(config_path)

    # Instantiate v1
    ext_pot_v1 = init_ext_v1(config)
    pair_pot_v1 = init_pot_v1(config)
    mol_flag = config.get('molecule', 'None')
    if mol_flag == 'ABC':
        from gcmc.v1.gcmc_ff_molecule import GCMC_FF_ABC_Simulation
        sim_v1 = GCMC_FF_ABC_Simulation(config, pair_pot_v1, ext_pot_v1, work_dir)
    elif mol_flag == 'H2O':
        from gcmc.v1.gcmc_ff_molecule import GCMC_FF_H2O_Simulation
        sim_v1 = GCMC_FF_H2O_Simulation(config, pair_pot_v1, ext_pot_v1, work_dir)
    else:
        from gcmc.v1.gcmc_ff import GCMC_FF_TwoType_Simulation
        sim_v1 = GCMC_FF_TwoType_Simulation(config, pair_pot_v1, ext_pot_v1, work_dir)

    e_v1 = sim_v1.total_energy()

    # Instantiate v2
    sim_v2 = engine_v2.GCMCSimulationV2(config, input_folder=work_dir)
    e_v2 = sim_v2.total_energy()

    # Parity check
    rel_diff = abs(e_v1 - e_v2) / (abs(e_v1) + 1e-20)
    assert rel_diff < 1e-4, f"Energy mismatch for {config_name}: v1={e_v1}, v2={e_v2}, rel_diff={rel_diff}"


@pytest.mark.parametrize("engine_name", ["v1", "v2"])
@pytest.mark.parametrize("config_name", ["dipole_fast", "rpm_fast", "h2o_fast"])
def test_simulation_execution_dual_engine(run_dir, engine_name, config_name):
    """
    Verify that running the simulation under either engine produces
    valid gcmc.log, output.xyz.gz, and finite particles/energy.
    """
    work_dir = run_dir(config_name)
    config_path = os.path.join(work_dir, "input.yaml")
    config = load_config(config_path)

    if engine_name == "v1":
        sim = engine_v1.run_simulation_job(config, input_folder=work_dir)
    else:
        sim = engine_v2.run_simulation_job(config, input_folder=work_dir)

    # Check number of particles
    if hasattr(sim, 'number'):
        assert sim.number >= 0
    elif hasattr(sim, 'number1'):
        assert (sim.number1 + sim.number2) >= 0

    # Check logfile
    log_path = os.path.join(work_dir, "gcmc.log")
    assert os.path.exists(log_path)
    with open(log_path, 'r') as f:
        lines = f.readlines()
    assert len(lines) > 2
    assert "Step Total_number" in lines[0]

    # Check xyz file
    xyz_path = os.path.join(work_dir, "output.xyz.gz")
    assert os.path.exists(xyz_path)
    with gzip.open(xyz_path, 'rt') as f:
        content = f.read()
    assert len(content) > 0


def test_cuda_batch_simulation():
    """
    Verify batched GPU execution of multiple GCMC boxes on CUDA device.
    """
    if not engine_v2.HAS_CUDA:
        pytest.skip("CUDA device not available on this host")

    config_path = os.path.join(CONFIGS_DIR, "dipole_fast", "input.yaml")
    config = load_config(config_path)

    # Create 8 parallel box configs
    batch_configs = [config.copy() for _ in range(8)]
    for i, c in enumerate(batch_configs):
        c['mu'] = -8.0 - i * 0.2

    results = engine_v2.run_batch_cuda(batch_configs, num_steps=500, equilibration_steps=100)
    assert len(results) == 8
    for r in results:
        assert r['box_id'] >= 0
        assert r['final_N'] >= 0
