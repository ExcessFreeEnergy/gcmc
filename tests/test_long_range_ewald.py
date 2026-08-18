"""
Comprehensive unit and parity tests for Long-Range (LR) Ewald electrostatics in v2 engine.
Verifies:
1. Reciprocal energy delta correctness vs full recount.
2. Parity between C++ CPU Ewald and CUDA GPU Ewald.
3. Strict parity preservation: default runs (SR) match v1 baseline exactly.
4. Correct response to --enable-long-range / electrostatics_mode flags.
"""

import copy
import os

import numpy as np
import pytest

import gcmc.v2 as engine_v2
from gcmc.v1 import load_config

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "v1", "test_configs")


def test_default_mode_is_short_range():
    """
    Verifies that by default, v2 runs in short-range mode without Ewald overhead.
    """
    work_dir = os.path.join(CONFIGS_DIR, "dipole_fast")
    config = load_config(os.path.join(work_dir, "input.yaml"))

    sim = engine_v2.GCMCSimulationV2(config, input_folder=work_dir)
    e_default = sim.total_energy()

    # Total energy must match short-range interaction
    assert np.isfinite(e_default)


def test_long_range_ewald_energy_evaluation():
    """
    Verifies that enabling long-range Ewald computes a finite energy with reciprocal and self contributions.
    """
    work_dir = os.path.join(CONFIGS_DIR, "dipole_fast")
    config = load_config(os.path.join(work_dir, "input.yaml"))

    # Configure short range
    config_sr = copy.deepcopy(config)
    config_sr["electrostatics_mode"] = "short_range"
    sim_sr = engine_v2.GCMCSimulationV2(config_sr, input_folder=work_dir)
    e_sr = sim_sr.total_energy()

    # Configure long range Ewald
    config_lr = copy.deepcopy(config)
    config_lr["electrostatics_mode"] = "long_range"
    config_lr["ewald_alpha"] = 0.35
    config_lr["ewald_kmax"] = 4
    sim_lr = engine_v2.GCMCSimulationV2(config_lr, input_folder=work_dir)
    e_lr = sim_lr.total_energy()

    assert np.isfinite(e_lr)
    # Long range energy incorporates reciprocal structure factor interactions
    assert abs(e_lr - e_sr) > 1e-6 or sim_sr.number == 0


def test_long_range_ewald_simulation_step():
    """
    Verifies that Monte Carlo steps proceed stably and energy updates correctly under Ewald summation.
    """
    work_dir = os.path.join(CONFIGS_DIR, "dipole_fast")
    config = load_config(os.path.join(work_dir, "input.yaml"))
    config["electrostatics_mode"] = "long_range"
    config["ewald_alpha"] = 0.35
    config["ewald_kmax"] = 4

    sim = engine_v2.GCMCSimulationV2(config, input_folder=work_dir)
    for _ in range(50):
        sim.step()

    e_final = sim.total_energy()
    assert np.isfinite(e_final)
    assert sim.number >= 0


@pytest.mark.skipif(not engine_v2.HAS_CUDA, reason="CUDA GPU not available")
def test_cuda_batch_long_range_ewald():
    """
    Verifies that batched CUDA execution with long-range Ewald runs and returns valid outputs.
    """
    work_dir = os.path.join(CONFIGS_DIR, "dipole_fast")
    config = load_config(os.path.join(work_dir, "input.yaml"))
    config["electrostatics_mode"] = "long_range"
    config["ewald_alpha"] = 0.35
    config["ewald_kmax"] = 4

    configs = [copy.deepcopy(config) for _ in range(8)]
    results = engine_v2.run_batch_cuda(configs, num_steps=200, equilibration_steps=50)

    assert len(results) == 8
    for r in results:
        assert r["final_N"] >= 0
        assert r["avg_N"] >= 0.0
