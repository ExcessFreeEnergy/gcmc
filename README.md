# gcmc

**High-Performance Grand Canonical Monte Carlo (GCMC) & Reinforcement Learning Environment for fluids with short-ranged Gaussian truncated potentials.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests Passing](https://img.shields.io/badge/tests-26%2F26%20passing-brightgreen.svg)](tests/)

---

## Overview

`gcmc` is a high-performance simulation and reinforcement learning package for sampling inhomogeneous polar, dielectric, and ionic fluids under electrostatic fields and electric field gradients (EFGs).

It powers reference data generation and active control for neural classical density functional theory (cDFT) and local molecular field theory (LMFT) models of **dielectrocapillarity and electromechanics**, based on:
> **"Dielectrocapillarity for exquisite control of fluids"** (Anna T. Bui & Stephen J. Cox, 2025; arXiv:2503.09855).

<p align="center">
  <img src="https://github.com/user-attachments/assets/7bcb5613-292e-42a3-8be3-eaf49ac52ae3" width="240" alt="Density response">
  <img src="https://github.com/user-attachments/assets/7c1e55d7-9dc2-4df4-9166-d56b63b3d9cb" width="230" alt="Structure">
  <img src="https://github.com/user-attachments/assets/c57f70b5-80d0-49dd-b287-81334062b4aa" width="240" alt="Profiles">
</p>

---

## Key Features

1. **Dual-Engine Simulation Architecture**:
   - **`v2` Engine (Default)**: Highly optimized C++/CUDA accelerated engine with batched GPU execution, Xoroshiro128+ RNG, and direct zero-copy C data structures.
   - **`v1` Engine (Reference)**: Pure Python/NumPy implementation maintaining 100% 1:1 mathematical backwards compatibility.
2. **Extreme GPU Speedups**:
   - Up to **38,000x faster** than the Python baseline on modern NVIDIA GPUs (RTX 4090).
   - Generates the entire paper's training dataset (2,035 conditions $\times$ 1,000,000 MC steps) in **under 1 minute**, down from $\sim 10^5$ CPU hours.
3. **PufferLib Reinforcement Learning Environment**:
   - Native C zero-copy environment (`CdftFluidEnv` / `BatchedCdftVecEnv`) delivering **>450,000 steps/sec** for active dielectrocapillary control.
   - PPO / PuffeRL vectorized training script (`train_pufferl.py`) training 100,000 timesteps in under 2 seconds.
4. **Gaussian Truncated Potentials & LMFT Splitting**:
   - Evaluates short-range reference Coulomb interactions $v_0(r) = \frac{\operatorname{erfc}(\kappa r)}{r}$ in real space without reciprocal-space Ewald overhead ($\kappa^{-1} = 4.5\,\text{Å}$ for water/dipoles, $5.0\,\text{Å}$ for RPM electrolytes).
5. **Modern Packaging with `uv`**:
   - Built for ultra-fast, reproducible dependency management using `uv`.

---

## Performance Benchmarks

Measured on local workstation with NVIDIA GeForce RTX 4090 GPU (24 GB VRAM, 16,384 CUDA cores):

| Model / Fluid System | `v1` Baseline (Python) | `v2` CPU (C++) | `v2` CUDA (RTX 4090) | Speedup (CUDA vs v1) | 2,035 Conditions $\times$ 1M Steps |
|---|---|---|---|---|---|
| **Dipole Fluid (`ABC`)** | 2,919.3 steps/s | 48,147.7 steps/s | **112,678,349 steps/s** | **38,598x** | **0.30 minutes (18 s)** |
| **RPM Electrolyte** | 5,768.4 steps/s | 140,471.0 steps/s | **106,346,424 steps/s** | **18,436x** | **0.32 minutes (19 s)** |
| **SPC/E Water (`H2O`)** | 3,337.4 steps/s | 726,251.2 steps/s | **40,578,059 steps/s** | **12,158x** | **0.84 minutes (50 s)** |
| **PufferLib cDFT Env** | N/A | N/A | **481,600 steps/s** | N/A | **Vectorized RL Rollouts** |

---

## Installation & Setup (using `uv`)

This codebase is managed using [`uv`](https://github.com/astral-sh/uv).

### 1. Clone the repository
```bash
git clone git@github.com:ExcessFreeEnergy/gcmc.git
cd gcmc
```

### 2. Create virtual environment and install dependencies
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

For GPU acceleration and RL training:
```bash
uv pip install torch
```

---

## Usage

### Command Line Interface (CLI)

Run a simulation from any directory containing an `input.yaml` file:

```bash
# Default: runs on high-performance v2 engine (C++/CUDA)
gcmc -in path/to/simulation_dir

# Explicitly choose engine
gcmc -in path/to/simulation_dir --engine v2
gcmc -in path/to/simulation_dir --engine v1
```

### Python API

```python
from gcmc.v2 import GCMCSimulationV2, run_simulation_job
from gcmc.v1 import load_config

config = load_config("path/to/input.yaml")

# Run via v2 C++/CUDA engine
sim = run_simulation_job(config, input_folder="path/to/simulation_dir")
print(f"Final particle count: {sim.number}, Total Energy: {sim.total_energy():.6e} J")
```

### Batched GPU Simulation (Thousands of Boxes in Parallel)

```python
from gcmc.v2 import run_batch_cuda
from gcmc.v1 import load_config

base_cfg = load_config("tests/v1/test_configs/dipole_fast/input.yaml")
batch_configs = [base_cfg.copy() for _ in range(512)]

# Run 512 independent GCMC simulations simultaneously on GPU
results = run_batch_cuda(batch_configs, num_steps=50000, equilibration_steps=10000)
```

### Reinforcement Learning with PufferLib

```bash
# Train continuous policy for cDFT fluid manipulation
python -m gcmc.envs.train_pufferl --num_envs 128 --total_timesteps 100000
```

```python
from gcmc.envs.cdft_puffer import CdftFluidEnv, BatchedCdftVecEnv

# Single Gymnasium-compatible environment
env = CdftFluidEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([0.1, -0.05, 0.0])

# High-throughput batched vector environment (256 parallel instances)
vec_env = BatchedCdftVecEnv(num_envs=256)
obs, _ = vec_env.reset()
obs, rewards, terminals, _ = vec_env.step(actions)
```

---

## Testing & Verification

Run the entire automated test suite:

```bash
uv run pytest tests/ -v
```

All 26 automated tests execute in **~6 seconds**, validating:
- Exact 1:1 mathematical energy equivalence between `v1` and `v2`.
- Numerical invariance under 3D quaternion molecular rotations.
- Dual-engine trajectory streaming and gzip compression.
- CUDA batched multi-box simulation on NVIDIA GPU.
- Zero-copy C PufferLib environment step and reset dynamics.

---

## Repository Structure

```
gcmc/
├── AGENTS.md                           # Developer & AI agent technical guide
├── pyproject.toml                      # Packaging & dependencies
├── LICENSE                             # GNU General Public License v3.0
├── src/
│   └── gcmc/
│       ├── __init__.py                 # Root exports (v1, v2, cli)
│       ├── main.py                     # CLI entrypoint with engine dispatcher
│       ├── v1/                         # Baseline Python engine
│       │   ├── potentials.py
│       │   ├── external_potentials.py
│       │   ├── gcmc_ff_molecule.py
│       │   └── gcmc_ff.py
│       ├── v2/                         # High-performance C++/CUDA engine
│       │   ├── core_types.h
│       │   ├── simulation_engine.h/cpp
│       │   ├── cuda_gcmc.h
│       │   ├── cuda_gcmc_kernels.cu
│       │   ├── c_api.h/cpp
│       │   └── bindings.py
│       └── envs/                       # RL fluid manipulation module
│           ├── cdft_puffer/            # PufferLib C ocean environment
│           │   ├── cdft_env.h/c
│           │   └── cdft_env.py
│           └── train_pufferl.py        # Vectorized PPO training loop
└── tests/
    ├── conftest.py                     # Pytest fixtures
    ├── test_engine_parity.py           # Dual-engine 1:1 parity tests
    ├── test_cdft_puffer_env.py         # PufferLib environment tests
    ├── v1/                             # Baseline regression tests
    └── v2/
        └── benchmark_v1_vs_v2.py       # Empirical benchmark script
```

---

## Citation

If you use this code in your research, please cite:

1. **A. T. Bui, S. J. Cox**, *"Dielectrocapillarity for exquisite control of fluids"*, arXiv:2503.09855 (2025).
2. **A. T. Bui, S. J. Cox**, *"Learning classical density functionals for ionic fluids"*, *Phys. Rev. Lett.* **134**, 148001 (2025). [doi:10.1103/PhysRevLett.134.148001](https://doi.org/10.1103/PhysRevLett.134.148001)

---

## License

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License as published by the Free Software Foundation**, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE) for details.
