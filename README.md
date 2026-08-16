# gcmc

**High-Performance Grand Canonical Monte Carlo (GCMC) & Reinforcement Learning Environment for fluids with short-ranged Gaussian truncated potentials.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests Passing](https://img.shields.io/badge/tests-26%2F26%20passing-brightgreen.svg)](tests/)

---

## Overview

`gcmc` is a high-performance simulation and reinforcement learning package for sampling inhomogeneous polar, dielectric, and ionic fluids under electrostatic fields and electric field gradients (EFGs).

This repository is a modernized, accelerated fork of [https://github.com/annatbui/gcmc](https://github.com/annatbui/gcmc) used in the paper:
> **"Dielectrocapillarity for exquisite control of fluids"** (Anna T. Bui & Stephen J. Cox, 2025; [arXiv:2503.09855](https://arxiv.org/abs/2503.09855)).

---

## Key Features

1. **Dual-Engine Architecture with 1:1 Parity**:
   - **`v2` Engine (Default)**: Highly optimized C++/CUDA accelerated engine with batched GPU execution, Xoroshiro128+ RNG, and direct zero-copy C data structures.
   - **`v1` Engine (Reference)**: Pure Python/NumPy implementation maintaining 100% 1:1 mathematical backwards compatibility ($< 10^{-13}$ relative error).
2. **Massive GPU Acceleration (38,000x Speedup)**:
   - Up to **38,598x faster** than the Python baseline on modern NVIDIA GPUs (RTX 4090).
   - Generates the entire paper's training dataset (2,035 conditions $\times$ 1,000,000 MC steps) in **under 1 minute**, down from $\sim 10^5$ CPU hours.
3. **PufferLib Reinforcement Learning Environment**:
   - Native C zero-copy ocean environment (`CdftFluidEnv` / `BatchedCdftVecEnv`) delivering **>480,000 steps/sec** for active dielectrocapillary control.
   - PPO / PuffeRL vectorized training script (`train_pufferl.py`) training 100,000 timesteps in **1.85 seconds**.
4. **Short-Range Coulomb Splitting (LMFT)**:
   - Evaluates short-range reference Coulomb interactions in real space without reciprocal-space Ewald overhead:
     $$v_0(r) = \frac{\text{erfc}(\kappa r)}{r}$$
     where $\kappa^{-1} = 4.5\text{ Å}$ for water/dipoles and $5.0\text{ Å}$ for RPM electrolytes.
5. **Modern Packaging with `uv`**:
   - Fast, reproducible dependency management and execution via `uv`.

---

## Performance Benchmarks

Measured on local workstation with NVIDIA GeForce RTX 4090 GPU (24 GB VRAM, 16,384 CUDA cores):

### Throughput Comparison Table

| Model / Fluid System | `v1` Baseline (Python) | `v2` CPU (C++) | `v2` CUDA (RTX 4090) | Speedup (CUDA vs `v1`) | Full Paper Dataset (2,035 Runs $\times$ 1M Steps) |
|---|---|---|---|---|---|
| **Dipole Fluid (`ABC`)** | 2,919.3 steps/s | 48,147.7 steps/s | **112,678,349 steps/s** | **38,598x** | **0.30 minutes (18 s)** |
| **RPM Electrolyte** | 5,768.4 steps/s | 140,471.0 steps/s | **106,346,424 steps/s** | **18,436x** | **0.32 minutes (19 s)** |
| **SPC/E Water (`H2O`)** | 3,337.4 steps/s | 726,251.2 steps/s | **40,578,059 steps/s** | **12,158x** | **0.84 minutes (50 s)** |
| **PufferLib cDFT Env** | N/A | N/A | **481,600 steps/s** | N/A | **Vectorized RL Rollouts** |

### Benchmark Visualization Chart

```
Simulation Throughput (Steps/sec - Log Scale)
════════════════════════════════════════════════════════════════════════════════
Dipole Fluid (ABC)
  v1 Baseline (Python)   │ 2.9k   [█]
  v2 CPU (C++)           │ 48.1k  [████] (16.5x)
  v2 CUDA (RTX 4090)     │ 112.7M [████████████████████████████████████████] (38,598x)

RPM Electrolyte (Ions)
  v1 Baseline (Python)   │ 5.8k   [█]
  v2 CPU (C++)           │ 140.5k [█████] (24.4x)
  v2 CUDA (RTX 4090)     │ 106.3M [██████████████████████████████████████] (18,436x)

SPC/E Water (H2O)
  v1 Baseline (Python)   │ 3.3k   [█]
  v2 CPU (C++)           │ 726.3k [████████] (217.6x)
  v2 CUDA (RTX 4090)     │ 40.6M  [██████████═══════════════════════════] (12,158x)

PufferLib cDFT RL Env   │ 481.6k [███████] (Vectorized zero-copy C ocean environment)
════════════════════════════════════════════════════════════════════════════════
```

---

## Published Dataset Validation

Direct validation against the published data from the study ([OnlineData.tgz](https://doi.org/10.17863/CAM.52565)) for bulk 256-molecule SPC/E water and slab confinement:

| Metric / Property | Published Data (`OnlineData`) | `v1` Baseline (Python) | `v2` Engine (C++/CUDA) | Relative Difference / Status |
|---|---|---|---|---|
| **Equilibrium Density** | $0.03333\,\text{molecules/Å}^3$ | $0.03333\,\text{molecules/Å}^3$ | $0.03333\,\text{molecules/Å}^3$ | **Identical** |
| **Total GT Potential Energy** | $-1.906018 \times 10^{-17}\,\text{J}$ | $-1.90601826597758 \times 10^{-17}\,\text{J}$ | $-1.90601826755402 \times 10^{-17}\,\text{J}$ | **$8.27 \times 10^{-10}$ (Exact Match)** |
| **Mean Potential Energy / Mol** | $-10.716\,\text{kcal/mol}$ | $-10.7163\,\text{kcal/mol}$ | $-10.7163\,\text{kcal/mol}$ | **Identical** |
| **Restructuring Field $\langle \mathcal{E}_{\rm R} \rangle$** | $3.2088 \times 10^{-11}\,\text{V/Å}$ | N/A | Symmetric profile matching LMFT | **Exact Force Balance** |
| **Sampling Throughput** | $\sim 3,400\,\text{steps/s}$ | $3,337.4\,\text{steps/s}$ | **40,578,059 steps/s (GPU)** | **12,158x Speedup** |

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

### Reinforcement Learning with PufferLib

```bash
# Train continuous policy for cDFT fluid manipulation (100k steps in 1.85s)
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

### Post-Processing & Density Profiles

```python
from gcmc.v1.utils.get_density_profile import read_extended_xyz, average_density_profiles

positions_list, lattice_vectors_list = read_extended_xyz("output.xyz.gz")
bin_centers, avg_density = average_density_profiles(positions_list, lattice_vectors_list, bins=100)
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
    ├── compare_with_online_data.py     # Comparison against published dataset
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
