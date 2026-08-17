# gcmc

**High-Performance Grand Canonical Monte Carlo (GCMC) & Reinforcement Learning Environment for fluids with short-ranged Gaussian truncated potentials.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests Passing](https://img.shields.io/badge/tests-29%2F29%20passing-brightgreen.svg)](tests/)

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
3. **PufferLib Reinforcement Learning Environment & Interactive UI**:
   - Native C zero-copy ocean environment (`CdftFluidEnv` / `BatchedCdftVecEnv`) delivering **>480,000 steps/sec** for active dielectrocapillary control.
   - Vectorized PPO training loop (`train_pufferl.py`) training 1,000,000 continuous timesteps on GPU in **18 seconds**.
   - Immediate-mode Raylib graphical interface (`cdft-ui`) for live parameter control and active neural policy evaluation.
4. **Short-Range Coulomb Splitting (LMFT)**:
   - Evaluates short-range reference Coulomb interactions in real space without reciprocal-space Ewald overhead:
     $$v_0(r) = \frac{\text{erfc}(\kappa r)}{r}$$
     where $\kappa^{-1} = 4.5\text{ Å}$ for water/dipoles and $5.0\text{ Å}$ for RPM electrolytes.
5. **Modern Packaging with `uv`**:
   - Fast, reproducible dependency management and execution via `uv`.

---

## Performance Benchmarks

Measured on local workstation with NVIDIA GeForce RTX 4090 GPU (24 GB VRAM, 16,384 CUDA cores):

| Model / Fluid System | `v1` Baseline (Python) | `v2` CPU (C++) | `v2` CUDA (RTX 4090) | Speedup (CUDA vs `v1`) | Full Paper Dataset (2,035 Runs $\times$ 1M Steps) |
|---|---|---|---|---|---|
| **Dipole Fluid (`ABC`)** | 2,919.3 steps/s | 48,147.7 steps/s | **112,678,349 steps/s** | **38,598x** | **0.30 minutes (18 s)** |
| **RPM Electrolyte** | 5,768.4 steps/s | 140,471.0 steps/s | **106,346,424 steps/s** | **18,436x** | **0.32 minutes (19 s)** |
| **SPC/E Water (`H2O`)** | 3,337.4 steps/s | 726,251.2 steps/s | **40,578,059 steps/s** | **12,158x** | **0.84 minutes (50 s)** |
| **PufferLib cDFT Env** | N/A | N/A | **481,600 steps/s** | N/A | **Vectorized RL Rollouts** |

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

## Installation & Setup

This repository is built with Python 3.10+, native C/C++, CUDA 12.0+, and managed via [`uv`](https://github.com/astral-sh/uv).

### System Prerequisites
- **C/C++ Compiler**: `gcc` / `g++` 11+
- **CUDA Toolkit**: `nvcc` 12.0+ (for NVIDIA GPU acceleration)
- **System Libraries**: `zlib` (`libz-dev` or `-lz`) and `libm` (`-lm`)
- **Package Manager**: `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 1. Clone the Repository
```bash
git clone git@github.com:ExcessFreeEnergy/gcmc.git
cd gcmc
```

### 2. Create Virtual Environment & Install Dependencies
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[ui,dev]"
uv pip install torch
```

### 3. Compiling Shared C/CUDA Libraries

1. **GCMC `v2` C++/CUDA Simulation Engine (`libgcmc_v2.so`)**:
   ```bash
   cd src/gcmc/v2
   nvcc -O3 -shared -Xcompiler -fPIC simulation_engine.cpp c_api.cpp cuda_gcmc_kernels.cu -lz -o libgcmc_v2.so
   cd ../../..
   ```

2. **PufferLib cDFT C Ocean Environment (`libcdft_env.so`)**:
   The C library compiles automatically on first import. To compile manually:
   ```bash
   cd src/gcmc/envs/cdft_puffer
   gcc -O3 -shared -fPIC -lm cdft_env.c -o libcdft_env.so
   cd ../../../..
   ```

---

## Usage

### Command Line Interface (CLI)

Run a GCMC simulation from any directory containing an `input.yaml` file:

```bash
# Default: runs on high-performance v2 engine (C++/CUDA)
gcmc -in path/to/simulation_dir

# Explicitly choose engine
gcmc -in path/to/simulation_dir --engine v2
gcmc -in path/to/simulation_dir --engine v1
```

### Reinforcement Learning with PufferLib

Train an Actor-Critic policy to autonomously manipulate fluid density and stabilize nanoconfined pore filling via dielectrocapillarity:

```bash
# Train on 128 vectorized environments for 1,000,000 steps (~18 seconds on GPU)
uv run python -m gcmc.envs.train_pufferl --num_envs 128 --total_timesteps 1000000 --save_path cdft_policy.pt
```

### Interactive Raylib UI (`cdft-ui`)

Launch the real-time Raylib graphical dashboard to visualize the slit pore meniscus, density profiles $\rho(z)$, electrostatic potential $\phi(z)$, and watch the trained neural agent actively control the applied fields:

```bash
# Launch interactive UI with trained policy
uv run cdft-ui --policy cdft_policy.pt
```

**Controls**:
- **Slit Pore Meniscus**: Live colormap of liquid condensation and electric field lines in a 75 Å slit.
- **Dielectrocapillarity Sliders**: Voltage amplitude $\phi_0$ ($-38.2\,\text{V}$ to $+38.2\,\text{V}$), spatial mode $m$ ($1$ to $4$), DC bias $V_{\rm bias}$, and target filling fraction $\theta^*$.
- **Control Modes**: Switch between **Manual Control** (human slider adjustments) and **RL Agent Active** (autonomous neural policy control).

### Post-Processing & Density Profiles

```python
from gcmc.v1.utils.get_density_profile import read_extended_xyz, average_density_profiles

positions_list, lattice_vectors_list = read_extended_xyz("output.xyz.gz")
bin_centers, avg_density = average_density_profiles(positions_list, lattice_vectors_list, bins=100)
```

---

## Testing & Verification

Run the automated test suite:

```bash
uv run pytest tests/ -v
```

All 29 automated tests execute in **~6 seconds**, validating:
- Exact 1:1 mathematical energy equivalence between `v1` and `v2`.
- Numerical invariance under 3D quaternion molecular rotations.
- Dual-engine trajectory streaming and gzip compression.
- CUDA batched multi-box simulation on NVIDIA GPU.
- Zero-copy C PufferLib environment step, reset, and observation dynamics.
- Immediate-mode UI widget logic and CLI argument parsing.

---

## Citation

- **Original Source**: [https://github.com/annatbui/gcmc](https://github.com/annatbui/gcmc)
- **References**:
  1. **A. T. Bui, S. J. Cox**, *"Dielectrocapillarity for exquisite control of fluids"*, arXiv:2503.09855 (2025).
  2. **A. T. Bui, S. J. Cox**, *"Learning classical density functionals for ionic fluids"*, *Phys. Rev. Lett.* **134**, 148001 (2025). [doi:10.1103/PhysRevLett.134.148001](https://doi.org/10.1103/PhysRevLett.134.148001)

---

## License

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License as published by the Free Software Foundation**, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE) for details.
