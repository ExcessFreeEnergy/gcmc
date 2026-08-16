# gcmc

**Grand Canonical Monte Carlo (GCMC) for fluids with short-ranged Gaussian truncated potentials.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/v1/)

---

## About the Code

`gcmc` is a high-performance simulation engine designed to sample inhomogeneous polar, dielectric, and ionic fluids in the grand canonical ensemble $(\mu, V, T)$. It is used to generate exact reference training data for neural classical density functional theory (cDFT) and local molecular field theory (LMFT) models of dielectrocapillarity and electromechanics.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7bcb5613-292e-42a3-8be3-eaf49ac52ae3" width="240" alt="Density response">
  <img src="https://github.com/user-attachments/assets/7c1e55d7-9dc2-4df4-9166-d56b63b3d9cb" width="230" alt="Structure">
  <img src="https://github.com/user-attachments/assets/c57f70b5-80d0-49dd-b287-81334062b4aa" width="240" alt="Profiles">
</p>

### Gaussian Truncated Potential & LMFT
Direct calculation of long-range electrostatics in neural functionals is computationally prohibitive. `gcmc` utilizes a Coulombic splitting:

```math
\frac{1}{r} = v_0(r) + v_1(r) \equiv \frac{\operatorname{erfc}(\kappa r)}{r} + \frac{\operatorname{erf}(\kappa r)}{r}
```

The short-ranged reference potential $v_0(r)$ is purely local and evaluated in real space without reciprocal-space Ewald/PPPM solvers:
- **Water (SPC/E) & Dipolar Stockmayer Fluids**: $\kappa^{-1} = 4.5\,\text{Å}$
- **Electrolytes (Restricted Primitive Model)**: $\kappa^{-1} = 5.0\,\text{Å}$

<div align="center">
  <img src="https://github.com/user-attachments/assets/5c85b7f2-4042-4bcd-b0d6-5e0452f68a2b" width="30%" alt="Coulomb Splitting">
</div>

### Hybrid GCMC + MD Workflow
In dense, subcritical polar liquids where particle insertion acceptance is low, `gcmc` is used to determine the exact equilibrium average particle number $N_{\rm ave} = \langle N \rangle_{\mu, V, T}$. Canonical ($NVT$) Molecular Dynamics (via LAMMPS + LMFT `pair_lj_cut_coul_GT`) is subsequently initialized from this state to rapidly compute fine-grid number $\rho(z)$ and charge $n(z)$ density profiles ($\Delta z = 0.02\,\text{Å}$).

---

## Supported Systems & Features

- **Linear Polar / Stockmayer Fluids (`ABC`)**: Rigid triatomic linear dipoles with $+q, -q$ end-charges and central LJ site. Moves: Insertion, Deletion, Translation, Axis-aligned rotation.
- **Water Models (`H2O`)**: Rigid 3-site SPC/E water with non-linear 3D quaternion rotation moves.
- **Ionic Fluids (`RPM`, `PM`, `21PM`)**: Single and multi-component electrolytes with insertion, deletion, displacement, species mutation, and particle swapping.
- **Replica Exchange (`gcmc_re`)**: Parallel MPI-based temperature and chemical potential replica exchange.
- **External Potential Landscapes**: Hard walls, slit geometries, 9-3 LJ walls, and randomized sinusoidal/cosine charging potentials $\phi(z)$ for cDFT training.

---

## Installation & Setup (using `uv`)

This codebase is managed using [`uv`](https://github.com/astral-sh/uv).

### 1. Clone the repository
```bash
git clone https://github.com/annatbui/gcmc.git
cd gcmc
```

### 2. Create virtual environment and install dependencies
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

For MPI replica exchange support:
```bash
uv pip install -e ".[dev,mpi]"
```

---

## Usage

### Command Line Interface (CLI)
You can run a simulation from any directory containing an `input.yaml` configuration file:

```bash
gcmc -in path/to/simulation_dir
```

Or explicitly using the `v1` engine:
```bash
gcmc-v1 -in path/to/simulation_dir
```

### Python API
```python
from gcmc.v1 import (
    load_config,
    initialize_potentials,
    initialize_external_potentials,
    GCMC_FF_ABC_Simulation,
)

config = load_config("path/to/input.yaml")
potentials = initialize_potentials(config)
ext_potentials = initialize_external_potentials(config)

sim = GCMC_FF_ABC_Simulation(config, potentials, ext_potentials, input_folder=".")
sim.run_simulation()
```

### Post-Processing & Density Profiles
Compute spatial density profiles from the compressed trajectory `output.xyz.gz`:
```python
from gcmc.v1.utils.get_density_profile import read_extended_xyz, average_density_profiles

positions_list, lattice_vectors_list = read_extended_xyz("output.xyz.gz")
bin_centers, avg_density = average_density_profiles(positions_list, lattice_vectors_list, bins=100)
```

---

## Testing & Verification

The test suite validates pairwise potentials, quaternion rigid-body rotations, molecular geometries, and runs end-to-end simulations for dipoles, water, and electrolytes in **under 5 seconds**:

```bash
pytest tests/v1 -v
```

---

## Repository Structure

```
gcmc/
├── AGENTS.md               # Developer & AI agent reference guide with equations and module maps
├── pyproject.toml          # Modern packaging configuration (uv/pip)
├── LICENSE                 # GNU General Public License v3.0
├── src/
│   └── gcmc/
│       ├── __init__.py     # Package root
│       └── v1/             # Modularized v1 engine
│           ├── main.py
│           ├── potentials.py
│           ├── external_potentials.py
│           ├── gcmc_ff_molecule.py
│           ├── gcmc_ff.py
│           ├── tools.py
│           └── utils/
└── tests/
    └── v1/                 # Automated test suite and fast configurations
```

---

## Citation

If you use this code in your research, please cite:

1. **A. T. Bui, S. J. Cox**, *"Dielectrocapillarity for exquisite control of fluids"*, arXiv:2503.09855 (2025).
2. **A. T. Bui, S. J. Cox**, *"Learning classical density functionals for ionic fluids"*, *Phys. Rev. Lett.* **134**, 148001 (2025). [doi:10.1103/PhysRevLett.134.148001](https://doi.org/10.1103/PhysRevLett.134.148001)

---

## License

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License as published by the Free Software Foundation**, either version 3 of the License, or (at your option) any later version. See the [LICENSE](LICENSE) file for details.
