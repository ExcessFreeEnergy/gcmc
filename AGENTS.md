# AGENTS.md: Developer & Agent Reference Guide for `gcmc`

This document provides a comprehensive technical reference for the `gcmc` repository. It is designed to allow AI agents and developers to quickly understand the physical problem, theoretical framework, software architecture, file structure, and test suite without needing to re-analyze original research papers.

---

## 1. Domain Physics & Scientific Context

> [!IMPORTANT]
> **Tooling Rule**: Always use `uv` for Python environment management, package installation, and script/test execution (`uv venv`, `uv pip install`, `uv run pytest`, `uv run ...`). Do not use Conda or plain `pip`.

### 1.1 The Physical Problem: Dielectrocapillarity & Electrostriction
The `gcmc` code simulates polar, dielectric, and ionic fluids subjected to inhomogeneous electric fields and electric field gradients (EFGs), as described in the paper:
> **"Dielectrocapillarity for exquisite control of fluids"** (Anna T. Bui & Stephen J. Cox, 2025; arXiv:2503.09855).

Key physical phenomena investigated:
- **Dielectrophoretic Rise**: Neutral polar molecules experience a net dielectrophoretic force $f_{\rm DEP} \propto \nabla |E|^2$, driving the fluid towards regions of higher electric field strength.
- **Electrophoretic Rise**: Ionic fluids (restricted primitive model, RPM) experience net charge forces $f_{\rm EP} = qE$, driving anions and cations in opposite directions towards lower absolute field regions.
- **Electric-Field-Driven Critical Shift**: Sizable shifts in the liquid-vapor binodal and critical temperature $T_{\rm c}$ under microscopic EFGs.
- **Dielectrocapillarity & Controlled Capillary Condensation**: Reversible, tunable fluid uptake, hysteresis suppression, and programmable pore filling in confined nanoscale slits with applied electrostatic potentials $\phi(z)$.

---

## 2. Theoretical Framework: cDFT + LMFT + GCMC

### 2.1 Classical Density Functional Theory (cDFT)
Equilibrium structure and thermodynamics in an open system (grand canonical ensemble) are determined by minimizing the grand potential functional:
$$\Omega([\rho, \beta\phi], T) = \mathcal{F}_{\rm intr}^{\rm id}([\rho], T) + \mathcal{F}_{\rm intr}^{\rm ex}([\rho, \beta\phi], T) + \int dz \, \rho(z) [V_{\rm ext}(z) - \mu]$$
The equilibrium density $\rho(z)$ solves the Euler-Lagrange equation:
$$\rho(z) = \frac{\zeta}{\Lambda^3} \exp\left[-\beta V_{\rm ext}(z) + \beta \mu + c^{(1)}(z; [\rho, \beta\phi], T)\right]$$
where $c^{(1)} = -\delta \beta \mathcal{F}_{\rm intr}^{\rm ex} / \delta \rho(z)$ is the one-body direct correlation functional, and $n^{(1)} = \delta \mathcal{F}_{\rm intr}^{\rm ex} / \delta (\beta\phi(z))$ is the charge density functional.

### 2.2 Short-Range Coulomb Splitting (LMFT)
Direct machine learning of non-local long-range electrostatic functionals is intractable. The Coulomb potential is partitioned into a short-ranged (SR) reference contribution $v_0(r)$ and a smooth long-range (LR) contribution $v_1(r)$:
$$\frac{1}{r} = v_0(r) + v_1(r) \equiv \frac{\operatorname{erfc}(\kappa r)}{r} + \frac{\operatorname{erf}(\kappa r)}{r}$$
- For water (SPC/E) and the simple dipolar Stockmayer fluid: $\kappa^{-1} = 4.5\,\text{Å}$.
- For electrolytes (RPM): $\kappa^{-1} = 5.0\,\text{Å}$.

The SR reference fluid $v_0(r)$ has strictly local correlations, making it suitable for local neural functional learning. Long-range effects are restored analytically/mean-field via local molecular field theory (LMFT):
$$c^{(1)}(z) = c_{\rm R}^{(1)}(z; [\rho, \beta\phi_{\rm R}], T) - \beta\Delta\mu$$
$$n^{(1)}(z) = n_{\rm R}^{(1)}(z; [\rho, \beta\phi_{\rm R}], T)$$
$$\phi_{\rm R}(z) = \phi(z) + \int dz' \, n(z') v_1(|z - z'|)$$

### 2.3 Role of GCMC in Data Generation
1. **Training Data Generation**: `gcmc` generates reference training data for $c_{\rm R}^{(1)}$ and $n_{\rm R}^{(1)}$ across $\sim 2000$ randomized thermodynamic and external potential configurations ($T \in [250, 500]\,\text{K}$, $\mu/k_B \in [-5000, -1000]\,\text{K}$, $\phi(z) = \frac{\phi_0}{m}\cos(2\pi mz/L_z)$).
2. **Hybrid GCMC + MD Workflow**: Dense subcritical polar fluids have low GCMC insertion/deletion acceptance rates. To sample continuous high-resolution density profiles efficiently:
   - `gcmc` runs first to determine the average equilibrium particle number $N_{\rm ave} = \langle N \rangle_{\mu, V, T}$.
   - Canonical ($NVT$) Molecular Dynamics (LAMMPS with the `pair_lj_cut_coul_GT` pair style from LMFT) is initialized with $N_{\rm ave}$ molecules to sample smooth profiles $\rho_{\rm R}(z)$ and $n_{\rm R}(z)$ on a fine grid ($\Delta z = 0.02\,\text{Å}$).

---

## 3. Repository Architecture & File Mapping

The `gcmc` package is structured into versioned namespaces: `gcmc.v1` (the established baseline engine) and `gcmc.v2` (reserved for future major refactors/performance overhauls).

```
gcmc/
├── AGENTS.md                           # This developer & agent guide
├── pyproject.toml                      # Modern packaging configuration (uv/pip)
├── src/
│   └── gcmc/
│       ├── __init__.py                 # Root package; exposes v1 by default
│       └── v1/                         # Version 1 implementation
│           ├── __init__.py             # Exports simulation classes and helpers
│           ├── main.py                 # CLI entry point (gcmc, gcmc-v1) and runner
│           ├── read_input.py           # YAML input parser & validator
│           ├── constants.py            # Physical constants (kB, eV, e, N_A, etc.)
│           ├── tools.py                # Rigid molecule geometries & quaternion rotation math
│           ├── potentials.py           # Pair potentials (LJ, WCA, HS, HS+C, LJ+C with erfc)
│           ├── external_potentials.py  # External potentials (Slits, LJ93 walls, Cosine charges)
│           ├── molecule_base.py        # Base class for molecular GCMC (PBC, energy, logging)
│           ├── gcmc_ff_molecule.py     # Molecular simulations (ABC dipole, H2O, CO2)
│           ├── gcmc_ff.py              # Atomic & ionic simulations (SingleType, RPM TwoType)
│           ├── gcmc_re.py              # MPI-based replica exchange GCMC
│           └── utils/                  # Post-processing scripts (density & c1 profile extractors)
│               ├── get_density_profile.py
│               ├── get_profiles.py
│               ├── plot_Vext.py
│               └── plot_potential.py
└── tests/
    └── v1/                             # Test suite validating v1 functionality
        ├── conftest.py                 # Pytest fixtures and temporary run directories
        ├── test_potentials.py          # Unit tests: potential formulas & erfc splitting
        ├── test_rotations.py           # Unit tests: quaternion rotations & rigid constraints
        ├── test_dipole_ff.py           # Fast e2e test: ABC dipolar fluid (<10s)
        ├── test_rpm_ff.py              # Fast e2e test: RPM ionic fluid (<8s)
        ├── test_h2o_ff.py              # Fast e2e test: SPC/E water (<10s)
        ├── test_profiles_e2e.py        # Fast e2e test: density profile extraction (<15s)
        ├── test_configs/               # Miniature, fast YAML test configurations
        └── legacy_tests/               # Original full-length production configs (dipole, rpm, pm, 21pm)
```

---

## 4. Module & File Purposes in Detail

### Core Simulation Engines
- [`src/gcmc/v1/molecule_base.py`](file:///home/gauss/code/cdft_sim/gcmc/src/gcmc/v1/molecule_base.py):
  Provides `GCMCMoleculeBaseSimulation`. Implements vectorized pairwise minimum image distance matrices, intramolecular exclusions (excluding bonded sites from non-bonded pairwise sums), log-sum-exp numerical stability tricks, gzip-compressed extended XYZ trajectory streaming (`output.xyz.gz`), and log file generation.
- [`src/gcmc/v1/gcmc_ff_molecule.py`](file:///home/gauss/code/cdft_sim/gcmc/src/gcmc/v1/gcmc_ff_molecule.py):
  - `GCMC_FF_ABC_Simulation`: Linear rigid triatomic molecule with positive charge ($+q$), negative charge ($-q$), and central Lennard-Jones site ($A$). Moves: Insertion, Deletion, Translation, Axis-aligned Rotation.
  - `GCMC_FF_H2O_Simulation`: 3-site SPC/E water molecule with rigid geometry. Moves: Insertion, Deletion, Translation, 3D Quaternion Rotation.
  - `GCMC_FF_CO2_Simulation`: Linear triatomic CO2 molecule.
- [`src/gcmc/v1/gcmc_ff.py`](file:///home/gauss/code/cdft_sim/gcmc/src/gcmc/v1/gcmc_ff.py):
  - `GCMC_FF_SingleType_Simulation`: Single atomic/ionic species.
  - `GCMC_FF_TwoType_Simulation`: Binary mixtures/restricted primitive model (cations + anions). Supports particle swap moves and identity mutation moves in addition to standard GCMC moves. On-the-fly density histogram accumulation along $x$.
- [`src/gcmc/v1/gcmc_re.py`](file:///home/gauss/code/cdft_sim/gcmc/src/gcmc/v1/gcmc_re.py):
  Parallel replica exchange across temperature $T$ and chemical potential $\mu$ using `mpi4py`.

### Potentials & Mathematics
- [`src/gcmc/v1/potentials.py`](file:///home/gauss/code/cdft_sim/gcmc/src/gcmc/v1/potentials.py):
  - `LennardJonesPotential`: Truncated and shifted LJ potential.
  - `WCAPotential`: Weeks-Chandler-Andersen purely repulsive potential ($r_c = 2^{1/6}\sigma$).
  - `HardSpherePotential`: Infinite step barrier for $r < \sigma$.
  - `HardSphereCoulombPotential` (`HS+C`): Hard sphere + Gaussian-truncated Coulomb $v_0(r) = \frac{\operatorname{erfc}(r/\kappa^{-1})}{r}$.
  - `LennardJonesCoulombPotential` (`LJ+C`): Truncated/shifted LJ + Gaussian-truncated Coulomb $v_0(r)$.
- [`src/gcmc/v1/external_potentials.py`](file:///home/gauss/code/cdft_sim/gcmc/src/gcmc/v1/external_potentials.py):
  Defines external potential profiles acting on specific sites (e.g. hard walls, slit 9-3 LJ walls, and randomized cosine external charging potentials `TrainingPotentialWithChargeCos` used to generate neural cDFT training data).
- [`src/gcmc/v1/tools.py`](file:///home/gauss/code/cdft_sim/gcmc/src/gcmc/v1/tools.py):
  Stores canonical molecular templates (`SPCE_origin`, `ABC_origin`, `CO2_origin`) and implements quaternion multiplication, vector rotation, and random rotation generators.

---

## 5. Monte Carlo Acceptance Criteria Summary

For a system with volume $V$, inverse temperature $\beta = 1/(k_B T)$, and chemical potential $\mu$:

1. **Insertion Move** (attempting to add a particle/molecule at random position $\mathbf{r}_{\rm new}$):
   $$P_{\rm acc}(N \to N+1) = \min\left[1, \frac{V}{N+1} \exp\left(-\beta(\Delta E - \mu)\right)\right]$$
2. **Deletion Move** (attempting to remove random particle $i$):
   $$P_{\rm acc}(N \to N-1) = \min\left[1, \frac{N}{V} \exp\left(-\beta(\Delta E + \mu)\right)\right]$
3. **Displacement / Rotation Move**:
   $$P_{\rm acc}(\mathbf{r} \to \mathbf{r}') = \min\left[1, \exp(-\beta \Delta E)\right]$$
4. **Species Mutation Move** (type $1 \to 2$):
   $$P_{\rm acc}(N_1, N_2 \to N_1-1, N_2+1) = \min\left[1, \frac{N_1}{N_2 + 1} \exp\left(-\beta(\Delta E - \mu_2 + \mu_1)\right)\right]$$

---

## 6. Test Suite Existence & Objectives

The test suite in `tests/v1/` provides automated regression and verification:
- **Target Execution Time**: Under 60 seconds total.
- **Verification Goals**:
  1. *Unit Tests*: Validate analytical correctness of potential energies, Gaussian cutoffs, Coulomb splitting, quaternion transformations, and rigid body constraint preservation.
  2. *Molecular End-to-End Tests*: Run fast, miniature GCMC simulations for ABC dipoles, SPC/E water, and RPM electrolytes with non-empty trajectories, finite energies, and proper trajectory compression.
  3. *Post-Processing Pipeline Tests*: Verify that `get_profiles.py` correctly parses compressed trajectories and computes smooth, non-NaN density $\rho(z)$ and charge density $n(z)$ profiles.
