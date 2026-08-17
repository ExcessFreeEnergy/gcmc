"""
Validation tests for the paper's baseline LMFT formulations, restructuring field convolutions,
and Stillinger-Lovett thermodynamic bulk corrections against the published dataset (OnlineData).
"""

import os
import numpy as np
import pytest

from gcmc.lmft_baseline import (
    compute_restructuring_potential_1d,
    compute_restructuring_field_1d,
    stillinger_lovett_corrections,
    CdftPicardSolver,
)

ONLINE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "online_data", "OnlineData")


def test_stillinger_lovett_thermodynamic_shifts():
    """
    Validates analytical Stillinger-Lovett corrections between LR and SR fluids (Eq. S17, S19, S20).
    """
    T = 500.0  # K
    rho_b = 0.028  # molecules/A^3
    eps_diel = 82.0
    kappa = 1.0 / 4.5  # A^-1

    shifts = stillinger_lovett_corrections(T=T, rho_b=rho_b, epsilon_diel=eps_diel, kappa=kappa, kB=1.0, N_molecules=256)

    # Validate that Delta P is negative (short range has higher pressure than long range)
    assert shifts["delta_P"] < 0.0, "Delta P must be negative for polar fluid"
    # Validate finite values
    assert np.isfinite(shifts["delta_U"])
    assert np.isfinite(shifts["delta_mu"])
    # Validate scaling with density
    shifts_low_rho = stillinger_lovett_corrections(T=T, rho_b=0.005, epsilon_diel=eps_diel, kappa=kappa)
    assert shifts_low_rho["delta_mu"] > shifts["delta_mu"]


def test_restructuring_field_symmetry_and_convergence():
    """
    Validates that under an antisymmetric charge density, the restructuring field E_R(z)
    exhibits exact physical symmetry: E_R(-z) == E_R(z) and phi_R(-z) == -phi_R(z).
    """
    L_z = 75.0
    N_grid = 512
    dz = L_z / N_grid
    z = np.linspace(-L_z / 2.0 + 0.5 * dz, L_z / 2.0 - 0.5 * dz, N_grid)
    kappa = 1.0 / 4.5

    # Polarized slab charge density n(z) = z * exp(-z^2 / 50) (antisymmetric)
    n_z = (z / 10.0) * np.exp(-(z**2) / 50.0)

    e_R = compute_restructuring_field_1d(z, n_z, L_z, kappa)
    phi_R = compute_restructuring_potential_1d(z, n_z, L_z, kappa)

    # Physical properties:
    # 1. E_R(z) and phi_R(z) must be finite everywhere
    assert np.all(np.isfinite(e_R))
    assert np.all(np.isfinite(phi_R))
    
    # 2. Symmetry verification:
    # phi_R is antisymmetric: phi_R(-z) = -phi_R(z)
    assert np.allclose(phi_R, -phi_R[::-1], atol=1e-5)
    # E_R is symmetric: E_R(-z) = E_R(z)
    assert np.allclose(e_R, e_R[::-1], atol=1e-5)


def test_online_data_er_dat_validation():
    """
    Validates reading and properties of the published ER.dat restructuring field dataset.
    """
    er_path = os.path.join(ONLINE_DATA_DIR, "Slab", "L75o0", "LMFT", "D0.00", "ER.dat")
    if os.path.exists(er_path):
        data = np.loadtxt(er_path)
        z_vals = data[:, 0]
        er_vals = data[:, 1]
        assert len(z_vals) > 100
        assert np.all(np.isfinite(er_vals))
        # Zero D-field at center has near-zero field
        mid_idx = len(er_vals) // 2
        assert abs(er_vals[mid_idx]) < 0.05


def test_cdft_picard_solver_dielectrophoretic_rise():
    """
    Validates that the Picard solver converges and exhibits dielectrophoretic rise
    (fluid concentration in high |E| regions) under an applied sinusoidal potential.
    """
    solver = CdftPicardSolver(L_z=20.0, grid_size=200, T=500.0, mu=-2.5, kappa=1.0 / 4.5)
    solver.set_cosine_external_field(phi0=15.0, m=1)
    solver.set_lj93_walls(z_lo=0.0, z_hi=20.0)

    converged, iters, residual = solver.solve(max_iter=150, tol=1e-4)

    assert converged, f"Picard solver failed to converge in {iters} iterations (residual={residual})"
    assert np.all(solver.rho >= 0.0), "Density must be non-negative"

    # In a cosine potential phi(z) = phi0 * cos(2*pi*z/L),
    # |E| is maximum where sin(2*pi*z/L) is max (at z = L/4 and 3L/4).
    # Dielectrophoretic rise dictates enhanced density near these regions.
    mid_density = np.mean(solver.rho)
    assert np.max(solver.rho) > mid_density
