"""Unit tests for pair potential functions and Gaussian-truncated Coulomb splitting."""

import numpy as np
import scipy.constants as const
from scipy.special import erfc

from gcmc.v1.constants import very_large_number
from gcmc.v1.potentials import (
    HardSphereCoulombPotential,
    HardSpherePotential,
    LennardJonesCoulombPotential,
    LennardJonesPotential,
    WCAPotential,
    initialize_potentials,
)


def test_lennard_jones_potential():
    epsilon = 1.5
    sigma = 3.0
    rc = 10.0
    lj = LennardJonesPotential(epsilon=epsilon, sigma=sigma, rc=rc)

    # Minimum of unshifted LJ is at r = 2^(1/6) * sigma, energy is -epsilon
    r_min = (2.0 ** (1.0 / 6.0)) * sigma
    shift = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
    expected_e_min = -epsilon - shift

    assert np.isclose(lj.calculate(np.array([r_min]))[0], expected_e_min, rtol=1e-5)
    # Cutoff behavior
    assert lj.calculate(np.array([rc + 0.1]))[0] == 0.0


def test_wca_potential():
    epsilon = 2.0
    sigma = 2.5
    wca = WCAPotential(epsilon=epsilon, sigma=sigma)
    rc = (2.0 ** (1.0 / 6.0)) * sigma

    # Inside cutoff: purely repulsive
    assert wca.calculate(np.array([0.9 * sigma]))[0] > 0.0
    # At cutoff: should approach 0
    assert np.isclose(wca.calculate(np.array([rc]))[0], 0.0, atol=1e-10)
    # Beyond cutoff: 0
    assert wca.calculate(np.array([rc + 0.5]))[0] == 0.0


def test_hard_sphere_potential():
    sigma = 3.0
    hs = HardSpherePotential(sigma=sigma)

    assert hs.calculate(np.array([2.5]))[0] == very_large_number
    assert hs.calculate(np.array([3.5]))[0] == 0.0


def test_hard_sphere_coulomb_potential():
    diameter = 2.76
    epsilon = 1.0
    q1 = 1.0
    q2 = -1.0
    kappa_inv = 5.0
    hsc = HardSphereCoulombPotential(diameter, epsilon, q1, q2, kappa_inv)

    # Overlap
    assert hsc.calculate(np.array([2.0]))[0] == very_large_number

    # Beyond diameter: Gaussian truncated Coulomb v0(r) = prefactor * q1 * q2 * erfc(r/kappa_inv) / r
    r_test = 4.0
    prefactor = (const.elementary_charge) ** 2 / (4 * const.pi * const.epsilon_0 * 1e-10 * epsilon)
    expected_v0 = prefactor * q1 * q2 * erfc(r_test / kappa_inv) / r_test
    assert np.isclose(hsc.calculate(np.array([r_test]))[0], expected_v0, rtol=1e-6)


def test_lennard_jones_coulomb_potential():
    epsilon_lj = 0.4469407  # kcal/mol
    sigma_lj = 3.024
    rc = 10.0
    epsilon_c = 1.0
    q1 = 0.382
    q2 = -0.382
    kappa_inv = 4.5

    ljc = LennardJonesCoulombPotential(epsilon_lj, sigma_lj, rc, epsilon_c, q1, q2, kappa_inv)

    r_test = 5.0
    e_val = ljc.calculate(np.array([r_test]))[0]
    assert np.isfinite(e_val)
    # Beyond rc: 0
    assert ljc.calculate(np.array([12.0]))[0] == 0.0


def test_initialize_potentials():
    config = {
        "potential_pairs": {
            "A_A": {
                "type": "LJ+C",
                "epsilon_lj": 0.44,
                "sigma_lj": 3.0,
                "rc": 10.0,
                "epsilon_c": 1.0,
                "q1": 0.0,
                "q2": 0.0,
                "kappa_inv": 4.5,
            },
            "H_O": {
                "type": "HS+C",
                "diameter": 2.76,
                "epsilon": 1.0,
                "q1": 1.0,
                "q2": -1.0,
                "kappa_inv": 5.0,
            },
        }
    }
    p_dict = initialize_potentials(config)
    assert "A_A" in p_dict
    assert isinstance(p_dict["A_A"], LennardJonesCoulombPotential)
    assert "H_O" in p_dict
    assert isinstance(p_dict["H_O"], HardSphereCoulombPotential)
