"""Unit tests for rigid molecular geometry transformations, rotations, and PBC handling."""

import numpy as np
import pytest

from gcmc.v1 import tools as tls


def test_quaternion_multiplication():
    # Identity quaternion [1, 0, 0, 0]
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    q_rot = np.array([0.70710678, 0.70710678, 0.0, 0.0])

    q_res = tls.quaternion_multiply(q_id, q_rot)
    assert np.allclose(q_res, q_rot)


def test_quaternion_rotation_vector():
    # 90 degree rotation about Z axis
    theta = np.pi / 2.0
    q_z = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
    v = np.array([1.0, 0.0, 0.0])

    v_rot = tls.quaternion_rotate_vector(q_z, v)
    assert np.allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-6)


def test_linear_molecule_rotation_invariance():
    bond_length = 0.5
    mol = tls.generate_random_linear_triatomic(bond_length)

    # Check initial geometry: A is at origin, B is at +bond_length, C is at -bond_length
    dist_AB = np.linalg.norm(mol[1] - mol[0])
    dist_AC = np.linalg.norm(mol[2] - mol[0])
    dist_BC = np.linalg.norm(mol[2] - mol[1])

    assert np.isclose(dist_AB, bond_length, atol=1e-6)
    assert np.isclose(dist_AC, bond_length, atol=1e-6)
    assert np.isclose(dist_BC, 2 * bond_length, atol=1e-6)

    # Apply multiple random linear shift rotations
    for _ in range(10):
        mol_rot = tls.RotMove_shift_linear(mol)
        d_AB = np.linalg.norm(mol_rot[1] - mol_rot[0])
        d_AC = np.linalg.norm(mol_rot[2] - mol_rot[0])
        d_BC = np.linalg.norm(mol_rot[2] - mol_rot[1])

        assert np.isclose(d_AB, bond_length, atol=1e-5)
        assert np.isclose(d_AC, bond_length, atol=1e-5)
        assert np.isclose(d_BC, 2 * bond_length, atol=1e-5)


def test_water_molecule_rotation_invariance():
    # SPC/E origin: mol[0] = O, mol[1] = H1, mol[2] = H2
    mol_init = tls.SPCE_origin.copy()
    init_d_OH1 = np.linalg.norm(mol_init[1] - mol_init[0])
    init_d_OH2 = np.linalg.norm(mol_init[2] - mol_init[0])
    init_d_H1H2 = np.linalg.norm(mol_init[2] - mol_init[1])

    mol_rot = mol_init.copy()
    for _ in range(10):
        mol_rot = tls.RotMove_shift_non_linear(mol_rot)
        d_OH1 = np.linalg.norm(mol_rot[1] - mol_rot[0])
        d_OH2 = np.linalg.norm(mol_rot[2] - mol_rot[0])
        d_H1H2 = np.linalg.norm(mol_rot[2] - mol_rot[1])

        assert np.isclose(d_OH1, init_d_OH1, atol=1e-5)
        assert np.isclose(d_OH2, init_d_OH2, atol=1e-5)
        assert np.isclose(d_H1H2, init_d_H1H2, atol=1e-5)
