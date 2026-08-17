"""
GCMC simulation for fluids with short-ranged potentials
Copyright (C) 2024  Anna Bui

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

# Origin Positions for Molecules
SPCE_origin = np.array(
    [
        [1.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.000000],
        [1.33331324756823743627, 0.9428161427317179, 0.000000],
    ]
)

CO2_origin = np.array([[1.0000, 1.000000, 1.160000], [1.000000, 1.000000, 0.000000], [1.000000, 1.000000, 2.320000]])

ABC_origin = np.array([[0.0000, 1.0000, 0.000000], [0.000000, 1.050, 0.000000], [0.000000, 0.950, 0.00000]])

AB_origin = np.array(
    [
        [0.0000, 1.0000, 0.000000],
        [0.000000, 1.000, 0.250000],
    ]
)

EMI_origin = np.array([[0.000, -0.527, 1.365], [0.000, 1.641, 2.987], [0.000, 0.187, -2.389]])


# Rotation Utilities
def genrot(alpha, beta, gamma):
    """
    Rotates a 3D vector by the angles alpha, beta, and gamma
    """
    # Precompute sin and cos values
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    # Directly compute the combined rotation matrix
    R = np.array(
        [
            [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
            [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
            [-sb, cb * sg, cb * cg],
        ]
    )

    return R


def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion to a 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions q1 and q2.
    q1, q2 = [w, x, y, z]
    Returns the product quaternion.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
        ]
    )


def quaternion_rotate_vector(q, v):
    """
    Rotates a vector v using quaternion q.
    q = [w, x, y, z]
    v = [vx, vy, vz]
    Returns the rotated vector.
    """
    w, x, y, z = q
    v_q = np.array([0, *v])  # Represent vector as a pure quaternion
    q_conj = np.array([w, -x, -y, -z])  # Conjugate of quaternion q

    # Perform quaternion multiplication: q * v_q * q_conj
    rotated_v = quaternion_multiply(quaternion_multiply(q, v_q), q_conj)
    return rotated_v[1:]  # Return the vector part


def sample_random_quaternion():
    """
    Samples a random quaternion uniformly on the 4D hypersphere.
    """
    u1, u2, u3 = np.random.rand(3)
    w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([w, x, y, z])


# Rotation Functions


def RotMove_init(pos, MAXANG=np.pi, MAXCOS=1):
    """
    Performs a random rotation of the water molecule

    pos    = position matrix of water molecule
    MAXANG = max angle to rotate through

    returns the modified positions
    """

    randf = np.random.rand(3)
    alpha = (2 * randf[0] - 1) * MAXANG
    cosbeta = (2 * randf[1] - 1) * MAXCOS
    gamma = (2 * randf[2] - 1) * MAXANG

    beta = np.arccos(cosbeta)

    # Generate rotation matrix
    Rot = genrot(alpha, beta, gamma)

    # Reference position of the first atom
    refpos = pos[0, :]

    # Shift positions to origin, rotate, and shift back
    newpos = (pos - refpos) @ Rot.T + refpos

    return newpos


def RotMove_init_linear(pos):
    """
    Performs a random rotation for a linear molecule like CO2.

    pos    = position matrix of the linear molecule
    MAXANG = max angle to rotate through

    returns the modified positions
    """
    # Generate two random angles
    randf = np.random.rand(2)
    theta = 2 * np.pi * randf[0]  # Full rotation around Z-axis
    phi = np.arccos(2 * randf[1] - 1)  # Random tilt for linear axis

    # Generate rotation matrix for linear molecule
    ca, sa = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)

    # Rotation matrix specific for a linear molecule
    R = np.array([[cp, -sp * sa, sp * ca], [sp * sa, cp * ca, -sp * ca], [-sp, sp * sa, cp]])

    # Reference position of the first atom
    refpos = pos[0, :]

    # Shift positions to origin, rotate, and shift back
    newpos = (pos - refpos) @ R.T + refpos

    return newpos


def RotMove_shift_linear(pos):
    """
    Performs a random rotation of a linear molecule like CO2, then shifts.
    Ensures uniform sampling of orientations on a sphere using quaternions.

    pos = position matrix of the linear molecule (shape: Nx3)
    returns the modified positions (shape: Nx3)
    """
    # Step 1: Sample a random quaternion and convert to a rotation matrix
    q = sample_random_quaternion()
    R = quaternion_to_rotation_matrix(q)

    # Step 2: Rotate and shift the molecule
    refpos = pos[0, :]  # Center the molecule at the origin
    shifted_pos = pos - refpos
    rotated_pos = shifted_pos @ R.T  # Apply the rotation
    shift_vector = np.array([10.0, 10.0, 10.0])  # Arbitrary shift vector
    newpos = rotated_pos + refpos + shift_vector

    return newpos


def RotMove_shift_non_linear(pos):
    """
    Applies a random rotation to a non-linear molecule and shifts it.
    Uses quaternion-based rotations for uniform sampling.

    pos = position matrix of the molecule (shape: Nx3)
    Returns the modified positions (shape: Nx3).
    """
    # Step 1: Sample a random quaternion
    q = sample_random_quaternion()

    # Step 2: Rotate each atom using the quaternion
    refpos = np.mean(pos, axis=0)  # Geometric center of the molecule
    shifted_pos = pos - refpos  # Shift to center
    rotated_pos = np.array([quaternion_rotate_vector(q, atom) for atom in shifted_pos])

    # Step 3: Shift the molecule to a new location
    shift_vector = np.array([10.0, 10.0, 10.0])  # Arbitrary shift vector
    newpos = rotated_pos + refpos + shift_vector

    return newpos


# Random Molecule Generation


def generate_random_linear_triatomic(bond_length):
    """
    Generates random positions for a linear CO2 molecule centered at the origin
    with the specified bond length and uniformly distributed orientation.

    Parameters:
    bond_length (float): Distance between the carbon and each oxygen atom.

    Returns:
    np.ndarray: A 3x3 array with positions of C, C, and O atoms.
    """

    # Generate a random orientation vector on the sphere
    z = 2 * np.random.rand() - 1  # Random z component in range [-1, 1]
    theta = 2 * np.pi * np.random.rand()  # Azimuthal angle in range [0, 2*pi]

    r = np.sqrt(1 - z**2)  # Radius in the xy-plane for a unit vector

    # Orientation vector components
    orientation_vec = np.array([r * np.cos(theta), r * np.sin(theta), z])

    # Define positions based on the bond length
    # Carbon atom is at the origin
    carbon_pos = np.array([2.0, 2.0, 2.0])

    # Oxygen atoms positioned along the orientation vector
    oxygen_pos1 = carbon_pos + bond_length * orientation_vec
    oxygen_pos2 = carbon_pos - bond_length * orientation_vec

    # Step 3: Combine positions into a single array
    molecule_pos = np.vstack([carbon_pos, oxygen_pos1, oxygen_pos2])

    return molecule_pos
