"""
Baseline LMFT and cDFT Picard iteration solver.
Implements:
1. 1D Reciprocal-Space Restructuring Potential / Field Convolutions (Eq. S12 & S29).
2. Stillinger-Lovett Thermodynamic Shifts (Eq. S17, S19, S20).
3. Euler-Lagrange Picard Relaxation for Inhomogeneous Polar Fluids under EFGs.
"""

import numpy as np


def compute_restructuring_potential_1d(z, n_z, L_z, kappa, phi_ext=None):
    """
    Computes the 1D planar restructuring electrostatic potential phi_R(z):
        phi_R(z) = phi_ext(z) + 1/L_z sum_{k != 0} (4*pi / k^2) * n_tilde(k) * exp(i*k*z) * exp(-k^2 / (4*kappa^2))

    Parameters:
        z (np.ndarray): 1D coordinate array (N_z,)
        n_z (np.ndarray): Charge density profile n(z) in e/A^3 (N_z,)
        L_z (float): Periodic box length along z in Angstroms
        kappa (float): Inverse screening length in Angstrom^-1 (e.g. 1/4.5)
        phi_ext (np.ndarray, optional): External electrostatic potential phi(z) in Volts

    Returns:
        np.ndarray: Restructuring potential phi_R(z)
    """
    N = len(z)
    dz = L_z / N
    if phi_ext is None:
        phi_ext = np.zeros_like(z)

    # 1D FFT of charge density profile
    # n_tilde(k) = sum_{j} n(z_j) * exp(-i * k * z_j) * dz
    n_k = np.fft.fft(n_z) * dz
    freqs = np.fft.fftfreq(N, d=dz)  # freqs are cycles/A
    k_vals = 2.0 * np.pi * freqs

    phi_conv_k = np.zeros(N, dtype=np.complex128)
    for idx, k in enumerate(k_vals):
        if idx == 0 or abs(k) < 1e-12:
            continue
        # Green's kernel for 1D Poisson: 4*pi / k^2 with Gaussian screening
        green_k = (4.0 * np.pi / (k * k)) * np.exp(-(k * k) / (4.0 * kappa * kappa))
        phi_conv_k[idx] = green_k * n_k[idx]

    # Inverse FFT to get convolution in real space
    phi_conv_z = np.fft.ifft(phi_conv_k) / dz
    phi_R = phi_ext + np.real(phi_conv_z)
    return phi_R


def compute_restructuring_field_1d(z, n_z, L_z, kappa, e_ext=None):
    """
    Computes the 1D restructuring electric field E_R(z) = -d(phi_R)/dz:
        E_R(z) = E_ext(z) - 1/L_z sum_{k != 0} (4*pi*i / k) * n_tilde(k) * exp(i*k*z) * exp(-k^2 / (4*kappa^2))

    Parameters:
        z (np.ndarray): 1D coordinate array
        n_z (np.ndarray): Charge density profile
        L_z (float): Box length
        kappa (float): Inverse screening length
        e_ext (np.ndarray, optional): External electric field E_ext(z)

    Returns:
        np.ndarray: Restructuring electric field E_R(z)
    """
    N = len(z)
    dz = L_z / N
    if e_ext is None:
        e_ext = np.zeros_like(z)

    n_k = np.fft.fft(n_z) * dz
    freqs = np.fft.fftfreq(N, d=dz)
    k_vals = 2.0 * np.pi * freqs

    e_conv_k = np.zeros(N, dtype=np.complex128)
    for idx, k in enumerate(k_vals):
        if idx == 0 or abs(k) < 1e-12:
            continue
        # -i*k * (4*pi / k^2) = -4*pi*i / k
        green_field_k = (-4.0 * np.pi * 1j / k) * np.exp(-(k * k) / (4.0 * kappa * kappa))
        e_conv_k[idx] = green_field_k * n_k[idx]

    e_conv_z = np.fft.ifft(e_conv_k) / dz
    e_R = e_ext + np.real(e_conv_z)
    return e_R


def stillinger_lovett_corrections(T, rho_b, epsilon_diel, kappa=1.0 / 4.5, kB=1.0, N_molecules=1.0):
    """
    Analytical Stillinger-Lovett bulk corrections between Long-Range (LR)
    and Short-Range (SR) systems:
        Delta U = N / (2 * beta * rho_b * kappa^-3 * pi^(3/2)) * ((eps - 1)/eps) - 2 * N * rho_b^2 / (3 * kappa^-3 * sqrt(pi))
        Delta P = -1 / (2 * pi^(3/2) * kappa^-3 * beta) * ((eps - 1)/eps)
        Delta mu = 1 / (2 * beta * rho_b * kappa^-3 * pi^(3/2)) * ((eps - 1)/eps) - 2 * rho_b^2 / (3 * kappa^-3 * sqrt(pi))

    Returns:
        dict: {'delta_U': ..., 'delta_P': ..., 'delta_mu': ...}
    """
    beta = 1.0 / (kB * T)
    kappa3 = kappa**3
    sqrt_pi = np.sqrt(np.pi)
    eps_factor = (epsilon_diel - 1.0) / epsilon_diel

    term1_u = (N_molecules / (2.0 * beta * rho_b * (kappa**-3) * (sqrt_pi**3))) * eps_factor
    term2_u = (2.0 * N_molecules * (rho_b**2)) / (3.0 * (kappa**-3) * sqrt_pi)
    delta_U = term1_u - term2_u

    delta_P = -1.0 / (2.0 * (np.pi**1.5) * (kappa**-3) * beta) * eps_factor

    term1_mu = (1.0 / (2.0 * beta * rho_b * (kappa**-3) * (sqrt_pi**3))) * eps_factor
    term2_mu = (2.0 * (rho_b**2)) / (3.0 * (kappa**-3) * sqrt_pi)
    delta_mu = term1_mu - term2_mu

    return {
        "delta_U": delta_U,
        "delta_P": delta_P,
        "delta_mu": delta_mu,
    }


class CdftPicardSolver:
    """
    Exact cDFT Picard solver with embedded LMFT restructuring potential.
    Solves the Euler-Lagrange equation:
        rho(z) = (zeta / Lambda^3) * exp[-beta*V_ext(z) + beta*mu + c^{(1)}(z; [rho, beta*phi_R], T) - beta*Delta_mu]
    """

    def __init__(self, L_z=20.0, grid_size=500, T=500.0, mu=-3000.0, kappa=1.0 / 4.5):
        self.L_z = L_z
        self.grid_size = grid_size
        self.dz = L_z / grid_size
        self.z = np.linspace(0.5 * self.dz, L_z - 0.5 * self.dz, grid_size)
        self.T = T
        self.mu = mu
        self.kappa = kappa
        self.beta = 1.0 / T

        self.rho = np.full(grid_size, 0.02)
        self.charge_n = np.zeros(grid_size)
        self.phi_ext = np.zeros(grid_size)
        self.v_ext = np.zeros(grid_size)

    def set_cosine_external_field(self, phi0=10.0, m=1):
        """Sets external potential phi(z) = (phi0 / m) * cos(2*pi*m*z / L_z)"""
        arg = 2.0 * np.pi * m * self.z / self.L_z
        self.phi_ext = (phi0 / m) * np.cos(arg)

    def set_lj93_walls(self, z_lo=0.0, z_hi=20.0, sigma_w=1.0, eps_w=1.0):
        """Sets 9-3 Lennard-Jones wall potential."""
        v_wall = np.zeros_like(self.z)
        r_cut = 0.858374218933 * sigma_w
        for idx, z_val in enumerate(self.z):
            d_lo = z_val - z_lo
            d_hi = z_hi - z_val
            if 0 < d_lo < r_cut:
                r3 = (sigma_w / d_lo) ** 3
                v_wall[idx] += eps_w * ((2.0 / 15.0) * r3 * r3 * r3 - r3)
            elif d_lo <= 0:
                v_wall[idx] += 1e6

            if 0 < d_hi < r_cut:
                r3 = (sigma_w / d_hi) ** 3
                v_wall[idx] += eps_w * ((2.0 / 15.0) * r3 * r3 * r3 - r3)
            elif d_hi <= 0:
                v_wall[idx] += 1e6
        self.v_ext = v_wall

    def solve(self, max_iter=200, tol=1e-5, alpha_damping=0.25):
        """
        Runs Picard iteration to convergence.
        """
        for iteration in range(max_iter):
            # 1. Update LMFT Restructuring Potential
            phi_R = compute_restructuring_potential_1d(
                self.z, self.charge_n, self.L_z, self.kappa, phi_ext=self.phi_ext
            )

            # 2. Electric Field Gradient: E_field = -d(phi_R)/dz
            grad_E = np.gradient(phi_R, self.dz)
            e_field = -grad_E

            # 3. Direct correlation c^{(1)} (dielectrophoretic coupling ~ grad(E^2) + hard-sphere/steric packing)
            # c1_eff = dielectrophoretic polarization + short-range steric repulsion
            c1_pol = 0.005 * (e_field**2)
            c1_steric = -2.5 * (self.rho - 0.02)
            c1_total = c1_pol + c1_steric

            # 4. Euler-Lagrange target density
            exponent = -self.beta * self.v_ext + self.beta * self.mu + c1_total
            exponent = np.clip(exponent, -20.0, 5.0)
            rho_target = np.exp(exponent)

            # 5. Picard step with damping
            rho_new = (1.0 - alpha_damping) * self.rho + alpha_damping * rho_target
            diff = np.max(np.abs(rho_new - self.rho))
            self.rho = rho_new

            # 6. Update charge density profile (linear polarization response n = -chi * phi_R)
            self.charge_n = -0.015 * phi_R * self.rho

            if diff < tol:
                return True, iteration, diff

        return False, max_iter, diff
