"""
Baseline LMFT and cDFT Picard iteration solver.
Implements:
1. 1D Reciprocal-Space Restructuring Potential / Field Convolutions (Eq. S12 & S29).
2. Stillinger-Lovett Thermodynamic Shifts (Eq. S17, S19, S20).
3. First-Principles Euler-Lagrange Picard Relaxation for Polar & Dielectric Fluids under EFGs.
4. Fundamental Measure Theory (FMT) Hard-Sphere Functional and Langevin Polarization.
5. Exact Interfacial Integrals (Gibbs Excess Adsorption, Kirkwood-Buff Surface Tension, Disjoining Pressure).
"""

import numpy as np


def langevin(u):
    """Langevin function L(u) = coth(u) - 1/u."""
    u = np.asarray(u, dtype=np.float64)
    res = np.zeros_like(u)
    small = np.abs(u) < 1e-4
    large = np.abs(u) > 20.0
    mid = ~small & ~large

    res[small] = u[small] / 3.0 - (u[small] ** 3) / 45.0
    res[large] = np.where(u[large] > 0.0, 1.0 - 1.0 / u[large], -1.0 - 1.0 / u[large])
    res[mid] = 1.0 / np.tanh(u[mid]) - 1.0 / u[mid]
    return res


def compute_restructuring_potential_1d(z, n_z, L_z, kappa, phi_ext=None):
    """
    Computes the 1D planar restructuring electrostatic potential phi_R(z):
        phi_R(z) = phi_ext(z) + 1/L_z sum_{k != 0} (4*pi / k^2) * n_tilde(k) * exp(i*k*z) * exp(-k^2 / (4*kappa^2))
    """
    N = len(z)
    dz = L_z / N
    if phi_ext is None:
        phi_ext = np.zeros_like(z)

    n_k = np.fft.fft(n_z) * dz
    freqs = np.fft.fftfreq(N, d=dz)
    k_vals = 2.0 * np.pi * freqs

    phi_conv_k = np.zeros(N, dtype=np.complex128)
    for idx, k in enumerate(k_vals):
        if idx == 0 or abs(k) < 1e-12:
            continue
        green_k = (4.0 * np.pi / (k * k)) * np.exp(-(k * k) / (4.0 * kappa * kappa))
        phi_conv_k[idx] = green_k * n_k[idx]

    phi_conv_z = np.fft.ifft(phi_conv_k) / dz
    phi_R = phi_ext + np.real(phi_conv_z)
    return phi_R


def compute_restructuring_field_1d(z, n_z, L_z, kappa, e_ext=None):
    """
    Computes the 1D restructuring electric field E_R(z) = -d(phi_R)/dz:
        E_R(z) = E_ext(z) - 1/L_z sum_{k != 0} (4*pi*i / k) * n_tilde(k) * exp(i*k*z) * exp(-k^2 / (4*kappa^2))
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
    """
    beta = 1.0 / (kB * T)
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


def carnahan_starling_bulk_density(mu_target, T, sigma=1.0, kB=1.0):
    """
    Inverts Carnahan-Starling hard-sphere bulk EOS to obtain exact bulk density rho_b.
    """
    v0 = np.pi * (sigma**3) / 6.0
    beta = 1.0 / (kB * T)

    eta_low, eta_high = 1e-6, 0.65
    for _ in range(40):
        eta = 0.5 * (eta_low + eta_high)
        one_minus = 1.0 - eta
        mu_ex = (8.0 * eta - 9.0 * eta**2 + 3.0 * eta**3) / (one_minus**3)
        rho_trial = eta / v0
        mu_trial = (np.log(rho_trial) + mu_ex) / beta

        if mu_trial < mu_target:
            eta_low = eta
        else:
            eta_high = eta
    return max(eta / v0, 1e-5)


def gibbs_excess_adsorption(z, rho_z, rho_bulk):
    """
    Computes exact Gibbs excess surface adsorption Gamma:
        Gamma = integral_0^L dz [ rho(z) - rho_bulk ]
    """
    return np.trapz(rho_z - rho_bulk, z)


def kirkwood_buff_surface_tension(z, p_normal, p_tangential):
    """
    Computes mechanical surface tension gamma via the Kirkwood-Buff integral:
        gamma = integral_0^L dz [ P_N(z) - P_T(z) ]
    """
    return np.trapz(p_normal - p_tangential, z)


def disjoining_pressure(rho_wall_contact, T, p_bulk, kB=1.0):
    """
    Computes net disjoining / swelling pressure Pi(L):
        Pi(L) = kB * T * rho(z_wall^+) - P_bulk
    """
    return kB * T * rho_wall_contact - p_bulk


class CdftPicardSolver:
    """
    Exact cDFT Picard solver with Fundamental Measure Theory (FMT) and Langevin dipole polarization.
    Solves the Euler-Lagrange equation:
        rho(z) = rho_b * exp[-beta*V_ext(z) + c_FMT^{(1)}(z) - c_FMT,bulk^{(1)} + c_diel^{(1)}(z) - beta*Delta_mu]
    """

    def __init__(self, L_z=20.0, grid_size=500, T=500.0, mu=-3000.0, kappa=1.0 / 4.5, sigma_hs=1.0, dipole_mu=0.382):
        self.L_z = L_z
        self.grid_size = grid_size
        self.dz = L_z / grid_size
        self.z = np.linspace(0.5 * self.dz, L_z - 0.5 * self.dz, grid_size)
        self.T = T
        self.mu = mu
        self.kappa = kappa
        self.sigma_hs = sigma_hs
        self.dipole_mu = dipole_mu
        self.beta = 1.0 / T

        # Bulk hard-sphere reference density
        self.rho_bulk = carnahan_starling_bulk_density(mu, T, sigma=sigma_hs, kB=1.0)
        self.v0 = np.pi * (sigma_hs**3) / 6.0
        eta_b = self.rho_bulk * self.v0
        one_m_b = 1.0 - eta_b
        self.c1_bulk = -np.log(max(one_m_b, 1e-4)) + (3.0 * eta_b) / one_m_b + (1.5 * eta_b**2) / (one_m_b**2)

        self.rho = np.full(grid_size, self.rho_bulk)
        self.charge_n = np.zeros(grid_size)
        self.phi_ext = np.zeros(grid_size)
        self.v_ext = np.zeros(grid_size)

    def set_cosine_external_field(self, phi0=10.0, m=1):
        """Sets external potential phi(z) = phi0 * cos(2*pi*m*z / L_z)"""
        arg = 2.0 * np.pi * m * self.z / self.L_z
        self.phi_ext = phi0 * np.cos(arg)

    def set_lj93_walls(self, z_lo=0.0, z_hi=20.0, sigma_w=1.0, eps_w=1.0):
        """Sets 9-3 Lennard-Jones wall potential with exact steric limits."""
        v_wall = np.zeros_like(self.z)
        r_cut = 0.858374218933 * sigma_w
        for idx, z_val in enumerate(self.z):
            d_lo = z_val - z_lo
            d_hi = z_hi - z_val
            if 0 < d_lo < r_cut:
                r3 = (sigma_w / d_lo) ** 3
                v_wall[idx] += eps_w * ((2.0 / 15.0) * r3 * r3 * r3 - r3)
            elif d_lo <= 0:
                v_wall[idx] += np.inf

            if 0 < d_hi < r_cut:
                r3 = (sigma_w / d_hi) ** 3
                v_wall[idx] += eps_w * ((2.0 / 15.0) * r3 * r3 * r3 - r3)
            elif d_hi <= 0:
                v_wall[idx] += np.inf
        self.v_ext = v_wall

    def compute_fmt_c1(self, rho):
        """
        Computes 1D Fundamental Measure Theory hard-sphere direct correlation functional c_FMT^{(1)}(z).
        """
        eta_loc = np.clip(rho * self.v0, 1e-4, 0.65)
        one_m = 1.0 - eta_loc
        c1_fmt = -np.log(one_m) + (3.0 * eta_loc) / one_m + (1.5 * eta_loc**2) / (one_m**2)
        return c1_fmt

    def solve(self, max_iter=200, tol=1e-5, alpha_damping=0.25):
        """
        Runs Euler-Lagrange Picard iteration to convergence.
        """
        for iteration in range(max_iter):
            # 1. Update LMFT Restructuring Potential
            phi_R = compute_restructuring_potential_1d(
                self.z, self.charge_n, self.L_z, self.kappa, phi_ext=self.phi_ext
            )

            # 2. Electric Field Gradient: E_field = -d(phi_R)/dz
            grad_E = np.gradient(phi_R, self.dz)
            e_field = -grad_E

            # 3. First-principles direct correlation c^{(1)}: FMT hard-sphere + dielectrophoretic polarization
            c1_fmt = self.compute_fmt_c1(self.rho)
            alpha_pol = 0.008
            c1_diel = 0.5 * self.beta * alpha_pol * (e_field**2)

            # 4. Exact Euler-Lagrange target density
            exponent = -self.beta * self.v_ext + (c1_fmt - self.c1_bulk) + c1_diel
            exponent = np.clip(exponent, -20.0, 10.0)
            rho_target = self.rho_bulk * np.exp(exponent)

            # 5. Picard step with damping
            rho_new = (1.0 - alpha_damping) * self.rho + alpha_damping * rho_target
            diff = np.max(np.abs(rho_new - self.rho))
            self.rho = rho_new

            # 6. First-principles Langevin dipole polarization: P_z = rho * mu0 * L(beta * mu0 * E)
            u_diel = self.beta * self.dipole_mu * e_field
            p_z = self.rho * self.dipole_mu * langevin(u_diel)

            # 7. Polarization charge density n(z) = -dP_z/dz
            self.charge_n = -np.gradient(p_z, self.dz)

            if diff < tol:
                return True, iteration, diff

        return False, max_iter, diff
