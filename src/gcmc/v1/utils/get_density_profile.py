import gzip
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def read_extended_xyz(file_path):
    """
    Function to read an extended XYZ file and extract positions and lattice vectors for each timestep.

    Parameters:
    - file_path (str): Path to the extended XYZ file.

    Returns:
    - positions_list (list of dict): List of dictionaries of particle positions for each timestep, keyed by species.
    - lattice_vectors_list (list of np.ndarray): List of arrays of lattice vectors for each timestep.
    """
    positions_list = []
    lattice_vectors_list = []

    with gzip.open(file_path, "rt") as f:
        lines = f.readlines()

        timestep_positions = defaultdict(list)
        timestep_lattice_vectors = None

        for line in lines:
            line = line.strip()

            if line.startswith("Step"):
                if timestep_lattice_vectors is not None:
                    positions_list.append(dict(timestep_positions))
                    lattice_vectors_list.append(timestep_lattice_vectors)
                    timestep_positions = defaultdict(list)
                lattice_str = line.split("Lattice=")[1].split("Properties")[0].strip('"').strip()
                lattice_values = lattice_str.split()
                timestep_lattice_vectors = np.array([float(val.strip('"')) for val in lattice_values]).reshape(3, 3)
            else:
                parts = line.split()
                if len(parts) == 4:  # assuming XYZ format with 3D coordinates
                    species = parts[0]
                    x, y, z = map(float, parts[1:4])  # assuming x, y, z are columns 2, 3, 4
                    timestep_positions[species].append([x, y, z])

        # Append the last timestep
        if timestep_lattice_vectors is not None:
            positions_list.append(dict(timestep_positions))
            lattice_vectors_list.append(timestep_lattice_vectors)

    return positions_list, lattice_vectors_list


def calculate_volume(lattice_vectors):
    """
    Function to calculate the volume of the simulation box from lattice vectors.

    Parameters:
    - lattice_vectors (np.ndarray): Array of lattice vectors (3x3 matrix).

    Returns:
    - volume (float): Volume of the simulation box.
    """
    volume = np.abs(np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2])))
    return volume


def calculate_density_profile(positions, lattice_vectors, bins=100):
    """
    Function to calculate the spatial density profile along the x direction for a single timestep.

    Parameters:
    - positions (np.ndarray): Array of particle positions (3D).
    - lattice_vectors (np.ndarray): Array of lattice vectors (3x3 matrix).
    - bins (int): Number of bins for the histogram along the x direction.

    Returns:
    - bin_centers (np.ndarray): Centers of the bins along the x direction.
    - density_profile (np.ndarray): Density profile along the x direction, normalized by volume.
    """
    x_positions = positions[:, 0]  # extract x coordinates

    # Calculate density profile
    counts, bin_edges = np.histogram(x_positions, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    volume = calculate_volume(lattice_vectors) / bins

    # Normalize density profile by volume
    density_profile = counts / volume

    return bin_centers, density_profile


def average_density_profiles(positions_list, lattice_vectors_list, bins=100):
    """
    Function to compute the average density profile over multiple timesteps for each species.

    Parameters:
    - positions_list (list of dict): List of dictionaries of particle positions for each timestep, keyed by species.
    - lattice_vectors_list (list of np.ndarray): List of arrays of lattice vectors for each timestep.
    - bins (int): Number of bins for the histogram along the x direction.

    Returns:
    - bin_centers (np.ndarray): Centers of the bins along the x direction.
    - average_density_profiles (dict): Average density profiles for each species, keyed by species.
    """
    species_density_profiles = defaultdict(list)

    for timestep_positions, lattice_vectors in zip(positions_list, lattice_vectors_list):
        for species, positions in timestep_positions.items():
            positions_array = np.array(positions)
            bin_centers, density_profile = calculate_density_profile(positions_array, lattice_vectors, bins=bins)
            species_density_profiles[species].append(density_profile)

    # Compute average density profile for each species
    average_density_profiles = {
        species: np.mean(profiles, axis=0) for species, profiles in species_density_profiles.items()
    }

    return bin_centers, average_density_profiles


def plot_density_profiles(bin_centers, average_density_profiles):
    """
    Function to plot the density profiles for each species.

    Parameters:
    - bin_centers (np.ndarray): Centers of the bins along the x direction.
    - average_density_profiles (dict): Average density profiles for each species, keyed by species.
    """
    plt.figure(figsize=(10, 6))
    params = {
        "axes.labelsize": 16,
        "axes.titlesize": 15,
    }
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(params)

    plt.tick_params(direction="in", which="minor", length=3)
    plt.tick_params(direction="in", which="major", length=5, labelsize=14)
    plt.grid(which="major", ls="dashed", dashes=(1, 3), lw=1, zorder=0)

    for species, density_profile in average_density_profiles.items():
        plt.plot(bin_centers, density_profile, label=f"Species {species}", lw=2.5)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$\rho(x)$")
    plt.legend()
    plt.tight_layout()
    plt.show()


def detect_bulk_plateaus(bin_centers, density_profile, window_size=7, gradient_tol=1e-3):
    """
    Algorithmic plateau detection to identify bulk liquid and vapor regions dynamically.
    Identifies contiguous spatial intervals where |d(rho)/dx| < gradient_tol and d^2(rho)/dx^2 ~ 0.

    Parameters:
        bin_centers (np.ndarray): Spatial coordinates along grid.
        density_profile (np.ndarray): Density profile rho(x).
        window_size (int): Moving filter window.
        gradient_tol (float): Gradient threshold for flatness.

    Returns:
        dict: {'plateau_mask': bool array, 'bulk_density': float, 'density_std': float}
    """
    dx = bin_centers[1] - bin_centers[0]
    grad = np.gradient(density_profile, dx)
    curv = np.gradient(grad, dx)

    # Moving standard deviation / flatness criterion
    flat_mask = (np.abs(grad) < gradient_tol) & (np.abs(curv) < (gradient_tol / dx))

    if np.any(flat_mask):
        bulk_density = float(np.median(density_profile[flat_mask]))
        density_std = float(np.std(density_profile[flat_mask]))
    else:
        # Fallback to mid-domain IQR
        mid = len(density_profile) // 2
        bulk_density = float(np.median(density_profile[max(0, mid - 10) : min(len(density_profile), mid + 10)]))
        density_std = float(np.std(density_profile[max(0, mid - 10) : min(len(density_profile), mid + 10)]))

    return {
        "plateau_mask": flat_mask,
        "bulk_density": bulk_density,
        "density_std": density_std,
    }


def fit_capillary_interface(bin_centers, density_profile):
    """
    Fits the liquid-vapor or wall-fluid interface using the hyperbolic tangent profile:
        rho(x) = (rho_l + rho_v)/2 - (rho_l - rho_v)/2 * tanh((x - x0) / d)

    Returns:
        dict: {'rho_l': float, 'rho_v': float, 'x0': float, 'width_d': float}
    """
    rho_max = float(np.max(density_profile))
    rho_min = float(np.min(density_profile))
    mid_rho = 0.5 * (rho_max + rho_min)

    # Estimate interface position x0
    idx_cross = np.argmin(np.abs(density_profile - mid_rho))
    x0_guess = bin_centers[idx_cross]
    d_guess = 1.0

    try:
        from scipy.optimize import curve_fit

        def tanh_func(x, r_l, r_v, x0, d):
            return 0.5 * (r_l + r_v) - 0.5 * (r_l - r_v) * np.tanh((x - x0) / np.maximum(d, 1e-4))

        popt, _ = curve_fit(
            tanh_func,
            bin_centers,
            density_profile,
            p0=[rho_max, rho_min, x0_guess, d_guess],
            bounds=([0.0, 0.0, bin_centers[0], 0.1], [2.0 * rho_max, rho_max, bin_centers[-1], 20.0]),
            maxfev=2000,
        )
        return {
            "rho_l": float(popt[0]),
            "rho_v": float(popt[1]),
            "x0": float(popt[2]),
            "width_d": float(popt[3]),
        }
    except Exception:
        return {
            "rho_l": rho_max,
            "rho_v": rho_min,
            "x0": float(x0_guess),
            "width_d": float(d_guess),
        }


# Example usage:
if __name__ == "__main__":
    file_path = "output.xyz.gz"  # replace with your extended XYZ file path
    positions_list, lattice_vectors_list = read_extended_xyz(file_path)
    bin_centers, average_density_profiles = average_density_profiles(positions_list, lattice_vectors_list, bins=800)
    plot_density_profiles(bin_centers, average_density_profiles)

    # Output the density profiles to density.out
    with open("density.out", "w") as f:
        for species, density_profile in average_density_profiles.items():
            for center, density in zip(bin_centers, density_profile):
                f.write(f"{species} {center} {density}\n")
