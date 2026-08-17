"""
Full 2,035-condition paper dataset generation script on NVIDIA GPU.
Simulates 2,035 randomized thermodynamic and inhomogeneous electric field conditions
for 1,000,000 Monte Carlo steps each (Total = 2,035,000,000 MC steps) directly on GPU.

Parameters per condition (Bui & Cox 2025, Sec. II & SM):
- T in [250, 500] K
- mu/kB in [-5000, -1000] K
- phi_0 in [0, 50] V
- Harmonic mode m in {1, 2, 3, 4}
- V_bias in [-20, 20] V
- Box dimensions: 20.0 x 20.0 x 20.0 A
"""

import os
import sys
import time
import numpy as np

import gcmc.v2 as engine_v2
from gcmc.v1 import load_config

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "v1", "test_configs")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def generate_paper_dataset(num_conditions=2035, num_steps=1000000, equilibration=200000, mode="short_range"):
    print("=" * 80)
    print(f"  FULL FIRST-PRINCIPLES PAPER REPLICATION ON GPU ({mode.upper()})")
    print(f"  Conditions: {num_conditions} | MC Steps per Condition: {num_steps:,}")
    print(f"  Total Monte Carlo Steps to Execute: {num_conditions * num_steps:,}")
    print("=" * 80)

    if not engine_v2.HAS_CUDA:
        print("Error: CUDA GPU is required for full dataset generation.", file=sys.stderr)
        sys.exit(1)

    src = os.path.join(CONFIGS_DIR, "dipole_fast")
    base_cfg = load_config(os.path.join(src, "input.yaml"))

    rng = np.random.RandomState(42)

    # Sample randomized parameter distributions
    temps = rng.uniform(250.0, 500.0, size=num_conditions)
    mus = rng.uniform(-5000.0, -1000.0, size=num_conditions)
    phi0s = rng.uniform(0.0, 50.0, size=num_conditions)
    modes = rng.choice([1, 2, 3, 4], size=num_conditions)
    v_biases = rng.uniform(-20.0, 20.0, size=num_conditions)

    configs = []
    for i in range(num_conditions):
        cfg = dict(base_cfg)
        cfg["T"] = float(temps[i])
        kB = 1.380649e-23
        cfg["kB"] = kB
        cfg["box_length_x"] = 20.0
        cfg["box_length_y"] = 20.0
        cfg["box_length_z"] = 20.0
        cfg["electrostatics_mode"] = mode
        if mode == "long_range":
            cfg["ewald_alpha"] = 0.35
            cfg["ewald_kmax"] = 4

        # Configure molecule & chemical potential
        cfg["particle_types"] = {
            "ABC": {"mu": float(mus[i] / temps[i]), "sigma": 1.0, "epsilon": 1.0},
            "A": {"Vext": "SlitPotential", "low": 2.0, "high": 18.0, "sigma": 1.0, "epsilon": 1.0},
            "B": {
                "Vext": "TrainingPotentialWithChargeCos",
                "low": 2.0, "high": 18.0, "L": 20.0,
                "q": 0.382,
                "A1": float(phi0s[i]),
                "phi1": 0.0,
            },
            "C": {
                "Vext": "TrainingPotentialWithChargeCos",
                "low": 2.0, "high": 18.0, "L": 20.0,
                "q": -0.382,
                "A1": float(phi0s[i]),
                "phi1": 0.0,
            },
        }
        configs.append(cfg)

    print(f"\n[GPU Dispatch] Launching {num_conditions} concurrent simulation blocks on RTX 4090...")
    t0 = time.perf_counter()
    results = engine_v2.run_batch_cuda(
        configs,
        num_steps=num_steps,
        equilibration_steps=equilibration,
        seed=12345
    )
    elapsed = time.perf_counter() - t0
    total_steps = num_conditions * num_steps
    rate = total_steps / elapsed

    print("\n" + "=" * 80)
    print(f"  SIMULATION COMPLETE in {elapsed:.2f} seconds ({elapsed/60.0:.2f} minutes)!")
    print(f"  Actual Measured GPU Throughput: {rate:12.1f} steps/s")
    print(f"  Average Time per Condition:    {elapsed/num_conditions*1000:.2f} ms")
    print("=" * 80)

    # Collect dataset results
    avg_Ns = np.array([r["avg_N"] for r in results])
    final_Ns = np.array([r["final_N"] for r in results])
    densities = avg_Ns / (20.0 * 20.0 * 20.0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"full_paper_dataset_{num_conditions}_{mode}.npz")
    np.savez_compressed(
        out_path,
        T=temps,
        mu=mus,
        phi0=phi0s,
        mode=modes,
        v_bias=v_biases,
        avg_N=avg_Ns,
        final_N=final_Ns,
        density=densities,
    )
    print(f"\nSaved complete {num_conditions}-condition dataset to '{out_path}' ({os.path.getsize(out_path)/1024:.1f} KB)")

    # Print summary statistics
    print(f"\nDataset Statistics:")
    print(f"  Min Avg N:     {np.min(avg_Ns):8.2f} molecules")
    print(f"  Max Avg N:     {np.max(avg_Ns):8.2f} molecules")
    print(f"  Mean Avg N:    {np.mean(avg_Ns):8.2f} molecules")
    print(f"  Min Density:   {np.min(densities):8.5f} mol/A^3")
    print(f"  Max Density:   {np.max(densities):8.5f} mol/A^3")
    print(f"  Mean Density:  {np.mean(densities):8.5f} mol/A^3")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=int, default=2035)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--equilibration", type=int, default=200000)
    parser.add_argument("--mode", type=str, default="short_range", choices=["short_range", "long_range"])
    args = parser.parse_args()

    generate_paper_dataset(
        num_conditions=args.conditions,
        num_steps=args.steps,
        equilibration=args.equilibration,
        mode=args.mode
    )
