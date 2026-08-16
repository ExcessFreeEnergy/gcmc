"""Benchmark script to measure GCMC sampling throughput and scaling."""

import time
import os
import shutil
import tempfile
import numpy as np

from gcmc.v1 import load_config
from gcmc.v1.main import run_simulation_job

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(TESTS_DIR, "test_configs")


def benchmark_model(model_name, step_counts=[5000, 10000, 20000]):
    print(f"\n=======================================================")
    print(f"  BENCHMARKING: {model_name}")
    print(f"=======================================================")

    results = []

    for n_steps in step_counts:
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_dir = os.path.join(CONFIGS_DIR, f"{model_name}_fast")
            work_dir = os.path.join(tmp_dir, "run")
            shutil.copytree(src_dir, work_dir)

            config_path = os.path.join(work_dir, "input.yaml")
            config = load_config(config_path)

            # Update step count
            config['max_steps'] = n_steps
            config['equilibration'] = int(n_steps * 0.2)
            config['output_interval'] = max(100, int(n_steps * 0.1))
            config['print_energy'] = False  # standard for high-throughput

            start_time = time.perf_counter()
            sim = run_simulation_job(config, input_folder=work_dir)
            elapsed = time.perf_counter() - start_time

            rate = n_steps / elapsed  # MC steps per sec

            if hasattr(sim, 'number'):
                n_particles = sim.number
            elif hasattr(sim, 'number1'):
                n_particles = sim.number1 + sim.number2
            else:
                n_particles = len(getattr(sim, 'positions', []))

            time_per_1m = (1_000_000 / rate) / 60.0  # minutes
            time_per_1b = (1_000_000_000 / rate) / 3600.0  # hours

            results.append({
                'steps': n_steps,
                'particles': n_particles,
                'elapsed_s': elapsed,
                'rate_steps_per_s': rate,
                'min_per_1M_steps': time_per_1m,
                'hours_per_1B_steps': time_per_1b
            })

            print(f"Steps: {n_steps:6d} | Particles: {n_particles:3d} | Time: {elapsed:6.2f} s | Rate: {rate:8.1f} steps/s | 1M steps: {time_per_1m:6.2f} min | 1B steps: {time_per_1b:6.2f} hrs")

    return results


def main():
    print("Starting GCMC Sampling Performance Benchmark...")
    
    models = ['dipole', 'h2o', 'rpm']
    all_results = {}

    for model in models:
        all_results[model] = benchmark_model(model, step_counts=[5000, 10000, 25000])

    print("\n" + "=" * 70)
    print("  SUMMARY: GCMC DATA GENERATION THROUGHPUT & SCALING ANALYSIS")
    print("=" * 70)
    print(f"{'Model':<10} | {'Throughput (steps/s)':<22} | {'Time per 1M steps':<18} | {'Time per 1B steps (Paper)':<25}")
    print("-" * 70)

    for model, res_list in all_results.items():
        avg_rate = np.mean([r['rate_steps_per_s'] for r in res_list])
        time_1m = (1_000_000 / avg_rate) / 60.0
        time_1b = (1_000_000_000 / avg_rate) / 3600.0
        print(f"{model:<10} | {avg_rate:10.1f} steps/s       | {time_1m:6.2f} min          | {time_1b:6.2f} hours")

    print("=" * 70)
    print("\nDataset Projection (2,035 conditions at 1B steps/run):")
    for model, res_list in all_results.items():
        avg_rate = np.mean([r['rate_steps_per_s'] for r in res_list])
        time_1b = (1_000_000_000 / avg_rate) / 3600.0
        total_cpu_hours = time_1b * 2035
        print(f"- {model:<10}: ~{total_cpu_hours:,.0f} CPU hours total ({time_1b:.1f} hrs/condition)")


if __name__ == '__main__':
    main()
