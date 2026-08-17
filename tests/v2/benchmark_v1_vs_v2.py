"""
Benchmark comparison: v1 (Python) vs v2 (C++ CPU) vs v2 (CUDA GPU Batched).
"""

import os
import shutil
import tempfile
import time

import gcmc.v1.main as engine_v1
import gcmc.v2 as engine_v2
from gcmc.v1 import load_config

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "v1", "test_configs")


def benchmark_v1(model_name="dipole_fast", steps=10000):
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = os.path.join(CONFIGS_DIR, model_name)
        dest = os.path.join(tmp_dir, "run")
        shutil.copytree(src, dest)
        cfg = load_config(os.path.join(dest, "input.yaml"))
        cfg["max_steps"] = steps
        cfg["equilibration"] = int(steps * 0.2)
        cfg["output_interval"] = int(steps * 0.5)
        cfg["print_energy"] = False

        t0 = time.perf_counter()
        engine_v1.run_simulation_job(cfg, input_folder=dest)
        elapsed = time.perf_counter() - t0
        rate = steps / elapsed
        return rate, elapsed


def benchmark_v2_cpu(model_name="dipole_fast", steps=50000):
    with tempfile.TemporaryDirectory() as tmp_dir:
        src = os.path.join(CONFIGS_DIR, model_name)
        dest = os.path.join(tmp_dir, "run")
        shutil.copytree(src, dest)
        cfg = load_config(os.path.join(dest, "input.yaml"))
        cfg["max_steps"] = steps
        cfg["equilibration"] = int(steps * 0.2)
        cfg["output_interval"] = int(steps * 0.5)
        cfg["print_energy"] = False

        t0 = time.perf_counter()
        engine_v2.run_simulation_job(cfg, input_folder=dest)
        elapsed = time.perf_counter() - t0
        rate = steps / elapsed
        return rate, elapsed


def benchmark_v2_cuda(model_name="dipole_fast", steps=50000, num_boxes=128):
    if not engine_v2.HAS_CUDA:
        return 0.0, 0.0

    src = os.path.join(CONFIGS_DIR, model_name)
    cfg = load_config(os.path.join(src, "input.yaml"))
    batch_configs = [cfg.copy() for _ in range(num_boxes)]

    # Warmup
    engine_v2.run_batch_cuda(batch_configs[:4], num_steps=1000, equilibration_steps=200)

    t0 = time.perf_counter()
    engine_v2.run_batch_cuda(batch_configs, num_steps=steps, equilibration_steps=int(steps * 0.2))
    elapsed = time.perf_counter() - t0
    total_steps = steps * num_boxes
    rate = total_steps / elapsed
    return rate, elapsed


def main():
    print("=" * 80)
    print("      GCMC PERFORMANCE BENCHMARK: v1 vs v2 (CPU & CUDA RTX 4090)")
    print("=" * 80)

    models = ["dipole_fast", "rpm_fast", "h2o_fast"]

    for m in models:
        print(f"\n--- Model: {m} ---")
        r_v1, t_v1 = benchmark_v1(m, steps=10000)
        print(f"  v1 (Python Baseline): {r_v1:10.1f} steps/s  (Time for 10k: {t_v1:.2f}s)")

        r_v2, t_v2 = benchmark_v2_cpu(m, steps=50000)
        speedup_cpu = r_v2 / r_v1
        print(f"  v2 (C++ CPU Core):    {r_v2:10.1f} steps/s  (Speedup vs v1: {speedup_cpu:6.1f}x)")

        if engine_v2.HAS_CUDA:
            num_boxes = 512
            r_cuda, t_cuda = benchmark_v2_cuda(m, steps=50000, num_boxes=num_boxes)
            speedup_cuda = r_cuda / r_v1
            print(f"  v2 (CUDA RTX 4090):   {r_cuda:10.1f} steps/s  ({num_boxes} parallel boxes)")
            print(f"       -> Cumulative Speedup vs single-core v1: {speedup_cuda:6.1f}x")
            time_for_paper_dataset = (2035 * 1_000_000 / r_cuda) / 60.0  # minutes for 2035 runs of 1M steps
            print(f"       -> 2,035 conditions x 1M steps time: {time_for_paper_dataset:.2f} minutes!")


if __name__ == "__main__":
    main()
