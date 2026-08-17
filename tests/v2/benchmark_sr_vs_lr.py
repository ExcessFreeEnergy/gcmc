"""
Empirical benchmark comparison: Short-Range (SR) vs Long-Range Ewald (LR) across CPU and CUDA GPU.
"""

import copy
import os
import shutil
import tempfile
import time

import gcmc.v2 as engine_v2
from gcmc.v1 import load_config

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "v1", "test_configs")


def benchmark_sr_vs_lr(model_name="dipole_fast", steps=20000, num_boxes=64):
    print("=" * 80)
    print(f"  BENCHMARK: Short-Range (SR) vs Long-Range Ewald (LR) [{model_name}]")
    print("=" * 80)

    src = os.path.join(CONFIGS_DIR, model_name)
    base_cfg = load_config(os.path.join(src, "input.yaml"))

    # 1. CPU Short-Range
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = os.path.join(tmp_dir, "sr_cpu")
        shutil.copytree(src, dest)
        cfg_sr = load_config(os.path.join(dest, "input.yaml"))
        cfg_sr["max_steps"] = steps
        cfg_sr["equilibration"] = int(steps * 0.2)
        cfg_sr["output_interval"] = int(steps * 0.5)
        cfg_sr["print_energy"] = False
        cfg_sr["electrostatics_mode"] = "short_range"

        t0 = time.perf_counter()
        engine_v2.run_simulation_job(cfg_sr, input_folder=dest)
        t_sr_cpu = time.perf_counter() - t0
        rate_sr_cpu = steps / t_sr_cpu
        print(f"[v2 CPU SR]  Throughput: {rate_sr_cpu:12.1f} steps/s ({t_sr_cpu:.3f} s)")

    # 2. CPU Long-Range Ewald
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest = os.path.join(tmp_dir, "lr_cpu")
        shutil.copytree(src, dest)
        cfg_lr = load_config(os.path.join(dest, "input.yaml"))
        cfg_lr["max_steps"] = steps
        cfg_lr["equilibration"] = int(steps * 0.2)
        cfg_lr["output_interval"] = int(steps * 0.5)
        cfg_lr["print_energy"] = False
        cfg_lr["electrostatics_mode"] = "long_range"
        cfg_lr["ewald_alpha"] = 0.35
        cfg_lr["ewald_kmax"] = 4

        t0 = time.perf_counter()
        engine_v2.run_simulation_job(cfg_lr, input_folder=dest)
        t_lr_cpu = time.perf_counter() - t0
        rate_lr_cpu = steps / t_lr_cpu
        print(f"[v2 CPU LR]  Throughput: {rate_lr_cpu:12.1f} steps/s ({t_lr_cpu:.3f} s)")

    # 3. GPU CUDA Batched SR
    if engine_v2.HAS_CUDA:
        configs_sr = [copy.deepcopy(base_cfg) for _ in range(num_boxes)]
        for c in configs_sr:
            c["electrostatics_mode"] = "short_range"
        total_steps = steps * num_boxes
        t0 = time.perf_counter()
        engine_v2.run_batch_cuda(configs_sr, num_steps=steps, equilibration_steps=int(steps * 0.2))
        t_sr_gpu = time.perf_counter() - t0
        rate_sr_gpu = total_steps / t_sr_gpu
        print(f"[v2 GPU SR]  Throughput: {rate_sr_gpu:12.1f} steps/s ({t_sr_gpu:.3f} s across {num_boxes} boxes)")

        # 4. GPU CUDA Batched LR
        configs_lr = [copy.deepcopy(base_cfg) for _ in range(num_boxes)]
        for c in configs_lr:
            c["electrostatics_mode"] = "long_range"
            c["ewald_alpha"] = 0.35
            c["ewald_kmax"] = 4
        t0 = time.perf_counter()
        engine_v2.run_batch_cuda(configs_lr, num_steps=steps, equilibration_steps=int(steps * 0.2))
        t_lr_gpu = time.perf_counter() - t0
        rate_lr_gpu = total_steps / t_lr_gpu
        print(f"[v2 GPU LR]  Throughput: {rate_lr_gpu:12.1f} steps/s ({t_lr_gpu:.3f} s across {num_boxes} boxes)")
        print(f"-> Speedup (GPU LR vs CPU LR): {rate_lr_gpu / rate_lr_cpu:.1f}x")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    benchmark_sr_vs_lr("dipole_fast", steps=20000, num_boxes=64)
    benchmark_sr_vs_lr("rpm_fast", steps=20000, num_boxes=64)
