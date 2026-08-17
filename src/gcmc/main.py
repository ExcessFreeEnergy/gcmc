"""
Main entry point for GCMC simulations with multi-engine support.
Default engine is v2 (C++/CUDA accelerated).
"""

import argparse
import os
import sys

import yaml


def cli(argv=None):
    parser = argparse.ArgumentParser(description="Run GCMC simulations for short-ranged Gaussian truncated potentials.")
    parser.add_argument(
        "-in",
        "--input_folder",
        required=False,
        type=str,
        default=".",
        help="Path to folder containing YAML input (input.yaml).",
    )
    parser.add_argument(
        "--engine",
        required=False,
        type=str,
        choices=["v2", "v1"],
        default="v2",
        help="Simulation engine: 'v2' (high-performance C++/CUDA, default) or 'v1' (Python baseline).",
    )
    parser.add_argument(
        "--enable-long-range",
        action="store_true",
        default=False,
        help="Enable full long-range Ewald electrostatics (v2 engine only). Default is short-range (SR).",
    )
    parser.add_argument(
        "--ewald-alpha",
        type=float,
        default=None,
        help="Ewald screening parameter alpha (default: 0.35 A^-1).",
    )
    parser.add_argument(
        "--ewald-kmax",
        type=int,
        default=None,
        help="Max reciprocal k-vector index kmax (default: 4).",
    )
    args = parser.parse_args(argv)

    input_folder = args.input_folder
    config_path = os.path.join(input_folder, "input.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.enable_long_range:
        config["enable_long_range"] = True
        config["electrostatics_mode"] = "long_range"
    if args.ewald_alpha is not None:
        config["ewald_alpha"] = args.ewald_alpha
    if args.ewald_kmax is not None:
        config["ewald_kmax"] = args.ewald_kmax

    if args.engine == "v2":
        try:
            from gcmc.v2 import run_simulation_job

            print(f"[gcmc] Running simulation with high-performance engine (v2) in '{input_folder}'...")
            return run_simulation_job(config, input_folder)
        except Exception as e:
            print(
                f"[gcmc] Warning: v2 engine encountered an issue ({e}), falling back to v1 engine...", file=sys.stderr
            )
            from gcmc.v1 import run_simulation_job

            return run_simulation_job(config, input_folder)
    else:
        from gcmc.v1 import run_simulation_job

        print(f"[gcmc] Running simulation with baseline reference engine (v1) in '{input_folder}'...")
        return run_simulation_job(config, input_folder)


if __name__ == "__main__":
    cli()
