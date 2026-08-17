"""High-performance GCMC simulation engine (v2 - C++/CUDA)."""

from .bindings import (
    HAS_CUDA,
    GCMCSimulationV2,
    run_batch_cuda,
)


def run_simulation_job(config, input_folder="."):
    """
    Run GCMC simulation using the high-performance v2 engine.
    """
    sim = GCMCSimulationV2(config, input_folder=input_folder)
    print_energy = config.get("print_energy", True)
    if print_energy:
        sim.run_simulation()
    else:
        sim.run_simulation_no_energy()
    return sim


__all__ = [
    "GCMCSimulationV2",
    "run_simulation_job",
    "run_batch_cuda",
    "HAS_CUDA",
]
