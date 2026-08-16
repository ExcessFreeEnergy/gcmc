'''
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
'''

import os
import sys
import argparse

try:
    from . import read_input
    from . import external_potentials
    from . import potentials
    from . import gcmc_ff
    from . import gcmc_ff_molecule
except ImportError:
    import read_input
    import external_potentials
    import potentials
    import gcmc_ff
    import gcmc_ff_molecule


def run_simulation_job(config, input_folder="."):
    """
    Execute a GCMC simulation given a configuration dictionary and working folder.
    """
    ext_potentials = external_potentials.initialize_external_potentials(config)
    replica_exchange = config.get('replica_exchange', False)
    print_energy = config.get('print_energy', True)
    molecule_flag = config.get('molecule', 'None')

    if not replica_exchange:
        pair_potentials = potentials.initialize_potentials(config)

        if molecule_flag == 'None':
            particle_types = config.get('particle_types', {})
            if len(particle_types) == 1:
                SimulationClass = gcmc_ff.GCMC_FF_SingleType_Simulation
            elif len(particle_types) == 2:
                SimulationClass = gcmc_ff.GCMC_FF_TwoType_Simulation
            else:
                SimulationClass = gcmc_ff.GCMC_FF_MultiType_Simulation
        else:
            if molecule_flag == 'ABC':
                SimulationClass = gcmc_ff_molecule.GCMC_FF_ABC_Simulation
            elif molecule_flag == 'H2O':
                SimulationClass = gcmc_ff_molecule.GCMC_FF_H2O_Simulation
            elif molecule_flag == 'CO2':
                SimulationClass = gcmc_ff_molecule.GCMC_FF_CO2_Simulation
            else:
                raise ValueError(f"Unknown molecule_flag: {molecule_flag}")

        # Instantiate and run
        simulation = SimulationClass(config, pair_potentials, ext_potentials, input_folder)
        if print_energy:
            simulation.run_simulation()
        else:
            simulation.run_simulation_no_energy()
        return simulation
    else:
        try:
            from . import gcmc_re
        except ImportError:
            import gcmc_re
        return gcmc_re.main(config, input_folder)


def cli(argv=None):
    """
    Command-line interface entry point.
    """
    parser = argparse.ArgumentParser(
        description="Run GCMC simulations for short-ranged potentials (v1 engine)."
    )
    parser.add_argument(
        "-in", "--input_folder",
        required=False, type=str, default=".",
        help="The path to folder containing YAML input (input.yaml).",
    )
    args = parser.parse_args(argv)

    input_folder = args.input_folder
    config_path = os.path.join(input_folder, "input.yaml")
    if not os.path.exists(config_path):
        print(f"Error: configuration file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)

    config = read_input.load_config(config_path)
    run_simulation_job(config, input_folder)


if __name__ == "__main__":
    cli()
