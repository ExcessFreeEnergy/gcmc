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

import numpy as np
import gzip

class GCMCMoleculeBaseSimulation:
    def __init__(self, config, input_folder):
        self.initial_config = input_folder + '/' + config.get('init_config', 'initial.xyz') 
        self.logfile = input_folder + '/' + config.get('logfile', 'gcmc.log')
        self.output_xyz = input_folder + '/' + 'output.xyz'
        
        self.global_rc = config['global_rc']
        self.T = config['T']
        self.kB = config.get('kB', 1.0)
        self.beta = 1.0 / (self.kB * self.T)
        self.box_length_x = config['box_length_x']
        self.box_length_y = config['box_length_y']
        self.box_length_z = config['box_length_z']
        self.box_length = np.array([self.box_length_x, self.box_length_y, self.box_length_z])
        self.volume = self.box_length_x * self.box_length_y * self.box_length_z

        self.max_steps = config['max_steps']
        self.equilibration_steps = config.get('equilibration', 1000)
        self.output_interval = config['output_interval']
        self.output_steps = set(range(0, self.max_steps + 1, self.output_interval))

        
        self.weights = config.get('weights', {'insert': 1.0, 'delete': 1.0, 'displace': 0.0, 'rotate': 0.0})
        total_weight = sum(self.weights.values())
        self.insert_prob = self.weights['insert'] / total_weight
        self.delete_prob = self.weights['delete'] / total_weight
        self.displace_prob = self.weights['displace'] / total_weight
        try:
            self.rotate_prob = self.weights['rotate'] / total_weight
        except KeyError:
            self.rotate_prob = 0.0
        
        self.maxdispl = config.get('maxdispl', 3.0)
        self.maxang = config.get('maxang', np.pi)
        self.maxcos = config.get('maxcos', 1.0)

    def minimum_image(self, pos):
        return pos - self.box_length * np.round(pos / self.box_length)

    def wrap_pbc(self, pos):
        return pos - self.box_length * np.floor(pos / self.box_length)

    def unwrap_pbc(self, pos):
        delta = pos - pos[0, :]
        delta = np.where(delta > self.box_length * 0.5, delta - self.box_length, delta)
        delta = np.where(delta < -self.box_length * 0.5, delta + self.box_length, delta)
        return pos[0, :] + delta

    def calculate_same_energy(self, positions1, positions2, potential):
        """
        Compute same-type pairwise interactions between positions1 and positions2,
        """
        deltas = positions1[:, np.newaxis, :] - positions2[np.newaxis, :, :]
        deltas = self.minimum_image(deltas)
        distances = np.linalg.norm(deltas, axis=2)
        np.fill_diagonal(distances, np.inf)
        within_rc = distances < self.global_rc
        energies = np.zeros_like(distances)
        energies[within_rc] = potential.calculate(distances[within_rc])
        return np.sum(np.triu(energies, k=1))
        
    
    def calculate_cross_energy(self, positions1, positions2, potential):
        """
        Compute cross-type pairwise interactions between positions1 and positions2,
        excluding interactions where particles are from the same molecule (i.e., same index).
        """
        
        n = positions1.shape[0]
        deltas = positions1[:, np.newaxis, :] - positions2[np.newaxis, :, :]
        deltas = self.minimum_image(deltas)
        distances = np.linalg.norm(deltas, axis=2)
        # Exclude same-index interactions
        mask = ~np.eye(n, dtype=bool)
        rij = distances[mask]
        return np.sum(potential.calculate(rij))
    
    def calculate_unrestricted_energy(self, positions1, positions2, potential):
        """
        Compute cross-type pairwise interactions between positions1 and positions2,
        assuming no intra-molecular exclusion is needed (i.e., from different atoms or molecules).
        """
        deltas = positions1[:, np.newaxis, :] - positions2[np.newaxis, :, :]
        deltas = self.minimum_image(deltas)
        distances = np.linalg.norm(deltas, axis=2)
        return np.sum(potential.calculate(distances))
    
    def calculate_local_pw_energy(self, ref_pos, other_positions, potential):
        deltas = other_positions - ref_pos
        deltas = self.minimum_image(deltas)
        distances = np.linalg.norm(deltas, axis=1)
        within_rc = distances < self.global_rc
        return np.sum(potential.calculate(distances[within_rc]))

    def write_log_header(self, header):
        with open(self.logfile, 'w') as f:
            f.write(header + "\n")

    def write_xyz_header(self):
        with gzip.open(self.output_xyz + '.gz', 'wt') as f:
            pass

    def run_simulation(self):
        """
        Run the GCMC simulation for the configured number of steps.
        """
        self.write_log_header()
        self.write_xyz_header()
        
        # Equilibration phase
        for step in range(self.equilibration_steps):
            self.gcmc_step()
            if step in self.output_steps:
                self.log(step)
            
        # Production phase
        for step in range(self.equilibration_steps, self.max_steps):
            self.gcmc_step()
            if step in self.output_steps:
                self.log(step)
                self.write_xyz(step)
                
        final_step = self.max_steps
        self.log(final_step)
        self.write_xyz(self.max_steps)

    def run_simulation_no_energy(self):
        """
        Run the GCMC simulation for the configured number of steps.
        """
        self.write_log_header()
        self.write_xyz_header()
        
        # Equilibration phase
        for step in range(self.equilibration_steps):
            self.gcmc_step()
            if step in self.output_steps:
                self.log_no_energy(step)
            
        # Production phase
        for step in range(self.equilibration_steps, self.max_steps):
            self.gcmc_step()
            if step in self.output_steps:
                self.log_no_energy(step)
                self.write_xyz(step)
                
        final_step = self.max_steps
        self.log(final_step)
        self.write_xyz(self.max_steps)
