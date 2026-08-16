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

try:
    from .molecule_base import GCMCMoleculeBaseSimulation
    from . import tools as tls
except ImportError:
    from molecule_base import GCMCMoleculeBaseSimulation
    import tools as tls
import numpy as np
from collections import Counter
import gzip


class GCMC_FF_ABC_Simulation(GCMCMoleculeBaseSimulation):
    def __init__(self, config, potentials, external_potentials, input_folder):
        super().__init__(config, input_folder)

        # Pair potentials
        self.potential_A_B = potentials['A_B']
        self.potential_A_C = potentials['A_C']
        self.potential_B_C = potentials['B_C']
        self.potential_A_A = potentials['A_A']
        self.potential_B_B = potentials['B_B']
        self.potential_C_C = potentials['C_C']

        # External potentials
        self.external_potential_A = external_potentials['A']
        self.external_potential_B = external_potentials['B']
        self.external_potential_C = external_potentials['C']

        # Chemical potential
        self.mu = config['particle_types']['ABC']['mu'] * self.kB * self.T

        # Initial structure
        self.positions, self.types = self.load_xyz(self.initial_config)
        self.number = len(self.positions)        

        # Parameters
        self.bond_length = config.get('bond_length', 0.5)
                
    def load_xyz(self, filename):
        """
        Load particle positions from an XYZ file.
        """
        try:
            data = np.genfromtxt(filename, skip_header=2, dtype='str')
            molecules = data[:, 1:].astype(float)
            positions = molecules.reshape(-1, 3, 3)
            types = ['ABC'] * len(positions)
           
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with an empty configuration.")
            types = ['ABC']
            positions = tls.ABC_origin
        return positions, types
    
    def get_individual_positions(self, positions):
        """
        Get the positions of individual atoms from the positions of molecules.
        """
        A_positions = positions[:, 0, :]
        B_positions = positions[:, 1, :]
        C_positions = positions[:, 2, :]
        return A_positions, B_positions, C_positions


    def total_energy(self):
        A_positions, B_positions, C_positions = self.get_individual_positions(self.positions)
        E_pairwise = 0.0

        # FUNCTIONS ALWAYS EVALUATE TO ZERO COMMENTED OUT FOR EFFICIENCY
        if self.number > 0:
            E_pairwise += self.calculate_same_energy(A_positions, A_positions, self.potential_A_A)
            E_pairwise += self.calculate_same_energy(B_positions, B_positions, self.potential_B_B)
            E_pairwise += self.calculate_same_energy(C_positions, C_positions, self.potential_C_C)
            #E_pairwise += self.calculate_cross_energy(A_positions, B_positions, self.potential_A_B)
            #E_pairwise += self.calculate_cross_energy(A_positions, C_positions, self.potential_A_C)
            E_pairwise += self.calculate_cross_energy(B_positions, C_positions, self.potential_B_C)

        E_ext = (
            np.sum(self.external_potential_A.calculate_multiple(A_positions)) +
            np.sum(self.external_potential_B.calculate_multiple(B_positions)) +
            np.sum(self.external_potential_C.calculate_multiple(C_positions))
        )

        return E_pairwise + E_ext
    
    
    def local_energy(self, pos, positions):
        """
        Calculate the local energy of a particle with respect to all other particles.
        Add external potential energy for the given particle.
        """
        A_pos, B_pos, C_pos = pos
        A_positions, B_positions, C_positions = self.get_individual_positions(positions)
        E = 0.0

        # FUNCTIONS ALWAYS EVALUATE TO ZERO COMMENTED OUT FOR EFFICIENCY
        if self.number > 0:
            E += self.calculate_local_pw_energy(A_pos, A_positions, self.potential_A_A)
            #E += self.calculate_local_pw_energy(A_pos, B_positions, self.potential_A_B)
            #E += self.calculate_local_pw_energy(A_pos, C_positions, self.potential_A_C)
            #E += self.calculate_local_pw_energy(B_pos, A_positions, self.potential_A_B)
            E += self.calculate_local_pw_energy(B_pos, B_positions, self.potential_B_B)
            E += self.calculate_local_pw_energy(B_pos, C_positions, self.potential_B_C)
            #E += calculate_energy(C_pos, A_positions, self.potential_A_C)
            E += self.calculate_local_pw_energy(C_pos, B_positions, self.potential_B_C)
            E += self.calculate_local_pw_energy(C_pos, C_positions, self.potential_C_C)
            
        E += self.external_potential_A(A_pos)
        E += self.external_potential_B(B_pos)
        E += self.external_potential_C(C_pos)

        return E
    
    def gcmc_step(self):
        """
        Perform a GCMC step, attempting either insertion, deletion, or displacement.
        """
        rand_num = np.random.rand()
        if rand_num < self.insert_prob:
            self.insert_particle()
        elif rand_num < self.insert_prob + self.delete_prob:
            self.delete_particle()
        elif rand_num < self.insert_prob + self.delete_prob + self.displace_prob:
            self.displace_particle()
        else:
            self.rotate_particle()

    def generate_new_molecule(self):
        """
        Generate a new linear molecule.
        """
        rotated_molecule = tls.generate_random_linear_triatomic(self.bond_length) + np.random.uniform(0, self.box_length, 3)
        return self.wrap_pbc(rotated_molecule)

    def insert_particle(self):
        """
        Attemp to insert a particle
        """
        new_pos = self.generate_new_molecule()
        delta_E = self.local_energy(new_pos, self.positions)
        
        prob = np.exp(-self.beta * (delta_E - self.mu)) * self.volume / (self.number+1)
        if np.random.rand() < prob:
            self.positions = np.concatenate((self.positions, [new_pos]), axis=0)
            
            self.number = self.number + 1
        
    def delete_particle(self):
        """
        Attemp to delete a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            del_pos = self.positions[idx]
            remain_positions = np.delete(self.positions, idx, axis=0)
            delta_E = -self.local_energy(del_pos, remain_positions)
        
            log_prob = -self.beta * (delta_E + self.mu) + np.log(self.number) - np.log(self.volume)
            if log_prob < np.log(np.finfo(float).max):
                prob = np.exp(log_prob)
            else:
                prob = 0

            if np.random.rand() < prob:
                self.positions = remain_positions
                self.number = self.number - 1           
        
    def rotate_particle(self):
        """
        Attemp to rotate a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            old_pos = self.positions[idx]
            unwrapped_pos = self.unwrap_pbc(old_pos)
            rotated_molecule = tls.RotMove_shift_linear(unwrapped_pos)
            new_pos = self.wrap_pbc(rotated_molecule)
            remain_positions = np.delete(self.positions, idx, axis=0)
            old_energy = self.local_energy(old_pos, remain_positions)
            new_energy = self.local_energy(new_pos, remain_positions)
            
            delta_E = new_energy - old_energy

            # Apply the log-sum-exp trick for numerical stability
            log_prob_accept = -self.beta * delta_E

            if log_prob_accept > 0:
                self.positions[idx] = new_pos   # if log_prob_accept is positive, accept the move
            else:
                prob_accept = np.exp(log_prob_accept)
                if np.random.rand() < prob_accept:
                    self.positions[idx] = new_pos

    def displace_particle(self):
        """
        Attemp to displace a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            old_pos = self.positions[idx]
            unwrapped_pos = self.unwrap_pbc(old_pos)
            new_pos = unwrapped_pos + np.random.uniform(-self.maxdispl, self.maxdispl, 3)
            new_pos = self.wrap_pbc(new_pos)
            remain_positions = np.delete(self.positions, idx, axis=0)
            old_energy = self.local_energy(old_pos, remain_positions)
            new_energy = self.local_energy(new_pos, remain_positions)
            
            delta_E = new_energy - old_energy

            # Apply the log-sum-exp trick for numerical stability
            log_prob_accept = -self.beta * delta_E

            if log_prob_accept > 0:
                self.positions[idx] = new_pos   # if log_prob_accept is positive, accept the move
            else:
                prob_accept = np.exp(log_prob_accept)
                if np.random.rand() < prob_accept:
                    self.positions[idx] = new_pos

                
    def write_xyz(self, step):
        """
        Write the positions to an Extended XYZ file format for a given step.
        """
        
        with gzip.open(self.output_xyz + '.gz', 'at') as f:
            cell = f"{self.box_length_x} 0.0 0.0 0.0 {self.box_length_y} 0.0 0.0 0.0 {self.box_length_z}"
            f.write(f"{self.number*3}\n")
            f.write(f"Step {step} Lattice=\"{cell}\" Properties=species:S:1:pos:R:3\n")
            for i, pos in enumerate(self.positions):
                f.write(f"A {pos[0,0]:.6f} {pos[0,1]:.6f} {pos[0,2]:.6f}\n")
                f.write(f"B {pos[1,0]:.6f} {pos[1,1]:.6f} {pos[1,2]:.6f}\n")
                f.write(f"C {pos[2,0]:.6f} {pos[2,1]:.6f} {pos[2,2]:.6f}\n")

    def log(self, step):
        """
        Log the step, total number of particles, and the number of particles of each available type.
        """
        total_energy = self.total_energy()
        with open(self.logfile, 'a') as f:
            f.write(f"{step} {self.number} {total_energy}\n")

    def log_no_energy(self, step):
        """
        Log the step, total number of particles, and the number of particles of each available type.
        """
        with open(self.logfile, 'a') as f:
            f.write(f"{step} {self.number} \n")

    def write_log_header(self):
        """
        Write the header to the log file.
        """
        with open(self.logfile, 'w') as f:
            f.write("Step Total_number Energy\n")


class GCMC_FF_H2O_Simulation(GCMCMoleculeBaseSimulation):
    def __init__(self, config, potentials, external_potentials, input_folder):
        super().__init__(config, input_folder)

        # Pair potentials
        self.potential_H_H = potentials['H_H']
        self.potential_O_O = potentials['O_O']
        self.potential_H_O = potentials['H_O']

        # External potentials
        self.external_potential_H = external_potentials['H']
        self.external_potential_O = external_potentials['O']   

        # Chemical potential
        self.mu = config['particle_types']['H2O']['mu'] * self.kB * self.T

        # Initial structure
        self.positions, self.types = self.load_xyz(self.initial_config)
        self.number = len(self.positions)        
        
    def load_xyz(self, filename):
        """
        Load particle positions from an XYZ file.
        """
        try:
            data = np.genfromtxt(filename, skip_header=2, dtype='str')
            molecules = data[:, 1:].astype(float)
            positions = molecules.reshape(-1, 3, 3)
            types = ['H2O'] * len(positions)
           
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with an empty configuration.")
            types = ['H2O']
            positions = tls.SPCE_origin
        return positions, types
    
    def get_individual_positions(self, positions):
        """
        Get the positions of individual atoms from the positions of molecules.
        """
        O_positions = positions[:, 0, :]
        H1_positions = positions[:, 1, :]
        H2_positions = positions[:, 2, :]
        return O_positions, H1_positions, H2_positions
        
    def total_energy(self):
        O_positions, H1_positions, H2_positions = self.get_individual_positions(self.positions)
        E_pairwise = 0.0
        
        if self.number > 0:
            E_pairwise += self.calculate_same_energy(O_positions, O_positions, self.potential_O_O)
            E_pairwise += self.calculate_same_energy(H1_positions, H1_positions, self.potential_H_H)
            E_pairwise += self.calculate_same_energy(H2_positions, H2_positions, self.potential_H_H)
            E_pairwise += self.calculate_cross_energy(H1_positions, H2_positions, self.potential_H_H)
            E_pairwise += self.calculate_cross_energy(H1_positions, O_positions, self.potential_H_O)
            E_pairwise += self.calculate_cross_energy(H2_positions, O_positions, self.potential_H_O)    
            
        E_ext = (
            np.sum(self.external_potential_H.calculate_multiple(H1_positions)) +
            np.sum(self.external_potential_H.calculate_multiple(H2_positions)) +
            np.sum(self.external_potential_O.calculate_multiple(O_positions))
        )

        return E_pairwise + E_ext

    def local_energy(self, pos, positions):
        """
        Calculate the local energy of a particle with respect to all other particles.
        Add external potential energy for the given particle.
        """
        O_pos, H1_pos, H2_pos = pos
        O_positions, H1_positions, H2_positions = self.get_individual_positions(positions)
        H_positions = np.vstack([H1_positions, H2_positions])
        E = 0.0
        
        if self.number > 0:
            E += self.calculate_local_pw_energy(O_pos, O_positions, self.potential_O_O)
            E += self.calculate_local_pw_energy(O_pos, H_positions, self.potential_H_O)
            E += self.calculate_local_pw_energy(H1_pos, O_positions, self.potential_H_O)
            E += self.calculate_local_pw_energy(H1_pos, H_positions, self.potential_H_H)
            E += self.calculate_local_pw_energy(H2_pos, O_positions, self.potential_H_O)
            E += self.calculate_local_pw_energy(H2_pos, H_positions, self.potential_H_H)
            
        E += self.external_potential_H(H1_pos)
        E += self.external_potential_H(H2_pos)
        E += self.external_potential_O(O_pos)
            
        return E
    
    def gcmc_step(self):
        """
        Perform a GCMC step, attempting either insertion, deletion, or displacement.
        """
        rand_num = np.random.rand()
        if rand_num < self.insert_prob:
            self.insert_particle()
        elif rand_num < self.insert_prob + self.delete_prob:
            self.delete_particle()
        elif rand_num < self.insert_prob + self.delete_prob + self.displace_prob:
            self.displace_particle()
        else:
            self.rotate_particle()
        
            
    def generate_new_molecule(self):
        """
        Generate a new water molecule.
        """
        new_molecule = tls.SPCE_origin + np.random.uniform(0, self.box_length, 3)
        rotated_molecule = tls.RotMove_init(new_molecule)
        return self.wrap_pbc(rotated_molecule)

    def insert_particle(self):
        """
        Attemp to insert a particle
        """
        new_pos = self.generate_new_molecule()
        delta_E = self.local_energy(new_pos, self.positions)
        
        prob = np.exp(-self.beta * (delta_E - self.mu)) * self.volume / (self.number+1)
        if np.random.rand() < prob:
            self.positions = np.concatenate((self.positions, [new_pos]), axis=0)
            
            self.number = self.number + 1
        
    def delete_particle(self):
        """
        Attemp to delete a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            del_pos = self.positions[idx]
            remain_positions = np.delete(self.positions, idx, axis=0)
            delta_E = -self.local_energy(del_pos, remain_positions)
        
            log_prob = -self.beta * (delta_E + self.mu) + np.log(self.number) - np.log(self.volume)
            if log_prob < np.log(np.finfo(float).max):
                prob = np.exp(log_prob)
            else:
                prob = 0

            if np.random.rand() < prob:
                self.positions = remain_positions
                self.number = self.number - 1           
        
    def rotate_particle(self):
        """
        Attemp to rotate a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            old_pos = self.positions[idx]
            unwrapped_pos = self.unwrap_pbc(old_pos)
            rotated_molecule = tls.RotMove_shift_non_linear(unwrapped_pos)
            new_pos = self.wrap_pbc(rotated_molecule)
            remain_positions = np.delete(self.positions, idx, axis=0)
            old_energy = self.local_energy(old_pos, remain_positions)
            new_energy = self.local_energy(new_pos, remain_positions)
            
            delta_E = new_energy - old_energy

            # Apply the log-sum-exp trick for numerical stability
            log_prob_accept = -self.beta * delta_E

            if log_prob_accept > 0:
                self.positions[idx] = new_pos   # if log_prob_accept is positive, accept the move
            else:
                prob_accept = np.exp(log_prob_accept)
                if np.random.rand() < prob_accept:
                    self.positions[idx] = new_pos

    def displace_particle(self):
        """
        Attemp to displace a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            old_pos = self.positions[idx]
            unwrapped_pos = self.unwrap_pbc(old_pos)
            new_pos = unwrapped_pos + np.random.uniform(-self.maxdispl, self.maxdispl, 3)
            new_pos = self.wrap_pbc(new_pos)
            remain_positions = np.delete(self.positions, idx, axis=0)
            old_energy = self.local_energy(old_pos, remain_positions)
            new_energy = self.local_energy(new_pos, remain_positions)
            
            delta_E = new_energy - old_energy

            # Apply the log-sum-exp trick for numerical stability
            log_prob_accept = -self.beta * delta_E

            if log_prob_accept > 0:
                self.positions[idx] = new_pos   # if log_prob_accept is positive, accept the move
            else:
                prob_accept = np.exp(log_prob_accept)
                if np.random.rand() < prob_accept:
                    self.positions[idx] = new_pos


    def displace_and_rotate_particle(self):
        """
        Attemp to displace a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            old_pos = self.positions[idx]
            unwrapped_pos = self.unwrap_pbc(old_pos)
            rotated_molecule = tls.RotMove_shift_non_linear(unwrapped_pos)
            new_pos = rotated_molecule + np.random.uniform(-self.maxdispl, self.maxdispl, 3)
            new_pos = self.wrap_pbc(new_pos)
            remain_positions = np.delete(self.positions, idx, axis=0)
            old_energy = self.local_energy(old_pos, remain_positions)
            new_energy = self.local_energy(new_pos, remain_positions)
            
            delta_E = new_energy - old_energy

            # Apply the log-sum-exp trick for numerical stability
            log_prob_accept = -self.beta * delta_E

            if log_prob_accept > 0:
                self.positions[idx] = new_pos   # if log_prob_accept is positive, accept the move
            else:
                prob_accept = np.exp(log_prob_accept)
                if np.random.rand() < prob_accept:
                    self.positions[idx] = new_pos
                
    def write_xyz(self, step):
        """
        Write the positions to an Extended XYZ file format for a given step.
        """
        
        with gzip.open(self.output_xyz + '.gz', 'at') as f:
            cell = f"{self.box_length_x} 0.0 0.0 0.0 {self.box_length_y} 0.0 0.0 0.0 {self.box_length_z}"
            f.write(f"{self.number*3}\n")
            f.write(f"Step {step} Lattice=\"{cell}\" Properties=species:S:1:pos:R:3\n")
            for i, pos in enumerate(self.positions):
                f.write(f"O {pos[0,0]:.6f} {pos[0,1]:.6f} {pos[0,2]:.6f}\n")
                f.write(f"H1 {pos[1,0]:.6f} {pos[1,1]:.6f} {pos[1,2]:.6f}\n")
                f.write(f"H2 {pos[2,0]:.6f} {pos[2,1]:.6f} {pos[2,2]:.6f}\n")

    def log(self, step):
        """
        Log the step, total number of particles, and the number of particles of each available type.
        """
        total_energy = self.total_energy()
        with open(self.logfile, 'a') as f:
            f.write(f"{step} {self.number} {total_energy}\n")


    def log_no_energy(self, step):
        """
        Log the step, total number of particles, and the number of particles of each available type.
        """
        with open(self.logfile, 'a') as f:
            f.write(f"{step} {self.number} \n")

    def write_log_header(self):
        """
        Write the header to the log file.
        """
        with open(self.logfile, 'w') as f:
            f.write("Step Total_number Energy\n")
        
class GCMC_FF_CO2_Simulation(GCMCMoleculeBaseSimulation):
    def __init__(self, config, potentials, external_potentials, input_folder):
        super().__init__(config, input_folder)
        
        # Pair potentials
        self.potential_O_O = potentials['O_O']
        self.potential_C_C = potentials['C_C']
        self.potential_C_O = potentials['C_O']
        
        # External potentials
        self.external_potential_C = external_potentials['C']
        self.external_potential_O = external_potentials['O']      
        
        # Chemical potential
        self.mu = config['particle_types']['CO2']['mu'] * self.kB * self.T

        
        # Initial structure
        self.positions, self.types = self.load_xyz(self.initial_config) 
        self.number = len(self.positions)
        
        self.bond_length = config.get('bond_length', 1.16)     
        
    def load_xyz(self, filename):
        """
        Load particle positions from an XYZ file.
        """
        try:
            data = np.genfromtxt(filename, skip_header=2, dtype='str')
            molecules = data[:, 1:].astype(float)
            positions = molecules.reshape(-1, 3, 3)
            types = ['CO2'] * len(positions)
           
        except FileNotFoundError:
            print(f"File {filename} not found. Starting with an empty configuration.")
            types = ['CO2']
            positions = tls.CO2_origin
        return positions, types
    
    def get_individual_positions(self, positions):
        """
        Get the positions of individual atoms from the positions of molecules.
        """
        C_positions = positions[:, 0, :]
        O1_positions = positions[:, 1, :]
        O2_positions = positions[:, 2, :]
        return C_positions, O1_positions, O2_positions

    def total_energy(self):
        C_positions, O1_positions, O2_positions = self.get_individual_positions(self.positions)
        E_pairwise = 0.0
            
        if self.number > 0:
            E_pairwise += self.calculate_same_energy(C_positions, C_positions, self.potential_C_C)
            E_pairwise += self.calculate_same_energy(O1_positions, O1_positions, self.potential_O_O)
            E_pairwise += self.calculate_same_energy(O2_positions, O2_positions, self.potential_O_O)
            E_pairwise += self.calculate_cross_energy(O1_positions, O2_positions, self.potential_O_O)
            E_pairwise += self.calculate_cross_energy(C_positions, O1_positions, self.potential_C_O)
            E_pairwise += self.calculate_cross_energy(C_positions, O2_positions, self.potential_C_O)

        E_ext = (
            np.sum(self.external_potential_C.calculate_multiple(C_positions)) +
            np.sum(self.external_potential_O.calculate_multiple(O1_positions)) +
            np.sum(self.external_potential_O.calculate_multiple(O2_positions))
        )
        return E_pairwise + E_ext

    
    def local_energy(self, pos, positions):
        """
        Calculate the local energy of a particle with respect to all other particles.
        Add external potential energy for the given particle.
        """
        C_pos, O1_pos, O2_pos = pos
        C_positions, O1_positions, O2_positions = self.get_individual_positions(positions)
        O_positions = np.vstack([O1_positions, O2_positions])
        
        E = 0.0 
        
        if self.number > 0:
            E += self.calculate_local_pw_energy(C_pos, C_positions, self.potential_C_C)
            E += self.calculate_local_pw_energy(C_pos, O_positions, self.potential_C_O)
            E += self.calculate_local_pw_energy(O1_pos, C_positions, self.potential_C_O)
            E += self.calculate_local_pw_energy(O1_pos, O_positions, self.potential_O_O)
            E += self.calculate_local_pw_energy(O2_pos, C_positions, self.potential_C_O)
            E += self.calculate_local_pw_energy(O2_pos, O_positions, self.potential_O_O)
        
        E += self.external_potential_O(O1_pos)
        E += self.external_potential_O(O2_pos)
        E += self.external_potential_C(C_pos)
            
        return E
    
    
    def gcmc_step(self):
        """
        Perform a GCMC step, attempting either insertion, deletion, or displacement.
        """
        rand_num = np.random.rand()
        if rand_num < self.insert_prob:
            self.insert_particle()
        elif rand_num < self.insert_prob + self.delete_prob:
            self.delete_particle()
        elif rand_num < self.insert_prob + self.delete_prob + self.displace_prob:
            self.displace_particle()
        else:
            self.rotate_particle()
        
            
    def generate_new_molecule(self):
        """
        Generate a new water molecule.
        """
        rotated_molecule = tls.generate_random_linear_triatomic(self.bond_length) + np.random.uniform(0, self.box_length, 3)
        return self.wrap_pbc(rotated_molecule)

    def insert_particle(self):
        """
        Attemp to insert a particle
        """
        new_pos = self.generate_new_molecule()
        delta_E = self.local_energy(new_pos, self.positions)
        
        prob = np.exp(-self.beta * (delta_E - self.mu)) * self.volume / (self.number+1)
        if np.random.rand() < prob:
            self.positions = np.concatenate((self.positions, [new_pos]), axis=0)
            
            self.number = self.number + 1
        
    def delete_particle(self):
        """
        Attemp to delete a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            del_pos = self.positions[idx]
            remain_positions = np.delete(self.positions, idx, axis=0)
            delta_E = -self.local_energy(del_pos, remain_positions)
        
            log_prob = -self.beta * (delta_E + self.mu) + np.log(self.number) - np.log(self.volume)
            if log_prob < np.log(np.finfo(float).max):
                prob = np.exp(log_prob)
            else:
                prob = 0

            if np.random.rand() < prob:
                self.positions = remain_positions
                self.number = self.number - 1           
        
    def rotate_particle(self):
        """
        Attemp to rotate a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            old_pos = self.positions[idx]
            unwrapped_pos = self.unwrap_pbc(old_pos)
            rotated_molecule = tls.RotMove_shift_linear(unwrapped_pos)
            new_pos = self.wrap_pbc(rotated_molecule)
            remain_positions = np.delete(self.positions, idx, axis=0)
            old_energy = self.local_energy(old_pos, remain_positions)
            new_energy = self.local_energy(new_pos, remain_positions)
            
            delta_E = new_energy - old_energy

            # Apply the log-sum-exp trick for numerical stability
            log_prob_accept = -self.beta * delta_E

            if log_prob_accept > 0:
                self.positions[idx] = new_pos   # if log_prob_accept is positive, accept the move
            else:
                prob_accept = np.exp(log_prob_accept)
                if np.random.rand() < prob_accept:
                    self.positions[idx] = new_pos

    def displace_particle(self):
        """
        Attemp to displace a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            old_pos = self.positions[idx]
            unwrapped_pos = self.unwrap_pbc(old_pos)
            new_pos = unwrapped_pos + np.random.uniform(-self.maxdispl, self.maxdispl, 3)
            new_pos = self.wrap_pbc(new_pos)
            remain_positions = np.delete(self.positions, idx, axis=0)
            old_energy = self.local_energy(old_pos, remain_positions)
            new_energy = self.local_energy(new_pos, remain_positions)
            
            delta_E = new_energy - old_energy

            # Apply the log-sum-exp trick for numerical stability
            log_prob_accept = -self.beta * delta_E

            if log_prob_accept > 0:
                self.positions[idx] = new_pos   # if log_prob_accept is positive, accept the move
            else:
                prob_accept = np.exp(log_prob_accept)
                if np.random.rand() < prob_accept:
                    self.positions[idx] = new_pos


    def displace_and_rotate_particle(self):
        """
        Attemp to displace a particle
        """
        if self.number > 0:
            idx = np.random.randint(0, self.number)
            old_pos = self.positions[idx]
            unwrapped_pos = self.unwrap_pbc(old_pos)
            rotated_molecule = tls.RotMove_shift_linear(unwrapped_pos)
            new_pos = rotated_molecule + np.random.uniform(-self.maxdispl, self.maxdispl, 3)
            new_pos = self.wrap_pbc(new_pos)
            remain_positions = np.delete(self.positions, idx, axis=0)
            old_energy = self.local_energy(old_pos, remain_positions)
            new_energy = self.local_energy(new_pos, remain_positions)
            
            delta_E = new_energy - old_energy

            # Apply the log-sum-exp trick for numerical stability
            log_prob_accept = -self.beta * delta_E

            if log_prob_accept > 0:
                self.positions[idx] = new_pos   # if log_prob_accept is positive, accept the move
            else:
                prob_accept = np.exp(log_prob_accept)
                if np.random.rand() < prob_accept:
                    self.positions[idx] = new_pos
                
    def write_xyz(self, step):
        """
        Write the positions to an Extended XYZ file format for a given step.
        """
        
        with gzip.open(self.output_xyz + '.gz', 'at') as f:
            cell = f"{self.box_length_x} 0.0 0.0 0.0 {self.box_length_y} 0.0 0.0 0.0 {self.box_length_z}"
            f.write(f"{self.number*3}\n")
            f.write(f"Step {step} Lattice=\"{cell}\" Properties=species:S:1:pos:R:3\n")
            for i, pos in enumerate(self.positions):
                f.write(f"C {pos[0,0]:.6f} {pos[0,1]:.6f} {pos[0,2]:.6f}\n")
                f.write(f"O1 {pos[1,0]:.6f} {pos[1,1]:.6f} {pos[1,2]:.6f}\n")
                f.write(f"O2 {pos[2,0]:.6f} {pos[2,1]:.6f} {pos[2,2]:.6f}\n")

    def log(self, step):
        """
        Log the step, total number of particles, and the number of particles of each available type.
        """
        total_energy = self.total_energy()
        with open(self.logfile, 'a') as f:
            f.write(f"{step} {self.number} {total_energy}\n")

    def log_no_energy(self, step):
        """
        Log the step, total number of particles, and the number of particles of each available type.
        """
        with open(self.logfile, 'a') as f:
            f.write(f"{step} {self.number} \n")

    def write_log_header(self):
        """
        Write the header to the log file.
        """
        with open(self.logfile, 'w') as f:
            f.write("Step Total_number Energy\n")



        
