#include "simulation_engine.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace gcmc_v2 {

// SPC/E reference geometry
static const Vec3 SPCE_O(0.0, 0.0, 0.0);
static const Vec3 SPCE_H1(0.0, 1.0, 0.0);
static const Vec3 SPCE_H2(0.94281615, -0.333313, 0.0);

GCMCSimulationV2::GCMCSimulationV2() {
    rng.set_seed(12345);
}

GCMCSimulationV2::~GCMCSimulationV2() {}

void GCMCSimulationV2::init_move_probabilities() {
    double total = weight_insert + weight_delete + weight_displace + weight_rotate + weight_mutate + weight_swap;
    if (total <= 0.0) total = 1.0;
    prob_insert = weight_insert / total;
    prob_delete = weight_delete / total;
    prob_displace = weight_displace / total;
    prob_rotate = weight_rotate / total;
    prob_mutate = weight_mutate / total;
    prob_swap = weight_swap / total;

    volume = box_x * box_y * box_z;
    beta = 1.0 / (kB * T);

    init_structure_factor();
}

double GCMCSimulationV2::get_site_charge(const Molecule& mol, int site_idx) const {
    if (mol_type == MoleculeType::TWO_TYPE_RPM) {
        return (mol.species_id == 0) ? site_charges[0] : site_charges[1];
    }
    if (mol_type == MoleculeType::ABC_DIPOLE) {
        if (site_idx == 0) return 0.0;
        return (site_idx == 1) ? site_charges[1] : site_charges[2];
    }
    if (site_idx >= 0 && site_idx < 3) return site_charges[site_idx];
    return 0.0;
}

double GCMCSimulationV2::get_mol_self_energy(const Molecule& mol) const {
    if (electrostatics_mode != ElectrostaticsMode::LONG_RANGE_EWALD) return 0.0;
    double sum_q2 = 0.0;
    for (int s = 0; s < mol.num_sites; ++s) {
        double q = get_site_charge(mol, s);
        sum_q2 += q * q;
    }
    return ewald_params.self_energy_per_q2 * sum_q2;
}

void GCMCSimulationV2::init_structure_factor() {
    if (electrostatics_mode != ElectrostaticsMode::LONG_RANGE_EWALD) return;
    ewald_params.init(box_x, box_y, box_z, ewald_params.prefactor);
    rho_k.assign(ewald_params.k_vectors.size(), {0.0, 0.0});
    for (int i = 0; i < number; ++i) {
        std::vector<ComplexDouble> delta;
        calc_mol_delta_rho_k(molecules[i], delta, 1.0);
        for (size_t k = 0; k < delta.size(); ++k) {
            rho_k[k].re += delta[k].re;
            rho_k[k].im += delta[k].im;
        }
    }
}

void GCMCSimulationV2::calc_mol_delta_rho_k(const Molecule& mol, std::vector<ComplexDouble>& delta, double sign) const {
    delta.resize(ewald_params.k_vectors.size());
    for (size_t m = 0; m < ewald_params.k_vectors.size(); ++m) {
        const auto& kv = ewald_params.k_vectors[m];
        double dre = 0.0, dim = 0.0;
        for (int s = 0; s < mol.num_sites; ++s) {
            double q = get_site_charge(mol, s);
            if (std::abs(q) < 1e-12) continue;
            double k_dot_r = kv.kx * mol.sites[s].x + kv.ky * mol.sites[s].y + kv.kz * mol.sites[s].z;
            dre += q * std::cos(k_dot_r);
            dim += q * std::sin(k_dot_r);
        }
        delta[m] = {sign * dre, sign * dim};
    }
}

double GCMCSimulationV2::calc_ewald_reciprocal_energy_delta(const std::vector<ComplexDouble>& delta) const {
    if (electrostatics_mode != ElectrostaticsMode::LONG_RANGE_EWALD) return 0.0;
    double delta_U = 0.0;
    for (size_t m = 0; m < ewald_params.k_vectors.size(); ++m) {
        double w = ewald_params.k_vectors[m].weight;
        double r_re = rho_k[m].re;
        double r_im = rho_k[m].im;
        double d_re = delta[m].re;
        double d_im = delta[m].im;
        delta_U += w * (2.0 * (r_re * d_re + r_im * d_im) + (d_re * d_re + d_im * d_im));
    }
    return delta_U;
}

double GCMCSimulationV2::ewald_reciprocal_energy() const {
    if (electrostatics_mode != ElectrostaticsMode::LONG_RANGE_EWALD) return 0.0;
    double u_recip = 0.0;
    for (size_t m = 0; m < ewald_params.k_vectors.size(); ++m) {
        double w = ewald_params.k_vectors[m].weight;
        u_recip += w * (rho_k[m].re * rho_k[m].re + rho_k[m].im * rho_k[m].im);
    }
    double u_self = 0.0;
    for (int i = 0; i < number; ++i) {
        u_self += get_mol_self_energy(molecules[i]);
    }
    return u_recip - u_self;
}

double GCMCSimulationV2::calc_local_energy(const Molecule& mol, int exclude_idx) const {
    double energy = 0.0;

    for (int i = 0; i < number; ++i) {
        if (i == exclude_idx) continue;
        const Molecule& other = molecules[i];

        for (int s1 = 0; s1 < mol.num_sites; ++s1) {
            Vec3 pos1 = mol.sites[s1];
            int type1 = (mol_type == MoleculeType::TWO_TYPE_RPM) ? mol.species_id : s1;

            for (int s2 = 0; s2 < other.num_sites; ++s2) {
                Vec3 pos2 = other.sites[s2];
                int type2 = (mol_type == MoleculeType::TWO_TYPE_RPM) ? other.species_id : s2;

                Vec3 delta = (pos2 - pos1).minimum_image(box_x, box_y, box_z);
                double r_sq = delta.norm_sq();
                if (r_sq < global_rc * global_rc) {
                    double r = std::sqrt(r_sq);
                    energy += pair_potentials[type1][type2].calculate(r);
                }
            }
        }
    }

    // External potential
    for (int s = 0; s < mol.num_sites; ++s) {
        int ext_type = (mol_type == MoleculeType::TWO_TYPE_RPM) ? mol.species_id : s;
        energy += ext_potentials[ext_type].calculate(mol.sites[s]);
    }

    return energy;
}

double GCMCSimulationV2::total_energy() const {
    if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD && rho_k.empty() && number > 0) {
        const_cast<GCMCSimulationV2*>(this)->init_structure_factor();
    }

    double e_pair = 0.0;
    double e_ext = 0.0;

    for (int i = 0; i < number; ++i) {
        const Molecule& mol_i = molecules[i];

        // Pairwise interactions with j > i
        for (int j = i + 1; j < number; ++j) {
            const Molecule& mol_j = molecules[j];

            for (int s1 = 0; s1 < mol_i.num_sites; ++s1) {
                int type1 = (mol_type == MoleculeType::TWO_TYPE_RPM) ? mol_i.species_id : s1;
                Vec3 pos1 = mol_i.sites[s1];

                for (int s2 = 0; s2 < mol_j.num_sites; ++s2) {
                    int type2 = (mol_type == MoleculeType::TWO_TYPE_RPM) ? mol_j.species_id : s2;
                    Vec3 pos2 = mol_j.sites[s2];

                    Vec3 delta = (pos2 - pos1).minimum_image(box_x, box_y, box_z);
                    double r_sq = delta.norm_sq();
                    if (r_sq < global_rc * global_rc) {
                        double r = std::sqrt(r_sq);
                        e_pair += pair_potentials[type1][type2].calculate(r);
                    }
                }
            }
        }

        // External potential
        for (int s = 0; s < mol_i.num_sites; ++s) {
            int ext_type = (mol_type == MoleculeType::TWO_TYPE_RPM) ? mol_i.species_id : s;
            e_ext += ext_potentials[ext_type].calculate(mol_i.sites[s]);
        }
    }

    double e_ewald = (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) ? ewald_reciprocal_energy() : 0.0;
    return e_pair + e_ext + e_ewald;
}

Molecule GCMCSimulationV2::generate_random_molecule() {
    Molecule mol;
    Vec3 center(rng.uniform_range(0, box_x), rng.uniform_range(0, box_y), rng.uniform_range(0, box_z));

    if (mol_type == MoleculeType::ABC_DIPOLE) {
        mol.num_sites = 3;
        // Random unit vector
        double u1 = rng.uniform();
        double u2 = rng.uniform();
        double theta = std::acos(2.0 * u1 - 1.0);
        double phi = 2.0 * PI * u2;
        Vec3 dir(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));

        mol.sites[0] = wrap(center);
        mol.sites[1] = wrap(center + dir * bond_length);
        mol.sites[2] = wrap(center - dir * bond_length);
    } else if (mol_type == MoleculeType::H2O_SPCE) {
        mol.num_sites = 3;
        // Random quaternion rotation
        double u1 = rng.uniform();
        double u2 = rng.uniform();
        double u3 = rng.uniform();
        Quaternion q(
            std::sqrt(1.0 - u1) * std::sin(2.0 * PI * u2),
            std::sqrt(1.0 - u1) * std::cos(2.0 * PI * u2),
            std::sqrt(u1) * std::sin(2.0 * PI * u3),
            std::sqrt(u1) * std::cos(2.0 * PI * u3)
        );

        mol.sites[0] = wrap(center + q.rotate(SPCE_O));
        mol.sites[1] = wrap(center + q.rotate(SPCE_H1));
        mol.sites[2] = wrap(center + q.rotate(SPCE_H2));
    } else if (mol_type == MoleculeType::TWO_TYPE_RPM) {
        mol.num_sites = 1;
        mol.species_id = (rng.uniform() < 0.5) ? 0 : 1;
        mol.sites[0] = wrap(center);
    } else {
        mol.num_sites = 1;
        mol.species_id = 0;
        mol.sites[0] = wrap(center);
    }

    return mol;
}

Molecule GCMCSimulationV2::rotate_molecule(const Molecule& mol) {
    Molecule res = mol;
    Vec3 center = mol.sites[0];

    if (mol_type == MoleculeType::ABC_DIPOLE) {
        // Small random angular displacement
        double d_theta = rng.uniform_range(-0.2, 0.2);
        double d_phi = rng.uniform_range(-0.2, 0.2);
        double d_psi = rng.uniform_range(-0.2, 0.2);
        Quaternion dq(std::cos(d_theta), std::sin(d_phi), std::sin(d_psi), std::sin(d_theta));
        double norm = std::sqrt(dq.w * dq.w + dq.x * dq.x + dq.y * dq.y + dq.z * dq.z);
        dq = Quaternion(dq.w / norm, dq.x / norm, dq.y / norm, dq.z / norm);

        Vec3 arm = (mol.sites[1] - center).minimum_image(box_x, box_y, box_z);
        Vec3 new_arm = dq.rotate(arm);
        double len = new_arm.norm();
        if (len > 1e-12) new_arm = new_arm * (bond_length / len);

        res.sites[0] = center;
        res.sites[1] = wrap(center + new_arm);
        res.sites[2] = wrap(center - new_arm);
    } else if (mol_type == MoleculeType::H2O_SPCE) {
        double d_theta = rng.uniform_range(-0.2, 0.2);
        double d_phi = rng.uniform_range(-0.2, 0.2);
        double d_psi = rng.uniform_range(-0.2, 0.2);
        Quaternion dq(std::cos(d_theta), std::sin(d_phi), std::sin(d_psi), std::sin(d_theta));
        double norm = std::sqrt(dq.w * dq.w + dq.x * dq.x + dq.y * dq.y + dq.z * dq.z);
        dq = Quaternion(dq.w / norm, dq.x / norm, dq.y / norm, dq.z / norm);

        Vec3 arm1 = (mol.sites[1] - center).minimum_image(box_x, box_y, box_z);
        Vec3 arm2 = (mol.sites[2] - center).minimum_image(box_x, box_y, box_z);

        res.sites[0] = center;
        res.sites[1] = wrap(center + dq.rotate(arm1));
        res.sites[2] = wrap(center + dq.rotate(arm2));
    }
    return res;
}

void GCMCSimulationV2::step_abc() {
    double r = rng.uniform();

    if (r < prob_insert) {
        // Insert
        Molecule new_mol = generate_random_molecule();
        double delta_E = calc_local_energy(new_mol);
        std::vector<ComplexDouble> delta_k;
        if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
            calc_mol_delta_rho_k(new_mol, delta_k, 1.0);
            delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
            delta_E -= get_mol_self_energy(new_mol);
        }
        double prob = std::exp(-beta * (delta_E - mu)) * volume / (number + 1);
        if (rng.uniform() < prob) {
            molecules.push_back(new_mol);
            number++;
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                for (size_t k = 0; k < delta_k.size(); ++k) {
                    rho_k[k].re += delta_k[k].re;
                    rho_k[k].im += delta_k[k].im;
                }
            }
        }
    } else if (r < prob_insert + prob_delete) {
        // Delete
        if (number > 0) {
            int idx = rng.randint(0, number - 1);
            double delta_E = -calc_local_energy(molecules[idx], idx);
            std::vector<ComplexDouble> delta_k;
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                calc_mol_delta_rho_k(molecules[idx], delta_k, -1.0);
                delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
                delta_E += get_mol_self_energy(molecules[idx]);
            }
            double log_prob = -beta * (delta_E + mu) + std::log(static_cast<double>(number)) - std::log(volume);
            double prob = (log_prob < 700.0) ? std::exp(log_prob) : 0.0;
            if (rng.uniform() < prob) {
                molecules.erase(molecules.begin() + idx);
                number--;
                if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                    for (size_t k = 0; k < delta_k.size(); ++k) {
                        rho_k[k].re += delta_k[k].re;
                        rho_k[k].im += delta_k[k].im;
                    }
                }
            }
        }
    } else if (r < prob_insert + prob_delete + prob_displace) {
        // Displace
        if (number > 0) {
            int idx = rng.randint(0, number - 1);
            Molecule old_mol = molecules[idx];
            Vec3 displ(
                rng.uniform_range(-maxdispl, maxdispl),
                rng.uniform_range(-maxdispl, maxdispl),
                rng.uniform_range(-maxdispl, maxdispl)
            );
            Molecule new_mol = old_mol;
            for (int s = 0; s < new_mol.num_sites; ++s) {
                new_mol.sites[s] = wrap(old_mol.sites[s] + displ);
            }
            double old_e = calc_local_energy(old_mol, idx);
            double new_e = calc_local_energy(new_mol, idx);
            double delta_E = new_e - old_e;
            std::vector<ComplexDouble> delta_k;
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                std::vector<ComplexDouble> d_new, d_old;
                calc_mol_delta_rho_k(new_mol, d_new, 1.0);
                calc_mol_delta_rho_k(old_mol, d_old, -1.0);
                delta_k.resize(d_new.size());
                for (size_t k = 0; k < d_new.size(); ++k) {
                    delta_k[k] = {d_new[k].re + d_old[k].re, d_new[k].im + d_old[k].im};
                }
                delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
            }
            double log_p = -beta * delta_E;
            if (log_p > 0.0 || rng.uniform() < std::exp(log_p)) {
                molecules[idx] = new_mol;
                if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                    for (size_t k = 0; k < delta_k.size(); ++k) {
                        rho_k[k].re += delta_k[k].re;
                        rho_k[k].im += delta_k[k].im;
                    }
                }
            }
        }
    } else {
        // Rotate
        if (number > 0) {
            int idx = rng.randint(0, number - 1);
            Molecule old_mol = molecules[idx];
            Molecule new_mol = rotate_molecule(old_mol);
            double old_e = calc_local_energy(old_mol, idx);
            double new_e = calc_local_energy(new_mol, idx);
            double delta_E = new_e - old_e;
            std::vector<ComplexDouble> delta_k;
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                std::vector<ComplexDouble> d_new, d_old;
                calc_mol_delta_rho_k(new_mol, d_new, 1.0);
                calc_mol_delta_rho_k(old_mol, d_old, -1.0);
                delta_k.resize(d_new.size());
                for (size_t k = 0; k < d_new.size(); ++k) {
                    delta_k[k] = {d_new[k].re + d_old[k].re, d_new[k].im + d_old[k].im};
                }
                delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
            }
            double log_p = -beta * delta_E;
            if (log_p > 0.0 || rng.uniform() < std::exp(log_p)) {
                molecules[idx] = new_mol;
                if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                    for (size_t k = 0; k < delta_k.size(); ++k) {
                        rho_k[k].re += delta_k[k].re;
                        rho_k[k].im += delta_k[k].im;
                    }
                }
            }
        }
    }
}

void GCMCSimulationV2::step_h2o() {
    step_abc(); // Same move logic with H2O 3D rotation
}

void GCMCSimulationV2::step_two_type() {
    double r = rng.uniform();

    if (r < prob_insert) {
        // Insert
        Molecule new_mol;
        new_mol.num_sites = 1;
        new_mol.species_id = (rng.uniform() < 0.5) ? 0 : 1;
        new_mol.sites[0] = Vec3(rng.uniform_range(0, box_x), rng.uniform_range(0, box_y), rng.uniform_range(0, box_z));

        double target_mu = (new_mol.species_id == 0) ? mu1 : mu2;
        int target_num = (new_mol.species_id == 0) ? number1 : number2;

        double delta_E = calc_local_energy(new_mol);
        std::vector<ComplexDouble> delta_k;
        if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
            calc_mol_delta_rho_k(new_mol, delta_k, 1.0);
            delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
            delta_E -= get_mol_self_energy(new_mol);
        }
        double prob = std::exp(-beta * (delta_E - target_mu)) * volume / (target_num + 1);
        if (rng.uniform() < prob) {
            molecules.push_back(new_mol);
            number++;
            if (new_mol.species_id == 0) number1++;
            else number2++;
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                for (size_t k = 0; k < delta_k.size(); ++k) {
                    rho_k[k].re += delta_k[k].re;
                    rho_k[k].im += delta_k[k].im;
                }
            }
        }
    } else if (r < prob_insert + prob_delete) {
        // Delete
        if (number > 0) {
            int idx = rng.randint(0, number - 1);
            const Molecule& del_mol = molecules[idx];
            double target_mu = (del_mol.species_id == 0) ? mu1 : mu2;
            int target_num = (del_mol.species_id == 0) ? number1 : number2;

            double delta_E = -calc_local_energy(del_mol, idx);
            std::vector<ComplexDouble> delta_k;
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                calc_mol_delta_rho_k(del_mol, delta_k, -1.0);
                delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
                delta_E += get_mol_self_energy(del_mol);
            }
            double log_prob = -beta * (delta_E + target_mu) + std::log(static_cast<double>(target_num)) - std::log(volume);
            double prob = (log_prob < 700.0) ? std::exp(log_prob) : 0.0;
            if (rng.uniform() < prob) {
                if (del_mol.species_id == 0) number1--;
                else number2--;
                molecules.erase(molecules.begin() + idx);
                number--;
                if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                    for (size_t k = 0; k < delta_k.size(); ++k) {
                        rho_k[k].re += delta_k[k].re;
                        rho_k[k].im += delta_k[k].im;
                    }
                }
            }
        }
    } else if (r < prob_insert + prob_delete + prob_displace) {
        // Displace
        if (number > 0) {
            int idx = rng.randint(0, number - 1);
            Molecule old_mol = molecules[idx];
            Vec3 displ(
                rng.uniform_range(-maxdispl, maxdispl),
                rng.uniform_range(-maxdispl, maxdispl),
                rng.uniform_range(-maxdispl, maxdispl)
            );
            Molecule new_mol = old_mol;
            new_mol.sites[0] = wrap(old_mol.sites[0] + displ);

            double old_e = calc_local_energy(old_mol, idx);
            double new_e = calc_local_energy(new_mol, idx);
            double delta_E = new_e - old_e;
            std::vector<ComplexDouble> delta_k;
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                std::vector<ComplexDouble> d_new, d_old;
                calc_mol_delta_rho_k(new_mol, d_new, 1.0);
                calc_mol_delta_rho_k(old_mol, d_old, -1.0);
                delta_k.resize(d_new.size());
                for (size_t k = 0; k < d_new.size(); ++k) {
                    delta_k[k] = {d_new[k].re + d_old[k].re, d_new[k].im + d_old[k].im};
                }
                delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
            }
            double log_p = -beta * delta_E;
            if (log_p > 0.0 || rng.uniform() < std::exp(log_p)) {
                molecules[idx] = new_mol;
                if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                    for (size_t k = 0; k < delta_k.size(); ++k) {
                        rho_k[k].re += delta_k[k].re;
                        rho_k[k].im += delta_k[k].im;
                    }
                }
            }
        }
    } else {
        // Mutate / Swap species type
        if (number > 0) {
            int idx = rng.randint(0, number - 1);
            Molecule old_mol = molecules[idx];
            Molecule new_mol = old_mol;
            new_mol.species_id = 1 - old_mol.species_id;

            double target_mu_new = (new_mol.species_id == 0) ? mu1 : mu2;
            double target_mu_old = (old_mol.species_id == 0) ? mu1 : mu2;

            double old_e = calc_local_energy(old_mol, idx);
            double new_e = calc_local_energy(new_mol, idx);
            double delta_E = new_e - old_e;
            double delta_mu = target_mu_new - target_mu_old;

            double log_p = -beta * (delta_E - delta_mu);
            if (log_p > 0.0 || rng.uniform() < std::exp(log_p)) {
                molecules[idx] = new_mol;
                if (old_mol.species_id == 0) {
                    number1--;
                    number2++;
                } else {
                    number2--;
                    number1++;
                }
            }
        }
    }
}

void GCMCSimulationV2::step() {
    if (mol_type == MoleculeType::ABC_DIPOLE) step_abc();
    else if (mol_type == MoleculeType::H2O_SPCE) step_h2o();
    else if (mol_type == MoleculeType::TWO_TYPE_RPM) step_two_type();
    else step_abc();
}

void GCMCSimulationV2::write_log_header() {
    std::string path = input_folder + "/" + logfile_path;
    std::ofstream ofs(path);
    if (mol_type == MoleculeType::TWO_TYPE_RPM) {
        ofs << "Step Total_number " << type1_name << " " << type2_name << "\n";
    } else {
        ofs << "Step Total_number Energy\n";
    }
}

void GCMCSimulationV2::write_log_entry(int step_num, double energy) {
    std::string path = input_folder + "/" + logfile_path;
    std::ofstream ofs(path, std::ios::app);
    if (mol_type == MoleculeType::TWO_TYPE_RPM) {
        ofs << step_num << " " << number << " " << number1 << " " << number2 << "\n";
    } else {
        ofs << step_num << " " << number << " " << std::setprecision(16) << energy << "\n";
    }
}

void GCMCSimulationV2::write_xyz_frame(gzFile gz_out, int step_num) {
    if (!gz_out) return;
    int total_atoms = 0;
    for (const auto& m : molecules) total_atoms += m.num_sites;

    std::ostringstream ss;
    ss << total_atoms << "\n";
    ss << "Step " << step_num << " Lattice=\"" << box_x << " 0.0 0.0 0.0 "
       << box_y << " 0.0 0.0 0.0 " << box_z << "\" Properties=species:S:1:pos:R:3\n";

    for (const auto& m : molecules) {
        if (mol_type == MoleculeType::ABC_DIPOLE) {
            ss << "A " << m.sites[0].x << " " << m.sites[0].y << " " << m.sites[0].z << "\n";
            ss << "B " << m.sites[1].x << " " << m.sites[1].y << " " << m.sites[1].z << "\n";
            ss << "C " << m.sites[2].x << " " << m.sites[2].y << " " << m.sites[2].z << "\n";
        } else if (mol_type == MoleculeType::H2O_SPCE) {
            ss << "O " << m.sites[0].x << " " << m.sites[0].y << " " << m.sites[0].z << "\n";
            ss << "H1 " << m.sites[1].x << " " << m.sites[1].y << " " << m.sites[1].z << "\n";
            ss << "H2 " << m.sites[2].x << " " << m.sites[2].y << " " << m.sites[2].z << "\n";
        } else if (mol_type == MoleculeType::TWO_TYPE_RPM) {
            std::string name = (m.species_id == 0) ? type1_name : type2_name;
            ss << name << " " << m.sites[0].x << " " << m.sites[0].y << " " << m.sites[0].z << "\n";
        }
    }
    std::string str = ss.str();
    gzwrite(gz_out, str.c_str(), str.size());
}

void GCMCSimulationV2::write_density_profile() {
    if (nbins_x <= 0 || density_samples <= 0) return;
    std::string path = input_folder + "/density_x.dat";
    std::ofstream ofs(path);
    ofs << "# x rho1 rho2\n";
    double dx = box_x / nbins_x;
    for (int b = 0; b < nbins_x; ++b) {
        double x_center = (b + 0.5) * dx;
        double rho1 = (density_accum1[b] / density_samples) / (dx * box_y * box_z);
        double rho2 = (density_accum2[b] / density_samples) / (dx * box_y * box_z);
        ofs << x_center << " " << rho1 << " " << rho2 << "\n";
    }
}

void GCMCSimulationV2::run_simulation() {
    init_move_probabilities();
    write_log_header();

    if (mol_type == MoleculeType::TWO_TYPE_RPM && nbins_x > 0) {
        density_accum1.assign(nbins_x, 0.0);
        density_accum2.assign(nbins_x, 0.0);
        density_samples = 0;
    }

    std::string xyz_path = input_folder + "/" + output_xyz_path + ".gz";
    gzFile gz_out = gzopen(xyz_path.c_str(), "wb");

    // Initial state logging
    write_log_entry(0, total_energy());
    if (gz_out) write_xyz_frame(gz_out, 0);

    for (int s = 1; s <= max_steps; ++s) {
        step();

        // Sample density
        if (mol_type == MoleculeType::TWO_TYPE_RPM && s > equilibration_steps && nbins_x > 0 && (s % density_output_interval == 0)) {
            double dx = box_x / nbins_x;
            for (const auto& m : molecules) {
                int bin = static_cast<int>(m.sites[0].x / dx);
                if (bin >= 0 && bin < nbins_x) {
                    if (m.species_id == 0) density_accum1[bin] += 1.0;
                    else density_accum2[bin] += 1.0;
                }
            }
            density_samples++;
        }

        if (s % output_interval == 0 || s == max_steps) {
            double e = print_energy ? total_energy() : 0.0;
            write_log_entry(s, e);
            if (gz_out && s >= equilibration_steps) {
                write_xyz_frame(gz_out, s);
            }
        }
    }

    if (gz_out) gzclose(gz_out);
    if (mol_type == MoleculeType::TWO_TYPE_RPM && nbins_x > 0) {
        write_density_profile();
    }
}

void GCMCSimulationV2::run_simulation_no_energy() {
    print_energy = false;
    run_simulation();
}

} // namespace gcmc_v2
