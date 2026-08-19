#ifndef GCMC_V2_SIMULATION_ENGINE_H
#define GCMC_V2_SIMULATION_ENGINE_H

#include "core_types.h"
#include <string>
#include <vector>
#include <map>
#include <random>
#include <zlib.h>

namespace gcmc_v2 {

class FastRNG {
private:
    uint64_t s[2];

    static uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    FastRNG(uint64_t seed = 42) {
        set_seed(seed);
    }

    void set_seed(uint64_t seed) {
        uint64_t z = (seed + 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s[0] = z ^ (z >> 31);
        z = (seed + 0x9e3779b97f4a7c15ULL * 2);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s[1] = z ^ (z >> 31);
        if (s[0] == 0 && s[1] == 0) {
            s[0] = 1;
        }
    }

    uint64_t next_u64() {
        const uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        const uint64_t result = s0 + s1;
        s1 ^= s0;
        s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
        s[1] = rotl(s1, 36);
        return result;
    }

    double uniform() {
        return (next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }

    double uniform_range(double min_val, double max_val) {
        return min_val + uniform() * (max_val - min_val);
    }

    int randint(int min_val, int max_val) {
        if (min_val >= max_val) {
            return min_val;
        }
        uint64_t range = (uint64_t)(max_val - min_val + 1);
        return min_val + (int)(next_u64() % range);
    }
};

class GCMCSimulationV2 {
public:
    MoleculeType mol_type = MoleculeType::NONE;

    double T = 500.0;
    double kB = KB_DEFAULT;
    double beta = 1.0 / (KB_DEFAULT * 500.0);
    double mu = -8.0;
    double box_x = 20.0, box_y = 20.0, box_z = 20.0;
    double volume = 8000.0;
    double global_rc = 10.0;
    double bond_length = 0.5;

    int max_steps = 1000;
    int equilibration_steps = 200;
    int output_interval = 100;
    bool print_energy = true;

    double weight_insert = 1.0;
    double weight_delete = 1.0;
    double weight_displace = 0.2;
    double weight_rotate = 0.2;
    double weight_mutate = 0.0;
    double weight_swap = 0.0;

    double prob_insert = 0.0;
    double prob_delete = 0.0;
    double prob_displace = 0.0;
    double prob_rotate = 0.0;
    double prob_mutate = 0.0;
    double prob_swap = 0.0;

    double maxdispl = 3.0;
    double maxrot = 0.2;

    std::string input_folder = ".";
    std::string output_xyz_path = "output.xyz";
    std::string logfile_path = "gcmc.log";
    std::string init_config_file = "";

    std::vector<Molecule> molecules;
    int number = 0;
    int number1 = 0;
    int number2 = 0;

    PairPotentialParams pair_potentials[3][3];
    ExternalPotentialParams ext_potentials[3];

    ElectrostaticsMode electrostatics_mode = ElectrostaticsMode::SHORT_RANGE;
    EwaldParams ewald_params;
    struct ComplexDouble { double re, im; };
    std::vector<ComplexDouble> rho_k;
    double site_charges[3] = {0.0, 0.0, 0.0};

    std::string type1_name = "H";
    std::string type2_name = "O";
    double mu1 = -8.0;
    double mu2 = -8.0;
    int nbins_x = 100;
    int density_output_interval = 100;
    std::vector<double> density_accum1;
    std::vector<double> density_accum2;
    int density_samples = 0;

    FastRNG rng;

    GCMCSimulationV2();
    ~GCMCSimulationV2();

    bool load_config_yaml(const std::string& yaml_path);
    bool load_initial_xyz(const std::string& xyz_path);

    void init_move_probabilities();
    void init_structure_factor();
    double get_site_charge(const Molecule& mol, int site_idx) const;
    double get_mol_self_energy(const Molecule& mol) const;
    void calc_mol_delta_rho_k(const Molecule& mol, std::vector<ComplexDouble>& delta, double sign) const;
    double calc_ewald_reciprocal_energy_delta(const std::vector<ComplexDouble>& delta) const;
    double ewald_reciprocal_energy() const;
    double calc_local_energy(const Molecule& mol, int exclude_idx = -1) const;
    double total_energy() const;

    Molecule generate_random_molecule();
    Molecule rotate_molecule(const Molecule& mol);

    void step();
    void step_single_type();
    void step_two_type();
    void step_abc();

    void run_simulation();
    void run_simulation_no_energy();

    void write_log_header();
    void write_log_entry(int step_num, double energy);
    void write_xyz_frame(gzFile gz_out, int step_num);
    void write_density_profile();

    Vec3 wrap(const Vec3& v) const {
        return v.wrap_pbc(box_x, box_y, box_z);
    }

    Vec3 min_image(const Vec3& v) const {
        return v.minimum_image(box_x, box_y, box_z);
    }
};

} // namespace gcmc_v2

#endif // GCMC_V2_SIMULATION_ENGINE_H
