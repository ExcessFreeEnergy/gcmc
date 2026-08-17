#ifndef GCMC_V2_CUDA_GCMC_H
#define GCMC_V2_CUDA_GCMC_H

#include <cstdint>
#include <vector>

namespace gcmc_v2 {

constexpr int MAX_MOLECULES_PER_BOX = 1024;
constexpr int MAX_SITES_PER_MOL = 3;

struct CUDAPairParams {
    int kind; // 0=NONE, 1=LJ, 2=WCA, 3=HS, 4=HS_C, 5=LJ_C
    float epsilon_lj;
    float sigma_lj;
    float rc;
    float epsilon_c;
    float q1, q2;
    float kappa_inv;
    float diameter;
    float prefactor;
    float shift_lj;
};

struct CUDAExternalParams {
    int kind;
    float low, high, width, L;
    float epsilon, sigma, cutoff, shift, q;
    float A1, A2, A3, A4;
    float phi1, phi2, phi3, phi4;
    float q_A1, q_A2, q_A3, q_A4;
    float q_phi1, q_phi2, q_phi3, q_phi4;
};

constexpr int MAX_EWALD_K_VECTORS = 128;

struct CUDAEwaldKVector {
    float kx, ky, kz, weight;
};

struct CUDABoxConfig {
    int mol_type; // 0=NONE, 1=SINGLE, 2=TWO_TYPE, 3=ABC, 4=H2O
    float box_x, box_y, box_z;
    float beta;
    float mu1, mu2;
    float bond_length;
    float maxdispl;
    float prob_insert, prob_delete, prob_displace, prob_rotate, prob_mutate;
    float global_rc;

    int electrostatics_mode; // 0=SR (default), 1=LR Ewald
    float ewald_alpha;
    float ewald_self_per_q2;
    int num_k_vectors;
    CUDAEwaldKVector k_vectors[MAX_EWALD_K_VECTORS];
    float site_charges[3];

    CUDAPairParams pair_potentials[3][3];
    CUDAExternalParams ext_potentials[3];
};

struct CUDABoxOutput {
    int final_num_molecules;
    int final_num1;
    int final_num2;
    float avg_num_molecules;
    float avg_energy;
    float final_energy;
};

// Host API to launch batched CUDA Monte Carlo simulations
extern "C" {
    bool cuda_is_available();
    int cuda_get_device_count();
    
    void run_cuda_batch_gcmc(
        int num_boxes,
        int num_steps,
        int equilibration_steps,
        const CUDABoxConfig* h_configs,
        CUDABoxOutput* h_outputs,
        uint64_t seed
    );
}

} // namespace gcmc_v2

#endif // GCMC_V2_CUDA_GCMC_H
