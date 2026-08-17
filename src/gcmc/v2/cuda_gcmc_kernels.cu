#include "cuda_gcmc.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>
#include <cmath>

namespace gcmc_v2 {

__device__ inline float3 make_f3(float x, float y, float z) {
    float3 v; v.x = x; v.y = y; v.z = z; return v;
}

__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_f3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_f3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(const float3& a, float s) {
    return make_f3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 min_image_dev(const float3& delta, float lx, float ly, float lz) {
    return make_f3(
        delta.x - lx * rintf(delta.x / lx),
        delta.y - ly * rintf(delta.y / ly),
        delta.z - lz * rintf(delta.z / lz)
    );
}

__device__ inline float3 wrap_pbc_dev(const float3& pos, float lx, float ly, float lz) {
    return make_f3(
        pos.x - lx * floorf(pos.x / lx),
        pos.y - ly * floorf(pos.y / ly),
        pos.z - lz * floorf(pos.z / lz)
    );
}

// Fast XORShift32 RNG per CUDA thread
struct DevRNG {
    uint32_t state;

    __device__ inline void init(uint32_t seed) {
        state = seed ? seed : 123456789;
    }

    __device__ inline uint32_t next_u32() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }

    __device__ inline float uniform() {
        return (next_u32() & 0x00FFFFFF) * (1.0f / 16777216.0f);
    }

    __device__ inline float uniform_range(float a, float b) {
        return a + uniform() * (b - a);
    }

    __device__ inline int randint(int min_val, int max_val) {
        if (min_val >= max_val) return min_val;
        return min_val + (int)(next_u32() % (uint32_t)(max_val - min_val + 1));
    }
};

__device__ inline float calc_pair_energy_dev(const CUDAPairParams& p, float r) {
    if (r >= p.rc) return 0.0f;
    switch (p.kind) {
        case 1: { // LJ
            float inv_r = 1.0f / r;
            float r2 = (p.sigma_lj * inv_r) * (p.sigma_lj * inv_r);
            float r6 = r2 * r2 * r2;
            return 4.0f * p.epsilon_lj * (r6 * r6 - r6) - p.shift_lj;
        }
        case 2: { // WCA
            float r_wca = 1.12246205f * p.sigma_lj;
            if (r >= r_wca) return 0.0f;
            float inv_r = 1.0f / r;
            float r2 = (p.sigma_lj * inv_r) * (p.sigma_lj * inv_r);
            float r6 = r2 * r2 * r2;
            return 4.0f * p.epsilon_lj * (r6 * r6 - r6) + p.epsilon_lj;
        }
        case 3: { // HS
            return (r < p.sigma_lj) ? 1.0e30f : 0.0f;
        }
        case 4: { // HS+C
            if (r < p.diameter) return 1.0e30f;
            return p.prefactor * p.q1 * p.q2 * erfcf(r / p.kappa_inv) / r;
        }
        case 5: { // LJ+C
            float inv_r = 1.0f / r;
            float r2 = (p.sigma_lj * inv_r) * (p.sigma_lj * inv_r);
            float r6 = r2 * r2 * r2;
            float u_lj = 4.0f * p.epsilon_lj * (r6 * r6 - r6) - p.shift_lj;
            float u_c = p.prefactor * p.q1 * p.q2 * erfcf(r / p.kappa_inv) * inv_r;
            return u_lj + u_c;
        }
        default:
            return 0.0f;
    }
}

__device__ inline float calc_ext_energy_dev(const CUDAExternalParams& ep, const float3& pos) {
    if (ep.kind == 0) return 0.0f;
    float x = pos.x;

    if (ep.kind == 1) { // Wall
        if (x < ep.width || x > ep.L - ep.width) return 1.0e30f;
        return 0.0f;
    }
    if (ep.kind == 2) { // Slit
        if (x < ep.low || x > ep.high) return 1.0e30f;
        return 0.0f;
    }
    if (ep.kind == 5) { // Training potential cosine
        if (x < ep.low || x > ep.high) return 1.0e30f;
        float arg = 2.0f * CUDART_PI_F * x / ep.L;
        float sines = ep.A1 * sinf(arg * 1.0f + ep.phi1) +
                      ep.A2 * sinf(arg * 2.0f + ep.phi2) +
                      ep.A3 * sinf(arg * 3.0f + ep.phi3) +
                      ep.A4 * sinf(arg * 4.0f + ep.phi4);
        float sines_q = ep.q_A1 * cosf(arg * 1.0f + ep.q_phi1) +
                        ep.q_A2 * cosf(arg * 2.0f + ep.q_phi2) +
                        ep.q_A3 * cosf(arg * 3.0f + ep.q_phi3) +
                        ep.q_A4 * cosf(arg * 4.0f + ep.q_phi4);

        float r_low = x - ep.low;
        float r_high = ep.high - x;
        float e_low = 0.0f, e_high = 0.0f;
        if (r_low < ep.cutoff && r_low > 0.0f) {
            float r3 = powf(ep.sigma / r_low, 3.0f);
            e_low = ep.epsilon * ((2.0f / 15.0f) * r3 * r3 * r3 - r3) - ep.shift;
        }
        if (r_high < ep.cutoff && r_high > 0.0f) {
            float r3 = powf(ep.sigma / r_high, 3.0f);
            e_high = ep.epsilon * ((2.0f / 15.0f) * r3 * r3 * r3 - r3) - ep.shift;
        }
        return sines + sines_q * ep.q + e_low + e_high;
    }
    return 0.0f;
}

__global__ void batch_gcmc_kernel(
    int num_steps,
    int equilibration_steps,
    const CUDABoxConfig* d_configs,
    CUDABoxOutput* d_outputs,
    uint64_t base_seed
) {
    int box_id = blockIdx.x;
    const CUDABoxConfig& cfg = d_configs[box_id];

    // Local coordinates in shared memory or registers
    __shared__ float3 s_pos[MAX_MOLECULES_PER_BOX][MAX_SITES_PER_MOL];
    __shared__ int s_species[MAX_MOLECULES_PER_BOX];
    __shared__ int s_num_molecules;
    __shared__ int s_num1;
    __shared__ int s_num2;
    __shared__ double s_accum_N;
    __shared__ int s_accum_samples;

    DevRNG rng;
    rng.init(static_cast<uint32_t>(base_seed + box_id * 104729 + threadIdx.x * 7919));

    if (threadIdx.x == 0) {
        s_num_molecules = 0;
        s_num1 = 0;
        s_num2 = 0;
        s_accum_N = 0.0;
        s_accum_samples = 0;
    }
    __syncthreads();

    float volume = cfg.box_x * cfg.box_y * cfg.box_z;
    int num_sites_per_mol = (cfg.mol_type == 3 || cfg.mol_type == 4) ? 3 : 1;

    for (int step = 0; step < num_steps; ++step) {
        if (threadIdx.x == 0) {
            float r_move = rng.uniform();

            if (r_move < cfg.prob_insert && s_num_molecules < MAX_MOLECULES_PER_BOX - 1) {
                // Generate trial molecule
                float3 center = make_f3(
                    rng.uniform_range(0, cfg.box_x),
                    rng.uniform_range(0, cfg.box_y),
                    rng.uniform_range(0, cfg.box_z)
                );
                float3 trial_sites[3];
                int trial_species = 0;

                if (cfg.mol_type == 3) { // ABC Dipole
                    float u1 = rng.uniform(), u2 = rng.uniform();
                    float theta = acosf(2.0f * u1 - 1.0f);
                    float phi = 2.0f * CUDART_PI_F * u2;
                    float3 dir = make_f3(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta));
                    trial_sites[0] = wrap_pbc_dev(center, cfg.box_x, cfg.box_y, cfg.box_z);
                    trial_sites[1] = wrap_pbc_dev(center + dir * cfg.bond_length, cfg.box_x, cfg.box_y, cfg.box_z);
                    trial_sites[2] = wrap_pbc_dev(center - dir * cfg.bond_length, cfg.box_x, cfg.box_y, cfg.box_z);
                } else if (cfg.mol_type == 2) { // TwoType RPM
                    trial_species = (rng.uniform() < 0.5f) ? 0 : 1;
                    trial_sites[0] = wrap_pbc_dev(center, cfg.box_x, cfg.box_y, cfg.box_z);
                } else {
                    trial_sites[0] = wrap_pbc_dev(center, cfg.box_x, cfg.box_y, cfg.box_z);
                }

                // Compute local energy
                float delta_E = 0.0f;
                for (int i = 0; i < s_num_molecules; ++i) {
                    for (int s1 = 0; s1 < num_sites_per_mol; ++s1) {
                        int t1 = (cfg.mol_type == 2) ? trial_species : s1;
                        for (int s2 = 0; s2 < num_sites_per_mol; ++s2) {
                            int t2 = (cfg.mol_type == 2) ? s_species[i] : s2;
                            float3 d = min_image_dev(s_pos[i][s2] - trial_sites[s1], cfg.box_x, cfg.box_y, cfg.box_z);
                            float r = sqrtf(dot(d, d));
                            delta_E += calc_pair_energy_dev(cfg.pair_potentials[t1][t2], r);
                        }
                    }
                }
                for (int s = 0; s < num_sites_per_mol; ++s) {
                    int et = (cfg.mol_type == 2) ? trial_species : s;
                    delta_E += calc_ext_energy_dev(cfg.ext_potentials[et], trial_sites[s]);
                }

                float target_mu = (trial_species == 0) ? cfg.mu1 : cfg.mu2;
                int target_N = (cfg.mol_type == 2) ? ((trial_species == 0) ? s_num1 : s_num2) : s_num_molecules;
                float prob = expf(-cfg.beta * (delta_E - target_mu)) * volume / (target_N + 1);

                if (rng.uniform() < prob) {
                    for (int s = 0; s < num_sites_per_mol; ++s) {
                        s_pos[s_num_molecules][s] = trial_sites[s];
                    }
                    s_species[s_num_molecules] = trial_species;
                    s_num_molecules++;
                    if (trial_species == 0) s_num1++;
                    else s_num2++;
                }
            } else if (r_move < cfg.prob_insert + cfg.prob_delete && s_num_molecules > 0) {
                // Delete
                int idx = rng.randint(0, s_num_molecules - 1);
                int del_species = s_species[idx];
                float delta_E = 0.0f;

                for (int i = 0; i < s_num_molecules; ++i) {
                    if (i == idx) continue;
                    for (int s1 = 0; s1 < num_sites_per_mol; ++s1) {
                        int t1 = (cfg.mol_type == 2) ? del_species : s1;
                        for (int s2 = 0; s2 < num_sites_per_mol; ++s2) {
                            int t2 = (cfg.mol_type == 2) ? s_species[i] : s2;
                            float3 d = min_image_dev(s_pos[i][s2] - s_pos[idx][s1], cfg.box_x, cfg.box_y, cfg.box_z);
                            float r = sqrtf(dot(d, d));
                            delta_E -= calc_pair_energy_dev(cfg.pair_potentials[t1][t2], r);
                        }
                    }
                }
                for (int s = 0; s < num_sites_per_mol; ++s) {
                    int et = (cfg.mol_type == 2) ? del_species : s;
                    delta_E -= calc_ext_energy_dev(cfg.ext_potentials[et], s_pos[idx][s]);
                }

                float target_mu = (del_species == 0) ? cfg.mu1 : cfg.mu2;
                int target_N = (cfg.mol_type == 2) ? ((del_species == 0) ? s_num1 : s_num2) : s_num_molecules;
                float log_p = -cfg.beta * (delta_E + target_mu) + logf((float)target_N) - logf(volume);
                float prob = (log_p < 80.0f) ? expf(log_p) : 0.0f;

                if (rng.uniform() < prob) {
                    if (idx != s_num_molecules - 1) {
                        for (int s = 0; s < num_sites_per_mol; ++s) {
                            s_pos[idx][s] = s_pos[s_num_molecules - 1][s];
                        }
                        s_species[idx] = s_species[s_num_molecules - 1];
                    }
                    s_num_molecules--;
                    if (del_species == 0) s_num1--;
                    else s_num2--;
                }
            } else if (s_num_molecules > 0) {
                // Displace
                int idx = rng.randint(0, s_num_molecules - 1);
                float3 displ = make_f3(
                    rng.uniform_range(-cfg.maxdispl, cfg.maxdispl),
                    rng.uniform_range(-cfg.maxdispl, cfg.maxdispl),
                    rng.uniform_range(-cfg.maxdispl, cfg.maxdispl)
                );
                float3 new_sites[3];
                for (int s = 0; s < num_sites_per_mol; ++s) {
                    new_sites[s] = wrap_pbc_dev(s_pos[idx][s] + displ, cfg.box_x, cfg.box_y, cfg.box_z);
                }

                float e_old = 0.0f, e_new = 0.0f;
                int sp = s_species[idx];
                for (int i = 0; i < s_num_molecules; ++i) {
                    if (i == idx) continue;
                    for (int s1 = 0; s1 < num_sites_per_mol; ++s1) {
                        int t1 = (cfg.mol_type == 2) ? sp : s1;
                        for (int s2 = 0; s2 < num_sites_per_mol; ++s2) {
                            int t2 = (cfg.mol_type == 2) ? s_species[i] : s2;
                            float3 d_old = min_image_dev(s_pos[i][s2] - s_pos[idx][s1], cfg.box_x, cfg.box_y, cfg.box_z);
                            float3 d_new = min_image_dev(s_pos[i][s2] - new_sites[s1], cfg.box_x, cfg.box_y, cfg.box_z);
                            e_old += calc_pair_energy_dev(cfg.pair_potentials[t1][t2], sqrtf(dot(d_old, d_old)));
                            e_new += calc_pair_energy_dev(cfg.pair_potentials[t1][t2], sqrtf(dot(d_new, d_new)));
                        }
                    }
                }
                for (int s = 0; s < num_sites_per_mol; ++s) {
                    int et = (cfg.mol_type == 2) ? sp : s;
                    e_old += calc_ext_energy_dev(cfg.ext_potentials[et], s_pos[idx][s]);
                    e_new += calc_ext_energy_dev(cfg.ext_potentials[et], new_sites[s]);
                }

                float delta_E = e_new - e_old;
                if (delta_E < 0.0f || rng.uniform() < expf(-cfg.beta * delta_E)) {
                    for (int s = 0; s < num_sites_per_mol; ++s) {
                        s_pos[idx][s] = new_sites[s];
                    }
                }
            }

            if (step >= equilibration_steps && (step % 50 == 0)) {
                s_accum_N += s_num_molecules;
                s_accum_samples++;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_outputs[box_id].final_num_molecules = s_num_molecules;
        d_outputs[box_id].final_num1 = s_num1;
        d_outputs[box_id].final_num2 = s_num2;
        d_outputs[box_id].avg_num_molecules = (s_accum_samples > 0) ? (float)(s_accum_N / s_accum_samples) : (float)s_num_molecules;
        d_outputs[box_id].final_energy = 0.0f;
    }
}

extern "C" bool cuda_is_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

extern "C" int cuda_get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

extern "C" void run_cuda_batch_gcmc(
    int num_boxes,
    int num_steps,
    int equilibration_steps,
    const CUDABoxConfig* h_configs,
    CUDABoxOutput* h_outputs,
    uint64_t seed
) {
    if (num_boxes <= 0) return;

    CUDABoxConfig* d_configs = nullptr;
    CUDABoxOutput* d_outputs = nullptr;

    cudaMalloc(&d_configs, sizeof(CUDABoxConfig) * num_boxes);
    cudaMalloc(&d_outputs, sizeof(CUDABoxOutput) * num_boxes);

    cudaMemcpy(d_configs, h_configs, sizeof(CUDABoxConfig) * num_boxes, cudaMemcpyHostToDevice);

    int threads_per_block = 32;
    int blocks = num_boxes;

    batch_gcmc_kernel<<<blocks, threads_per_block>>>(
        num_steps,
        equilibration_steps,
        d_configs,
        d_outputs,
        seed
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_outputs, d_outputs, sizeof(CUDABoxOutput) * num_boxes, cudaMemcpyDeviceToHost);

    cudaFree(d_configs);
    cudaFree(d_outputs);
}

} // namespace gcmc_v2
