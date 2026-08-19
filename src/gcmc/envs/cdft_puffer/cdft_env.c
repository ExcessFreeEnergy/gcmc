#include "cdft_env.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define PI_F 3.141592653589793f
#define KB_DEFAULT 1.0f

static float langevin(float u) {
    if (fabsf(u) < 1e-4f) {
        return u / 3.0f - (u * u * u) / 45.0f;
    }
    if (fabsf(u) > 20.0f) {
        return (u > 0.0f) ? (1.0f - 1.0f / u) : (-1.0f - 1.0f / u);
    }
    return 1.0f / tanhf(u) - 1.0f / u;
}

// Inverts Carnahan-Starling bulk chemical potential to obtain equilibrium bulk density rho_b
static float solve_bulk_density_cs(float beta_mu, float T) {
    float sigma = 1.0f;
    float v0 = PI_F * sigma * sigma * sigma / 6.0f;
    
    // Bisection search for eta in [1e-5, 0.65]
    float eta_low = 1e-5f;
    float eta_high = 0.65f;
    float eta = 0.1f;

    for (int iter = 0; iter < 30; ++iter) {
        eta = 0.5f * (eta_low + eta_high);
        float one_minus = 1.0f - eta;
        float mu_ex_cs = (8.0f * eta - 9.0f * eta * eta + 3.0f * eta * eta * eta) / (one_minus * one_minus * one_minus);
        float rho_trial = eta / v0;
        float mu_trial = logf(fmaxf(rho_trial, 1e-6f)) + mu_ex_cs - 2.0f * eta; // CS + mean-field attraction

        if (mu_trial < beta_mu) {
            eta_low = eta;
        } else {
            eta_high = eta;
        }
    }
    return fmaxf(eta / v0, 1e-4f);
}

static void pack_observations(CdftEnv* env) {
    int idx = 0;
    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        env->observations[idx++] = env->rho[i];
    }
    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        env->observations[idx++] = env->charge_n[i];
    }
    env->observations[idx++] = env->T / 600.0f;
    env->observations[idx++] = (env->mu) / 1000.0f;
    env->observations[idx++] = env->phi0 / 50.0f;
    env->observations[idx++] = (env->mode_m - 2.5f) / 1.5f;
    env->observations[idx++] = env->v_bias / 20.0f;
    env->observations[idx++] = env->target_theta;
}

void c_reset(CdftEnv* env) {
    env->tick = 0;
    if (env->max_ticks <= 0) {
        env->max_ticks = 100;
    }

    env->T = rndf(300.0f, 500.0f);
    env->mu = rndf(-5.0f, 2.0f); // Dimensionless chemical potential beta*mu in [-5, 2]
    env->L_slit = 20.0f;
    env->target_theta = rndf(0.2f, 0.8f);

    env->phi0 = rndf(-5.0f, 5.0f);
    env->mode_m = (float)(rand() % 4 + 1);
    env->v_bias = 0.0f;

    float beta = 1.0f / env->T;
    float rho_bulk = solve_bulk_density_cs(env->mu, env->T);
    float dz = env->L_slit / (float)CDFT_GRID_SIZE;

    // Dynamic initial profile from wall Boltzmann factors
    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        float z = (i + 0.5f) * dz;
        float v_wall = 0.0f;
        float r_lo = z;
        float r_hi = env->L_slit - z;
        if (r_lo < 0.858f && r_lo > 0.0f) {
            float r3 = powf(1.0f / r_lo, 3.0f);
            v_wall += ((2.0f / 15.0f) * r3 * r3 * r3 - r3);
        }
        if (r_hi < 0.858f && r_hi > 0.0f) {
            float r3 = powf(1.0f / r_hi, 3.0f);
            v_wall += ((2.0f / 15.0f) * r3 * r3 * r3 - r3);
        }
        env->rho[i] = clampf(rho_bulk * expf(-fminf(v_wall * beta, 10.0f)), 0.001f, 0.999f);
        env->charge_n[i] = 0.0f;
    }

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    pack_observations(env);
}

void c_step(CdftEnv* env) {
    env->tick++;

    // Continuous control deltas [d_phi0, d_m, d_vbias]
    float d_phi0 = env->actions[0] * 3.0f;
    float d_m = env->actions[1] * 0.5f;
    float d_vbias = env->actions[2] * 1.5f;

    env->phi0 = clampf(env->phi0 + d_phi0, -50.0f, 50.0f);
    env->mode_m = clampf(env->mode_m + d_m, 1.0f, 4.0f);
    env->v_bias = clampf(env->v_bias + d_vbias, -20.0f, 20.0f);

    float dz = env->L_slit / (float)CDFT_GRID_SIZE;
    float alpha = 0.25f; // Picard relaxation parameter
    float kappa = 1.0f / 4.5f; // Inverse screening length kappa = 1 / 4.5 A
    float beta = 1.0f / env->T;
    float rho_bulk = solve_bulk_density_cs(env->mu, env->T);
    float sigma_hs = 1.0f;
    float v0 = PI_F * sigma_hs * sigma_hs * sigma_hs / 6.0f;
    float eta_bulk = rho_bulk * v0;
    float one_minus_b = 1.0f - eta_bulk;
    float c1_bulk = -logf(fmaxf(one_minus_b, 1e-4f)) + (3.0f * eta_bulk) / one_minus_b + (1.5f * eta_bulk * eta_bulk) / (one_minus_b * one_minus_b);

    // 1. Compute Fourier mode projection of charge profile n(z) for mode_m
    float km = 2.0f * PI_F * env->mode_m / env->L_slit;
    float n_km_re = 0.0f;
    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        float z = (i + 0.5f) * dz;
        n_km_re += env->charge_n[i] * cosf(km * z) * dz;
    }

    // 2. LMFT restructuring kernel: v1(k) = (4*pi / k^2) * exp(-k^2 / (4*kappa^2))
    float km_sq = km * km;
    float v1_km = (4.0f * PI_F / (km_sq + 1e-6f)) * expf(-km_sq / (4.0f * kappa * kappa));
    float phi_restruct_amp = (2.0f / env->L_slit) * n_km_re * v1_km;
    float phi_R_amp = env->phi0 + phi_restruct_amp;

    float power_cost = 0.0f;
    float p_z[CDFT_GRID_SIZE];

    // 3. Exact Euler-Lagrange Picard Relaxation with FMT Hard-Sphere Functional
    float sum_rho = 0.0f;
    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        float z = (i + 0.5f) * dz;
        float arg = km * z;

        float phi_R = phi_R_amp * cosf(arg) + env->v_bias;
        float e_field_R = (km * phi_R_amp) * sinf(arg);

        power_cost += e_field_R * e_field_R * dz;

        // 9-3 Wall confinement at boundaries
        float v_wall = 0.0f;
        float r_lo = z;
        float r_hi = env->L_slit - z;
        if (r_lo < 0.858f && r_lo > 0.0f) {
            float r3 = powf(1.0f / r_lo, 3.0f);
            v_wall += ((2.0f / 15.0f) * r3 * r3 * r3 - r3);
        }
        if (r_hi < 0.858f && r_hi > 0.0f) {
            float r3 = powf(1.0f / r_hi, 3.0f);
            v_wall += ((2.0f / 15.0f) * r3 * r3 * r3 - r3);
        }

        // 1D FMT Local Hard-Sphere Packing
        float eta_loc = clampf(env->rho[i] * v0, 1e-4f, 0.65f);
        float one_minus_loc = 1.0f - eta_loc;
        float c1_fmt = -logf(one_minus_loc) + (3.0f * eta_loc) / one_minus_loc + (1.5f * eta_loc * eta_loc) / (one_minus_loc * one_minus_loc);

        // Dielectrophoretic coupling from Maxwell stress: c1_diel = 0.5 * beta * alpha_pol * |E|^2
        float alpha_pol = 0.008f;
        float c1_diel = 0.5f * beta * alpha_pol * (e_field_R * e_field_R);

        // First-principles Euler-Lagrange target density:
        // rho(z) = rho_bulk * exp[ -beta*V_ext + (c1_fmt - c1_bulk) + c1_diel - beta*q*phi_R ]
        float exponent = -beta * v_wall + (c1_fmt - c1_bulk) + c1_diel;
        exponent = clampf(exponent, -15.0f, 15.0f);
        float rho_target = rho_bulk * expf(exponent);

        env->rho[i] = clampf((1.0f - alpha) * env->rho[i] + alpha * rho_target, 0.001f, 0.999f);

        // 4. Langevin Molecular Polarization: P_z(z) = rho(z) * mu0 * L(beta * mu0 * E_R)
        float mu0 = 0.382f;
        float u_diel = beta * mu0 * e_field_R;
        p_z[i] = env->rho[i] * mu0 * langevin(u_diel);

        sum_rho += env->rho[i];
    }

    // 5. Self-Consistent Polarization Charge Density: n(z) = -dP_z/dz
    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        int im1 = (i > 0) ? (i - 1) : (CDFT_GRID_SIZE - 1);
        int ip1 = (i < CDFT_GRID_SIZE - 1) ? (i + 1) : 0;
        float dp_dz = (p_z[ip1] - p_z[im1]) / (2.0f * dz);
        env->charge_n[i] = clampf(-dp_dz, -1.0f, 1.0f);
    }

    power_cost += 0.5f * (env->v_bias * env->v_bias);

    float avg_theta = sum_rho / (float)CDFT_GRID_SIZE;
    float tracking_error = fabsf(avg_theta - env->target_theta);

    // Smooth Reward Function: Quadratic tracking penalty + power penalty + precision bonus
    float reward = -10.0f * (tracking_error * tracking_error) - 0.0004f * power_cost;

    if (tracking_error < 0.025f) {
        reward += 1.0f;
        env->log.score += 1.0f;
    }

    env->rewards[0] = reward;

    if (env->tick >= env->max_ticks) {
        env->terminals[0] = 1;
        env->log.n += 1.0f;
    } else {
        env->terminals[0] = 0;
    }

    pack_observations(env);
}

void c_render(CdftEnv* env) {
    printf("[cDFT Env] Tick %3d/%3d | Avg Density: %.3f (Target: %.3f) | phi0: %.2f | Reward: %.4f\n",
           env->tick, env->max_ticks,
           (float)env->observations[CDFT_OBS_SIZE - 1],
           env->target_theta,
           env->phi0,
           env->rewards[0]);
}

void c_close(CdftEnv* env) {
    (void)env;
}
