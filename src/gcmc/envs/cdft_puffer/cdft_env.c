#include "cdft_env.h"

#define PI_F 3.141592653589793f

static void pack_observations(CdftEnv* env) {
    int idx = 0;
    // Discretized density profile rho(z) in [0, 1]
    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        env->observations[idx++] = env->rho[i];
    }
    // Discretized charge profile n(z) in [-1, 1]
    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        env->observations[idx++] = env->charge_n[i];
    }
    // Normalized thermodynamic and control state
    env->observations[idx++] = env->T / 600.0f;
    env->observations[idx++] = (env->mu + 3000.0f) / 1500.0f;
    env->observations[idx++] = env->phi0 / 38.2f;
    env->observations[idx++] = (env->mode_m - 2.5f) / 1.5f;
    env->observations[idx++] = env->v_bias / 10.0f;
    env->observations[idx++] = env->target_theta;
}

void c_reset(CdftEnv* env) {
    env->tick = 0;
    if (env->max_ticks <= 0) {
        env->max_ticks = 100;
    }

    env->T = rndf(300.0f, 500.0f);
    env->mu = rndf(-3500.0f, -2500.0f);
    env->L_slit = 20.0f;
    env->target_theta = rndf(0.2f, 0.8f);

    env->phi0 = rndf(-5.0f, 5.0f);
    env->mode_m = (float)(rand() % 4 + 1);
    env->v_bias = 0.0f;

    // Initial density profile with baseline vapor state
    float initial_rho = rndf(0.05f, 0.2f);
    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        env->rho[i] = clampf(initial_rho + rndf(-0.01f, 0.01f), 0.0f, 1.0f);
        env->charge_n[i] = 0.0f;
    }

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    pack_observations(env);
}

void c_step(CdftEnv* env) {
    env->tick++;

    // Actions: continuous control deltas [d_phi0, d_m, d_vbias]
    float d_phi0 = env->actions[0] * 3.0f;
    float d_m = env->actions[1] * 0.5f;
    float d_vbias = env->actions[2] * 1.5f;

    env->phi0 = clampf(env->phi0 + d_phi0, -38.2f, 38.2f);
    env->mode_m = clampf(env->mode_m + d_m, 1.0f, 4.0f);
    env->v_bias = clampf(env->v_bias + d_vbias, -10.0f, 10.0f);

    // Fast Euler-Lagrange Picard cDFT Relaxation Step
    float dz = env->L_slit / (float)CDFT_GRID_SIZE;
    float alpha = 0.30f; // Picard relaxation parameter
    float sum_rho = 0.0f;
    float power_cost = 0.0f;

    // Thermal reduced chemical potential
    float beta_mu = (env->mu + 3000.0f) / env->T;

    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        float z = (i + 0.5f) * dz;
        float arg = 2.0f * PI_F * env->mode_m * z / env->L_slit;

        // External field: cosine wave potential + bias
        float phi_z = (env->phi0 / env->mode_m) * cosf(arg) + env->v_bias;
        float e_field = (2.0f * PI_F * env->phi0 / env->L_slit) * sinf(arg);

        power_cost += e_field * e_field * dz;

        // 9-3 Wall confinement at boundaries
        float v_wall = 0.0f;
        float r_lo = z;
        float r_hi = env->L_slit - z;
        if (r_lo < 1.0f && r_lo > 0.05f) {
            float r3 = powf(1.0f / r_lo, 3.0f);
            v_wall += ((2.0f / 15.0f) * r3 * r3 * r3 - r3);
        }
        if (r_hi < 1.0f && r_hi > 0.05f) {
            float r3 = powf(1.0f / r_hi, 3.0f);
            v_wall += ((2.0f / 15.0f) * r3 * r3 * r3 - r3);
        }

        // Dielectrocapillary coupling: body force ~ grad(E^2) produces spatial waves
        float c1_diel = 0.020f * (e_field * e_field);
        float mu_eff = beta_mu - v_wall + c1_diel - 2.8f * (env->rho[i] - 0.5f);

        // Fermi-Dirac / logistic density functional response: rho_eq in [0, 1]
        float exp_val = expf(clampf(-mu_eff, -15.0f, 15.0f));
        float rho_eq = 1.0f / (1.0f + exp_val);

        // Update density via Picard relaxation
        env->rho[i] = clampf((1.0f - alpha) * env->rho[i] + alpha * rho_eq, 0.001f, 0.999f);
        env->charge_n[i] = clampf(-0.05f * phi_z * env->rho[i], -1.0f, 1.0f);

        sum_rho += env->rho[i];
    }

    float avg_theta = sum_rho / (float)CDFT_GRID_SIZE;
    float tracking_error = fabsf(avg_theta - env->target_theta);

    // Smooth Reward Function: Quadratic tracking penalty + power penalty + precision bonus
    float reward = -8.0f * (tracking_error * tracking_error) - 0.0001f * power_cost;

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
    // Cleanup
}
