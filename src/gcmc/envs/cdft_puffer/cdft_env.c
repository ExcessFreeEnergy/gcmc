#include "cdft_env.h"

#define PI_F 3.141592653589793f

static void pack_observations(CdftEnv* env) {
    int idx = 0;
    // Discretized density profile rho(z)
    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        env->observations[idx++] = env->rho[i];
    }
    // Discretized charge profile n(z)
    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        env->observations[idx++] = env->charge_n[i];
    }
    // Thermodynamic and control state
    env->observations[idx++] = env->T / 1000.0f;
    env->observations[idx++] = env->mu / 5000.0f;
    env->observations[idx++] = env->phi0 / 40.0f;
    env->observations[idx++] = env->mode_m / 4.0f;
    env->observations[idx++] = env->target_theta;
    env->observations[idx++] = (float)env->tick / (float)env->max_ticks;
}

void c_reset(CdftEnv* env) {
    env->tick = 0;
    if (env->max_ticks <= 0) {
        env->max_ticks = 100;
    }

    env->T = rndf(300.0f, 600.0f);
    env->mu = rndf(-4500.0f, -1500.0f);
    env->L_slit = 20.0f;
    env->target_theta = rndf(0.2f, 0.8f);

    env->phi0 = rndf(-10.0f, 10.0f);
    env->mode_m = (float)(rand() % 4 + 1);
    env->v_bias = 0.0f;

    // Initial density profile with baseline vapor-liquid state
    float initial_rho = rndf(0.1f, 0.5f);
    for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
        env->rho[i] = initial_rho + rndf(-0.02f, 0.02f);
        env->charge_n[i] = 0.0f;
    }

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    pack_observations(env);
}

void c_step(CdftEnv* env) {
    env->tick++;

    // Actions: continuous control [d_phi0, d_m, d_vbias]
    float d_phi0 = env->actions[0] * 4.0f;
    float d_m = env->actions[1] * 0.5f;
    float d_vbias = env->actions[2] * 2.0f;

    env->phi0 = clampf(env->phi0 + d_phi0, -38.2f, 38.2f);
    env->mode_m = clampf(env->mode_m + d_m, 1.0f, 4.0f);
    env->v_bias = clampf(env->v_bias + d_vbias, -10.0f, 10.0f);

    // Fast Euler-Lagrange Picard cDFT Relaxation Step
    float dz = env->L_slit / (float)CDFT_GRID_SIZE;
    float alpha = 0.25f; // Relaxation parameter
    float sum_rho = 0.0f;
    float power_cost = 0.0f;

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

        // Dielectrocapillary hyperfunctional coupling: force ~ grad(E^2)
        float c1_local = 0.5f * (e_field * e_field) * 0.02f - 2.5f * env->rho[i];
        float beta = 1.0f / (1.380649e-23f * env->T * 1e20f); // Normalized units

        float exp_arg = clampf(beta * (env->mu - v_wall) + c1_local, -10.0f, 10.0f);
        float rho_eq = 0.5f * expf(exp_arg);

        // Update density via Picard relaxation
        env->rho[i] = (1.0f - alpha) * env->rho[i] + alpha * rho_eq;
        env->charge_n[i] = -0.1f * phi_z * env->rho[i];

        sum_rho += env->rho[i];
    }

    float avg_theta = sum_rho / (float)CDFT_GRID_SIZE;
    float tracking_error = fabsf(avg_theta - env->target_theta);

    // Reward Shaping: Tracking accuracy + energy efficiency penalty + achievement bonus
    float reward = -10.0f * (tracking_error * tracking_error) - 0.001f * power_cost;

    if (tracking_error < 0.03f) {
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
    // Console rendering summary
    printf("[cDFT Env] Tick %3d/%3d | Avg Density: %.3f (Target: %.3f) | phi0: %.2f | Reward: %.4f\n",
           env->tick, env->max_ticks,
           (float)env->observations[CDFT_OBS_SIZE - 2],
           env->target_theta,
           env->phi0,
           env->rewards[0]);
}

void c_close(CdftEnv* env) {
    // Cleanup if needed
}
