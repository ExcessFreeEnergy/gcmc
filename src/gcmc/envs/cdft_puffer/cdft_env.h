#ifndef GCMC_ENVS_CDFT_ENV_H
#define GCMC_ENVS_CDFT_ENV_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#define CDFT_GRID_SIZE 50
#define CDFT_NUM_ACTIONS 3
#define CDFT_OBS_SIZE (CDFT_GRID_SIZE * 2 + 6) // rho(50), charge_n(50), T, mu, phi0, m, target_theta, step

typedef struct {
    float score;
    float n; // Required as the last field by PufferLib
} Log;

typedef struct {
    Log log;                     // Required field
    float* observations;         // Required field (shape: [CDFT_OBS_SIZE])
    float* actions;              // Required field (shape: [CDFT_NUM_ACTIONS])
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field

    int tick;
    int max_ticks;

    // Thermodynamic parameters
    float T;
    float mu;
    float L_slit;
    float target_theta; // Target average filling density

    // State arrays
    float rho[CDFT_GRID_SIZE];
    float charge_n[CDFT_GRID_SIZE];
    float phi0;
    float mode_m;
    float v_bias;

} CdftEnv;

static inline float clampf(float v, float min_val, float max_val) {
    if (v < min_val) return min_val;
    if (v > max_val) return max_val;
    return v;
}

static inline float rndf(float a, float b) {
    return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

void c_reset(CdftEnv* env);
void c_step(CdftEnv* env);
void c_render(CdftEnv* env);
void c_close(CdftEnv* env);

#endif // GCMC_ENVS_CDFT_ENV_H
