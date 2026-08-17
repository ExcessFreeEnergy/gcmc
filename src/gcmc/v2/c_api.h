#ifndef GCMC_V2_C_API_H
#define GCMC_V2_C_API_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque simulation handle
typedef void* GCMCHandle;

GCMCHandle gcmc_v2_create();
void gcmc_v2_destroy(GCMCHandle handle);

void gcmc_v2_set_thermo(GCMCHandle handle, double T, double kB, double mu);
void gcmc_v2_set_box(GCMCHandle handle, double lx, double ly, double lz, double rc);
void gcmc_v2_set_steps(GCMCHandle handle, int max_steps, int eq_steps, int out_interval, bool print_energy);
void gcmc_v2_set_molecule_type(GCMCHandle handle, int mol_type, double bond_length, double maxdispl);
void gcmc_v2_set_weights(GCMCHandle handle, double w_ins, double w_del, double w_disp, double w_rot, double w_mut, double w_swp);
void gcmc_v2_set_paths(GCMCHandle handle, const char* folder, const char* logfile, const char* output_xyz);

void gcmc_v2_set_pair_potential(
    GCMCHandle handle,
    int site1, int site2,
    int kind,
    double eps_lj, double sig_lj, double rc,
    double eps_c, double q1, double q2, double kappa_inv, double diameter
);

void gcmc_v2_set_external_potential_cos(
    GCMCHandle handle,
    int site,
    double low, double high, double L,
    double eps, double sig, double cutoff, double q,
    double A1, double A2, double A3, double A4,
    double phi1, double phi2, double phi3, double phi4,
    double q_A1, double q_A2, double q_A3, double q_A4,
    double q_phi1, double q_phi2, double q_phi3, double q_phi4
);

void gcmc_v2_set_external_potential_slit(
    GCMCHandle handle,
    int site,
    double low, double high
);

void gcmc_v2_set_external_potential_none(
    GCMCHandle handle,
    int site
);

void gcmc_v2_add_linear_segment(
    GCMCHandle handle,
    int site,
    double Va, double Vb, double xa, double xb,
    bool is_charge
);

void gcmc_v2_add_molecule_3site(
    GCMCHandle handle,
    double x0, double y0, double z0,
    double x1, double y1, double z1,
    double x2, double y2, double z2
);

void gcmc_v2_add_molecule_1site(
    GCMCHandle handle,
    int species_id,
    double x0, double y0, double z0
);

void gcmc_v2_set_two_type_params(
    GCMCHandle handle,
    const char* type1_name, const char* type2_name,
    double mu1, double mu2,
    int nbins_x, int density_interval
);

void gcmc_v2_set_ewald(
    GCMCHandle handle,
    int mode,
    double alpha,
    int kmax,
    double pref,
    double q0, double q1, double q2
);

void gcmc_v2_set_seed(GCMCHandle handle, uint64_t seed);

double gcmc_v2_total_energy(GCMCHandle handle);
int gcmc_v2_get_number(GCMCHandle handle);
int gcmc_v2_get_number1(GCMCHandle handle);
int gcmc_v2_get_number2(GCMCHandle handle);

void gcmc_v2_run(GCMCHandle handle);
void gcmc_v2_run_no_energy(GCMCHandle handle);
void gcmc_v2_step(GCMCHandle handle);
void gcmc_v2_get_site_pos(GCMCHandle handle, int mol_idx, int site_idx, double* x, double* y, double* z);
int gcmc_v2_get_molecule_species(GCMCHandle handle, int mol_idx);

#ifdef __cplusplus
}
#endif

#endif // GCMC_V2_C_API_H
