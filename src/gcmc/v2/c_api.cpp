#include "c_api.h"
#include "simulation_engine.h"
#include <cstring>

using namespace gcmc_v2;

GCMCHandle gcmc_v2_create() {
    return new GCMCSimulationV2();
}

void gcmc_v2_destroy(GCMCHandle handle) {
    if (handle) {
        delete static_cast<GCMCSimulationV2*>(handle);
    }
}

void gcmc_v2_set_thermo(GCMCHandle handle, double T, double kB, double mu) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    sim->T = T;
    sim->kB = kB;
    sim->mu = mu * kB * T;
}

void gcmc_v2_set_box(GCMCHandle handle, double lx, double ly, double lz, double rc) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    sim->box_x = lx;
    sim->box_y = ly;
    sim->box_z = lz;
    sim->global_rc = rc;
    sim->volume = lx * ly * lz;
}

void gcmc_v2_set_steps(GCMCHandle handle, int max_steps, int eq_steps, int out_interval, bool print_energy) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    sim->max_steps = max_steps;
    sim->equilibration_steps = eq_steps;
    sim->output_interval = out_interval;
    sim->print_energy = print_energy;
}

void gcmc_v2_set_molecule_type(GCMCHandle handle, int mol_type, double bond_length, double maxdispl) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    sim->mol_type = static_cast<MoleculeType>(mol_type);
    sim->bond_length = bond_length;
    sim->maxdispl = maxdispl;
}

void gcmc_v2_set_weights(GCMCHandle handle, double w_ins, double w_del, double w_disp, double w_rot, double w_mut, double w_swp) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    sim->weight_insert = w_ins;
    sim->weight_delete = w_del;
    sim->weight_displace = w_disp;
    sim->weight_rotate = w_rot;
    sim->weight_mutate = w_mut;
    sim->weight_swap = w_swp;
}

void gcmc_v2_set_paths(GCMCHandle handle, const char* folder, const char* logfile, const char* output_xyz) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    sim->input_folder = folder ? folder : ".";
    sim->logfile_path = logfile ? logfile : "gcmc.log";
    sim->output_xyz_path = output_xyz ? output_xyz : "output.xyz";
}

void gcmc_v2_set_pair_potential(
    GCMCHandle handle,
    int site1, int site2,
    int kind,
    double eps_lj, double sig_lj, double rc,
    double eps_c, double q1, double q2, double kappa_inv, double diameter
) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    if (site1 < 0 || site1 >= 3 || site2 < 0 || site2 >= 3) return;

    PairPotentialParams& p = sim->pair_potentials[site1][site2];
    p.kind = static_cast<PotentialKind>(kind);
    // Convert kcal/mol to Joules for LJ+C
    if (p.kind == PotentialKind::LJ_C) {
        p.epsilon_lj = eps_lj * 4184.0 / AVOGADRO;
    } else {
        p.epsilon_lj = eps_lj;
    }
    p.sigma_lj = sig_lj;
    p.rc = rc;
    p.epsilon_c = eps_c;
    p.q1 = q1;
    p.q2 = q2;
    p.kappa_inv = kappa_inv;
    p.diameter = diameter;
    p.init();

    // Symmetrize
    sim->pair_potentials[site2][site1] = p;
}

void gcmc_v2_set_external_potential_cos(
    GCMCHandle handle,
    int site,
    double low, double high, double L,
    double eps, double sig, double cutoff, double q,
    double A1, double A2, double A3, double A4,
    double phi1, double phi2, double phi3, double phi4,
    double q_A1, double q_A2, double q_A3, double q_A4,
    double q_phi1, double q_phi2, double q_phi3, double q_phi4
) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    if (site < 0 || site >= 3) return;

    ExternalPotentialParams& ep = sim->ext_potentials[site];
    ep.kind = ExternalPotentialKind::TRAINING_POTENTIAL_WITH_CHARGE_COS;
    ep.low = low;
    ep.high = high;
    ep.L = L;
    double kb_t = sim->kB * sim->T;
    ep.epsilon = eps * kb_t;
    ep.sigma = sig;
    ep.cutoff = cutoff;
    ep.q = q;
    ep.A1 = A1 * kb_t; ep.A2 = A2 * kb_t; ep.A3 = A3 * kb_t; ep.A4 = A4 * kb_t;
    ep.phi1 = phi1; ep.phi2 = phi2; ep.phi3 = phi3; ep.phi4 = phi4;
    ep.q_A1 = q_A1 * kb_t; ep.q_A2 = q_A2 * kb_t; ep.q_A3 = q_A3 * kb_t; ep.q_A4 = q_A4 * kb_t;
    ep.q_phi1 = q_phi1; ep.q_phi2 = q_phi2; ep.q_phi3 = q_phi3; ep.q_phi4 = q_phi4;
    ep.init();
}

void gcmc_v2_set_external_potential_slit(GCMCHandle handle, int site, double low, double high) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    if (site < 0 || site >= 3) return;
    ExternalPotentialParams& ep = sim->ext_potentials[site];
    ep.kind = ExternalPotentialKind::SLIT;
    ep.low = low;
    ep.high = high;
}

void gcmc_v2_set_external_potential_none(GCMCHandle handle, int site) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    if (site < 0 || site >= 3) return;
    sim->ext_potentials[site].kind = ExternalPotentialKind::NONE;
}

void gcmc_v2_add_linear_segment(
    GCMCHandle handle,
    int site,
    double Va, double Vb, double xa, double xb,
    bool is_charge
) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    if (site < 0 || site >= 3) return;
    double kb_t = sim->kB * sim->T;
    LinearPotentialSegment seg{Va * kb_t, Vb * kb_t, xa, xb};
    if (is_charge) {
        sim->ext_potentials[site].q_linear_potentials.push_back(seg);
    } else {
        sim->ext_potentials[site].linear_potentials.push_back(seg);
    }
}

void gcmc_v2_add_molecule_3site(
    GCMCHandle handle,
    double x0, double y0, double z0,
    double x1, double y1, double z1,
    double x2, double y2, double z2
) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    Molecule m;
    m.num_sites = 3;
    m.sites[0] = Vec3(x0, y0, z0);
    m.sites[1] = Vec3(x1, y1, z1);
    m.sites[2] = Vec3(x2, y2, z2);
    sim->molecules.push_back(m);
    sim->number = sim->molecules.size();
}

void gcmc_v2_add_molecule_1site(
    GCMCHandle handle,
    int species_id,
    double x0, double y0, double z0
) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    Molecule m;
    m.num_sites = 1;
    m.species_id = species_id;
    m.sites[0] = Vec3(x0, y0, z0);
    sim->molecules.push_back(m);
    sim->number = sim->molecules.size();
    if (species_id == 0) sim->number1++;
    else sim->number2++;
}

void gcmc_v2_set_two_type_params(
    GCMCHandle handle,
    const char* type1_name, const char* type2_name,
    double mu1, double mu2,
    int nbins_x, int density_interval
) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    sim->type1_name = type1_name ? type1_name : "H";
    sim->type2_name = type2_name ? type2_name : "O";
    sim->mu1 = mu1 * sim->kB * sim->T;
    sim->mu2 = mu2 * sim->kB * sim->T;
    sim->nbins_x = nbins_x;
    sim->density_output_interval = density_interval;
}

void gcmc_v2_set_ewald(
    GCMCHandle handle,
    int mode,
    double alpha,
    int kmax,
    double pref,
    double q0, double q1, double q2
) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    sim->electrostatics_mode = static_cast<ElectrostaticsMode>(mode);
    sim->ewald_params.mode = sim->electrostatics_mode;
    sim->ewald_params.alpha = alpha;
    sim->ewald_params.kmax = kmax;
    sim->ewald_params.prefactor = pref;
    sim->site_charges[0] = q0;
    sim->site_charges[1] = q1;
    sim->site_charges[2] = q2;
}

void gcmc_v2_set_seed(GCMCHandle handle, uint64_t seed) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    sim->rng.set_seed(seed);
}

double gcmc_v2_total_energy(GCMCHandle handle) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    return sim->total_energy();
}

int gcmc_v2_get_number(GCMCHandle handle) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    return sim->number;
}

int gcmc_v2_get_number1(GCMCHandle handle) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    return sim->number1;
}

int gcmc_v2_get_number2(GCMCHandle handle) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    return sim->number2;
}

void gcmc_v2_run(GCMCHandle handle) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    sim->run_simulation();
}

void gcmc_v2_run_no_energy(GCMCHandle handle) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    sim->run_simulation_no_energy();
}

void gcmc_v2_step(GCMCHandle handle) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    sim->step();
}

void gcmc_v2_get_site_pos(GCMCHandle handle, int mol_idx, int site_idx, double* x, double* y, double* z) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    if (mol_idx < 0 || mol_idx >= static_cast<int>(sim->molecules.size())) {
        if (x) *x = 0.0;
        if (y) *y = 0.0;
        if (z) *z = 0.0;
        return;
    }
    const auto& mol = sim->molecules[mol_idx];
    if (site_idx < 0 || site_idx >= mol.num_sites) {
        if (x) *x = 0.0;
        if (y) *y = 0.0;
        if (z) *z = 0.0;
        return;
    }
    if (x) *x = mol.sites[site_idx].x;
    if (y) *y = mol.sites[site_idx].y;
    if (z) *z = mol.sites[site_idx].z;
}

int gcmc_v2_get_molecule_species(GCMCHandle handle, int mol_idx) {
    auto sim = static_cast<GCMCSimulationV2*>(handle);
    if (mol_idx < 0 || mol_idx >= static_cast<int>(sim->molecules.size())) return 0;
    return sim->molecules[mol_idx].species_id;
}

