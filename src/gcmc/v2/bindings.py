"""Python ctypes binding for the v2 C++/CUDA GCMC simulation engine."""

import ctypes
import os
from ctypes import POINTER, Structure, c_bool, c_char_p, c_double, c_float, c_int, c_uint64, c_void_p

import numpy as np

# Path to shared library
_DIR = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_DIR, "libgcmc_v2.so")


def _load_lib():
    if not os.path.exists(_SO_PATH):
        # Attempt to build if missing
        import subprocess

        try:
            cmd = "nvcc -O3 -shared -Xcompiler -fPIC simulation_engine.cpp c_api.cpp cuda_gcmc_kernels.cu -lz -o libgcmc_v2.so"
            subprocess.run(cmd, shell=True, cwd=_DIR, check=True)
        except Exception:
            cmd = "g++ -O3 -shared -fPIC simulation_engine.cpp c_api.cpp -lz -o libgcmc_v2.so"
            subprocess.run(cmd, shell=True, cwd=_DIR, check=True)

    return ctypes.CDLL(_SO_PATH)


_lib = _load_lib()

# Setup C function signatures
_lib.gcmc_v2_create.restype = c_void_p
_lib.gcmc_v2_destroy.argtypes = [c_void_p]

_lib.gcmc_v2_set_thermo.argtypes = [c_void_p, c_double, c_double, c_double]
_lib.gcmc_v2_set_box.argtypes = [c_void_p, c_double, c_double, c_double, c_double]
_lib.gcmc_v2_set_steps.argtypes = [c_void_p, c_int, c_int, c_int, c_bool]
_lib.gcmc_v2_set_molecule_type.argtypes = [c_void_p, c_int, c_double, c_double]
_lib.gcmc_v2_set_weights.argtypes = [c_void_p, c_double, c_double, c_double, c_double, c_double, c_double]
_lib.gcmc_v2_set_paths.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p]

_lib.gcmc_v2_set_pair_potential.argtypes = [
    c_void_p,
    c_int,
    c_int,
    c_int,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
]

_lib.gcmc_v2_set_external_potential_cos.argtypes = [
    c_void_p,
    c_int,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
]

_lib.gcmc_v2_set_external_potential_slit.argtypes = [c_void_p, c_int, c_double, c_double]
_lib.gcmc_v2_set_external_potential_none.argtypes = [c_void_p, c_int]

_lib.gcmc_v2_add_linear_segment.argtypes = [c_void_p, c_int, c_double, c_double, c_double, c_double, c_bool]

_lib.gcmc_v2_add_molecule_3site.argtypes = [
    c_void_p,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
]

_lib.gcmc_v2_add_molecule_1site.argtypes = [c_void_p, c_int, c_double, c_double, c_double]

_lib.gcmc_v2_set_two_type_params.argtypes = [c_void_p, c_char_p, c_char_p, c_double, c_double, c_int, c_int]

_lib.gcmc_v2_set_ewald.argtypes = [
    c_void_p,
    c_int,
    c_double,
    c_int,
    c_double,
    c_double,
    c_double,
    c_double,
]

_lib.gcmc_v2_set_seed.argtypes = [c_void_p, c_uint64]
_lib.gcmc_v2_total_energy.argtypes = [c_void_p]
_lib.gcmc_v2_total_energy.restype = c_double

_lib.gcmc_v2_get_number.argtypes = [c_void_p]
_lib.gcmc_v2_get_number.restype = c_int

_lib.gcmc_v2_get_number1.argtypes = [c_void_p]
_lib.gcmc_v2_get_number1.restype = c_int

_lib.gcmc_v2_get_number2.argtypes = [c_void_p]
_lib.gcmc_v2_run.argtypes = [c_void_p]
_lib.gcmc_v2_run_no_energy.argtypes = [c_void_p]
_lib.gcmc_v2_step.argtypes = [c_void_p]
_lib.gcmc_v2_get_site_pos.argtypes = [c_void_p, c_int, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
_lib.gcmc_v2_get_molecule_species.argtypes = [c_void_p, c_int]
_lib.gcmc_v2_get_molecule_species.restype = c_int


# CUDA batch definitions
class CUDAEwaldKVector(Structure):
    _fields_ = [
        ("kx", c_float),
        ("ky", c_float),
        ("kz", c_float),
        ("weight", c_float),
    ]


class CUDAPairParams(Structure):
    _fields_ = [
        ("kind", c_int),
        ("epsilon_lj", c_float),
        ("sigma_lj", c_float),
        ("rc", c_float),
        ("epsilon_c", c_float),
        ("q1", c_float),
        ("q2", c_float),
        ("kappa_inv", c_float),
        ("diameter", c_float),
        ("prefactor", c_float),
        ("shift_lj", c_float),
    ]


class CUDAExternalParams(Structure):
    _fields_ = [
        ("kind", c_int),
        ("low", c_float),
        ("high", c_float),
        ("width", c_float),
        ("L", c_float),
        ("epsilon", c_float),
        ("sigma", c_float),
        ("cutoff", c_float),
        ("shift", c_float),
        ("q", c_float),
        ("A1", c_float),
        ("A2", c_float),
        ("A3", c_float),
        ("A4", c_float),
        ("phi1", c_float),
        ("phi2", c_float),
        ("phi3", c_float),
        ("phi4", c_float),
        ("q_A1", c_float),
        ("q_A2", c_float),
        ("q_A3", c_float),
        ("q_A4", c_float),
        ("q_phi1", c_float),
        ("q_phi2", c_float),
        ("q_phi3", c_float),
        ("q_phi4", c_float),
    ]


class CUDABoxConfig(Structure):
    _fields_ = [
        ("mol_type", c_int),
        ("box_x", c_float),
        ("box_y", c_float),
        ("box_z", c_float),
        ("beta", c_float),
        ("mu1", c_float),
        ("mu2", c_float),
        ("bond_length", c_float),
        ("maxdispl", c_float),
        ("prob_insert", c_float),
        ("prob_delete", c_float),
        ("prob_displace", c_float),
        ("prob_rotate", c_float),
        ("prob_mutate", c_float),
        ("global_rc", c_float),
        ("electrostatics_mode", c_int),
        ("ewald_alpha", c_float),
        ("ewald_self_per_q2", c_float),
        ("num_k_vectors", c_int),
        ("k_vectors", CUDAEwaldKVector * 128),
        ("site_charges", c_float * 3),
        ("pair_potentials", (CUDAPairParams * 3) * 3),
        ("ext_potentials", CUDAExternalParams * 3),
    ]


class CUDABoxOutput(Structure):
    _fields_ = [
        ("final_num_molecules", c_int),
        ("final_num1", c_int),
        ("final_num2", c_int),
        ("avg_num_molecules", c_float),
        ("avg_energy", c_float),
        ("final_energy", c_float),
    ]


try:
    _lib.cuda_is_available.restype = c_bool
    _lib.cuda_get_device_count.restype = c_int
    _lib.run_cuda_batch_gcmc.argtypes = [c_int, c_int, c_int, POINTER(CUDABoxConfig), POINTER(CUDABoxOutput), c_uint64]
    HAS_CUDA = _lib.cuda_is_available()
except AttributeError:
    HAS_CUDA = False


class GCMCSimulationV2:
    """
    High-performance v2 GCMC simulation instance (1:1 compatible with v1 interface).
    """

    def __init__(self, config, potentials=None, external_potentials=None, input_folder="."):
        self.config = config
        self.input_folder = input_folder
        self.handle = _lib.gcmc_v2_create()

        self._configure()

    def __del__(self):
        if hasattr(self, "handle") and self.handle:
            _lib.gcmc_v2_destroy(self.handle)
            self.handle = None

    def _configure(self):
        cfg = self.config
        T = float(cfg.get("T", 500.0))
        kB = float(cfg.get("kB", 1.380649e-23))

        # Molecule type
        mol_flag = cfg.get("molecule", "None")
        particle_types = cfg.get("particle_types", {})
        if mol_flag == "ABC":
            mol_type_id = 3
        elif mol_flag == "H2O":
            mol_type_id = 4
        elif len(particle_types) == 2:
            mol_type_id = 2
        else:
            mol_type_id = 1

        # Chemical potential
        if mol_flag == "ABC":
            mu = float(particle_types.get("ABC", {}).get("mu", -8.0))
        elif mol_flag == "H2O":
            mu = float(particle_types.get("H2O", {}).get("mu", -8.0))
        elif mol_type_id == 2:
            keys = list(particle_types.keys())
            mu = float(particle_types.get(keys[0], {}).get("mu", -8.0))
        else:
            mu = -8.0

        _lib.gcmc_v2_set_thermo(self.handle, T, kB, mu)

        # Box dimensions
        lx = float(cfg.get("box_length_x", cfg.get("box_length", 20.0)))
        ly = float(cfg.get("box_length_y", cfg.get("box_length", 20.0)))
        lz = float(cfg.get("box_length_z", cfg.get("box_length", 20.0)))
        rc = float(cfg.get("global_rc", 10.0))
        _lib.gcmc_v2_set_box(self.handle, lx, ly, lz, rc)

        # Steps
        max_steps = int(cfg.get("max_steps", 1000))
        eq_steps = int(cfg.get("equilibration", 200))
        out_interval = int(cfg.get("output_interval", 100))
        print_energy = bool(cfg.get("print_energy", True))
        _lib.gcmc_v2_set_steps(self.handle, max_steps, eq_steps, out_interval, print_energy)

        # Molecule & moves
        bond_length = float(cfg.get("bond_length", 0.5))
        maxdispl = float(cfg.get("maxdispl", 3.0))
        _lib.gcmc_v2_set_molecule_type(self.handle, mol_type_id, bond_length, maxdispl)

        weights = cfg.get("weights", {})
        w_ins = float(weights.get("insert", 1.0))
        w_del = float(weights.get("delete", 1.0))
        w_disp = float(weights.get("displace", 0.2))
        w_rot = float(weights.get("rotate", 0.2))
        w_mut = float(weights.get("mutate", 0.1 if mol_type_id == 2 else 0.0))
        w_swp = float(cfg.get("swap_weights", 0.2 if mol_type_id == 2 else 0.0))
        _lib.gcmc_v2_set_weights(self.handle, w_ins, w_del, w_disp, w_rot, w_mut, w_swp)

        # Paths
        folder_bytes = self.input_folder.encode("utf-8")
        _lib.gcmc_v2_set_paths(self.handle, folder_bytes, b"gcmc.log", b"output.xyz")

        # Configure Pair Potentials
        self._configure_potentials()

        # Configure External Potentials
        self._configure_external_potentials()

        # Electrostatics & Ewald
        mode_str = cfg.get("electrostatics_mode", "short_range")
        enable_lr = cfg.get("enable_long_range", False) or (mode_str == "long_range")
        mode_id = 1 if enable_lr else 0
        alpha = float(cfg.get("ewald_alpha", 0.35))
        kmax = int(cfg.get("ewald_kmax", 4))

        q0, q1, q2 = 0.0, 0.0, 0.0
        if mol_flag == "ABC":
            q1 = float(particle_types.get("B", {}).get("q", 0.382))
            q2 = float(particle_types.get("C", {}).get("q", -0.382))
        elif mol_flag == "H2O":
            q0 = float(particle_types.get("O", {}).get("q", -0.8476))
            q1 = float(particle_types.get("H", {}).get("q", particle_types.get("H1", {}).get("q", 0.4238)))
            q2 = float(particle_types.get("H", {}).get("q", particle_types.get("H2", {}).get("q", 0.4238)))
        elif mol_type_id == 2:
            keys = list(particle_types.keys())
            q0 = float(particle_types.get(keys[0], {}).get("q", 1.0))
            q1 = float(particle_types.get(keys[1], {}).get("q", -1.0))

        eps_c = 1.0
        pref = (1.602176634e-19**2) / (4.0 * np.pi * 8.8541878128e-12 * 1.0e-10 * eps_c) if mol_flag == "H2O" else 1.0
        _lib.gcmc_v2_set_ewald(self.handle, mode_id, alpha, kmax, pref, q0, q1, q2)

        # Two type extra settings
        if mol_type_id == 2:
            pkeys = list(particle_types.keys())
            t1 = pkeys[0].encode("utf-8")
            t2 = pkeys[1].encode("utf-8")
            mu1 = float(particle_types[pkeys[0]].get("mu", -8.0))
            mu2 = float(particle_types[pkeys[1]].get("mu", -8.0))
            nbins = int(cfg.get("nbins_x", 100))
            dens_int = int(cfg.get("density_output_interval", 100))
            _lib.gcmc_v2_set_two_type_params(self.handle, t1, t2, mu1, mu2, nbins, dens_int)

        # Load initial coordinates if available
        init_file = cfg.get("init_config")
        if init_file:
            init_path = os.path.join(self.input_folder, init_file)
            if os.path.exists(init_path):
                self._load_xyz(init_path, mol_type_id)
                # Refresh Ewald structure factor with loaded molecules
                _lib.gcmc_v2_set_ewald(self.handle, mode_id, alpha, kmax, pref, q0, q1, q2)

    def _configure_potentials(self):
        pair_dict = self.config.get("potential_pairs", {})
        site_map_abc = {"A": 0, "B": 1, "C": 2}
        site_map_h2o = {"O": 0, "H": 1, "H1": 1, "H2": 2}
        site_map_rpm = {"H": 0, "O": 1}

        for pair_name, p in pair_dict.items():
            parts = pair_name.split("_")
            ptype = p.get("type", "")

            kind = 0
            if ptype == "LJ":
                kind = 1
            elif ptype == "WCA":
                kind = 2
            elif ptype == "HS":
                kind = 3
            elif ptype == "HS+C":
                kind = 4
            elif ptype == "LJ+C":
                kind = 5

            eps_lj = float(p.get("epsilon_lj", p.get("epsilon", 0.0)))
            sig_lj = float(p.get("sigma_lj", p.get("sigma", 1.0)))
            rc = float(p.get("rc", 10.0))
            eps_c = float(p.get("epsilon_c", p.get("epsilon", 1.0)))
            q1 = float(p.get("q1", 0.0))
            q2 = float(p.get("q2", 0.0))
            kappa_inv = float(p.get("kappa_inv", 4.5))
            diameter = float(p.get("diameter", 2.76))

            s1 = 0
            s2 = 0
            if parts[0] in site_map_abc:
                s1 = site_map_abc[parts[0]]
                s2 = site_map_abc[parts[1]]
            elif parts[0] in site_map_h2o:
                s1 = site_map_h2o[parts[0]]
                s2 = site_map_h2o[parts[1]]
            elif parts[0] in site_map_rpm:
                s1 = site_map_rpm[parts[0]]
                s2 = site_map_rpm[parts[1]]

            _lib.gcmc_v2_set_pair_potential(
                self.handle, s1, s2, kind, eps_lj, sig_lj, rc, eps_c, q1, q2, kappa_inv, diameter
            )
            # If H2O, expand H interactions to H1 (site 1) and H2 (site 2)
            if self.config.get("molecule") == "H2O":
                if s1 == 1 and s2 == 1:  # H-H
                    _lib.gcmc_v2_set_pair_potential(
                        self.handle, 2, 2, kind, eps_lj, sig_lj, rc, eps_c, q1, q2, kappa_inv, diameter
                    )
                    _lib.gcmc_v2_set_pair_potential(
                        self.handle, 1, 2, kind, eps_lj, sig_lj, rc, eps_c, q1, q2, kappa_inv, diameter
                    )
                elif s1 == 1 and s2 == 0:  # H-O
                    _lib.gcmc_v2_set_pair_potential(
                        self.handle, 2, 0, kind, eps_lj, sig_lj, rc, eps_c, q1, q2, kappa_inv, diameter
                    )
                elif s1 == 0 and s2 == 1:  # O-H
                    _lib.gcmc_v2_set_pair_potential(
                        self.handle, 0, 2, kind, eps_lj, sig_lj, rc, eps_c, q1, q2, kappa_inv, diameter
                    )

    def _configure_external_potentials(self):
        particle_types = self.config.get("particle_types", {})
        site_map_abc = {"A": 0, "B": 1, "C": 2}
        site_map_h2o = {"O": 0, "H": 1, "H1": 1, "H2": 2}
        site_map_rpm = {"H": 0, "O": 1}

        for name, p in particle_types.items():
            vext_type = p.get("Vext", "None")
            if name in site_map_abc:
                site = site_map_abc[name]
            elif name in site_map_h2o:
                site = site_map_h2o[name]
            elif name in site_map_rpm:
                site = site_map_rpm[name]
            else:
                continue

            if vext_type.startswith("TrainingPotentialWithChargeCos"):
                _lib.gcmc_v2_set_external_potential_cos(
                    self.handle,
                    site,
                    float(p.get("low", 2.0)),
                    float(p.get("high", 18.0)),
                    float(p.get("L", 20.0)),
                    float(p.get("epsilon", 1.0)),
                    float(p.get("sigma", 1.0)),
                    float(p.get("cutoff", 0.858)),
                    float(p.get("q", 0.0)),
                    float(p.get("A1", 0.0)),
                    float(p.get("A2", 0.0)),
                    float(p.get("A3", 0.0)),
                    float(p.get("A4", 0.0)),
                    float(p.get("phi1", 0.0)),
                    float(p.get("phi2", 0.0)),
                    float(p.get("phi3", 0.0)),
                    float(p.get("phi4", 0.0)),
                    float(p.get("q_A1", 0.0)),
                    float(p.get("q_A2", 0.0)),
                    float(p.get("q_A3", 0.0)),
                    float(p.get("q_A4", 0.0)),
                    float(p.get("q_phi1", 0.0)),
                    float(p.get("q_phi2", 0.0)),
                    float(p.get("q_phi3", 0.0)),
                    float(p.get("q_phi4", 0.0)),
                )
                # Add linear segments if present (bounded by ext_type digit suffix)
                num_segs = int(vext_type[-1]) if vext_type[-1].isdigit() else 0
                for i in range(1, num_segs + 1):
                    va_k, vb_k = f"Va{i}", f"Vb{i}"
                    xa_k, xb_k = f"xa{i}", f"xb{i}"
                    if va_k in p and vb_k in p:
                        _lib.gcmc_v2_add_linear_segment(
                            self.handle, site, float(p[va_k]), float(p[vb_k]), float(p[xa_k]), float(p[xb_k]), False
                        )
                    q_va_k, q_vb_k = f"q_Va{i}", f"q_Vb{i}"
                    q_xa_k, q_xb_k = f"q_xa{i}", f"q_xb{i}"
                    if q_va_k in p and q_vb_k in p:
                        _lib.gcmc_v2_add_linear_segment(
                            self.handle,
                            site,
                            float(p[q_va_k]),
                            float(p[q_vb_k]),
                            float(p[q_xa_k]),
                            float(p[q_xb_k]),
                            True,
                        )
            elif vext_type in ("SlitPotential", "WallPotential"):
                _lib.gcmc_v2_set_external_potential_slit(
                    self.handle, site, float(p.get("low", 2.0)), float(p.get("high", 18.0))
                )
            else:
                _lib.gcmc_v2_set_external_potential_none(self.handle, site)

            # If H2O and site is H, also set for H2
            if self.config.get("molecule") == "H2O" and site == 1:
                _lib.gcmc_v2_set_external_potential_none(self.handle, 2)

    def _load_xyz(self, filename, mol_type_id):
        try:
            data = np.genfromtxt(filename, skip_header=2, dtype="str")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            coords = data[:, 1:].astype(float)

            if mol_type_id in (3, 4):  # ABC or H2O
                molecules = coords.reshape(-1, 3, 3)
                for m in molecules:
                    _lib.gcmc_v2_add_molecule_3site(
                        self.handle, m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], m[2, 0], m[2, 1], m[2, 2]
                    )
            else:
                for i, row in enumerate(coords):
                    sp_id = 0 if data[i, 0] in ("H", "1", "A") else 1
                    _lib.gcmc_v2_add_molecule_1site(self.handle, sp_id, row[0], row[1], row[2])
        except Exception as e:
            print(f"Note: Could not load initial.xyz ({e}), starting empty.")

    @property
    def number(self):
        return _lib.gcmc_v2_get_number(self.handle)

    @property
    def number1(self):
        return _lib.gcmc_v2_get_number1(self.handle)

    @property
    def number2(self):
        return _lib.gcmc_v2_get_number2(self.handle)

    def total_energy(self):
        return _lib.gcmc_v2_total_energy(self.handle)

    def run_simulation(self):
        _lib.gcmc_v2_run(self.handle)

    def run_simulation_no_energy(self):
        _lib.gcmc_v2_run_no_energy(self.handle)

    def step(self):
        _lib.gcmc_v2_step(self.handle)

    def get_site_pos(self, mol_idx, site_idx=0):
        x = c_double(0.0)
        y = c_double(0.0)
        z = c_double(0.0)
        _lib.gcmc_v2_get_site_pos(self.handle, mol_idx, site_idx, ctypes.byref(x), ctypes.byref(y), ctypes.byref(z))
        return (x.value, y.value, z.value)

    def get_molecule_species(self, mol_idx):
        return _lib.gcmc_v2_get_molecule_species(self.handle, mol_idx)


def run_batch_cuda(configs, num_steps=1000, equilibration_steps=200, seed=12345):
    """
    Run hundreds or thousands of GCMC simulation boxes in parallel on the GPU.
    """
    if not HAS_CUDA:
        raise RuntimeError("CUDA is not available on this host.")

    num_boxes = len(configs)
    ConfigsArray = CUDABoxConfig * num_boxes
    OutputsArray = CUDABoxOutput * num_boxes

    c_configs = ConfigsArray()
    c_outputs = OutputsArray()

    for b, cfg in enumerate(configs):
        c = c_configs[b]
        mol_flag = cfg.get("molecule", "None")
        if mol_flag == "ABC":
            c.mol_type = 3
        elif mol_flag == "H2O":
            c.mol_type = 4
        elif len(cfg.get("particle_types", {})) == 2:
            c.mol_type = 2
        else:
            c.mol_type = 1

        c.box_x = float(cfg.get("box_length_x", 20.0))
        c.box_y = float(cfg.get("box_length_y", 20.0))
        c.box_z = float(cfg.get("box_length_z", 20.0))

        T = float(cfg.get("T", 500.0))
        kB = float(cfg.get("kB", 1.380649e-23))
        c.beta = 1.0 / (kB * T)

        ptypes = cfg.get("particle_types", {})
        if mol_flag == "ABC":
            c.mu1 = float(ptypes.get("ABC", {}).get("mu", -8.0)) * kB * T
        else:
            keys = list(ptypes.keys())
            c.mu1 = float(ptypes.get(keys[0], {}).get("mu", -8.0)) * kB * T
            if len(keys) > 1:
                c.mu2 = float(ptypes.get(keys[1], {}).get("mu", -8.0)) * kB * T

        c.bond_length = float(cfg.get("bond_length", 0.5))
        c.maxdispl = float(cfg.get("maxdispl", 3.0))
        c.global_rc = float(cfg.get("global_rc", 10.0))

        weights = cfg.get("weights", {})
        w_ins = float(weights.get("insert", 1.0))
        w_del = float(weights.get("delete", 1.0))
        w_disp = float(weights.get("displace", 0.2))
        w_rot = float(weights.get("rotate", 0.2))
        total_w = w_ins + w_del + w_disp + w_rot
        c.prob_insert = w_ins / total_w
        c.prob_delete = w_del / total_w
        c.prob_displace = w_disp / total_w
        c.prob_rotate = w_rot / total_w

        # Ewald configuration
        mode_str = cfg.get("electrostatics_mode", "short_range")
        enable_lr = cfg.get("enable_long_range", False) or (mode_str == "long_range")
        c.electrostatics_mode = 1 if enable_lr else 0
        alpha = float(cfg.get("ewald_alpha", 0.35))
        kmax = int(cfg.get("ewald_kmax", 4))
        c.ewald_alpha = alpha

        pref = (1.602176634e-19**2) / (4.0 * np.pi * 8.8541878128e-12 * 1.0e-10) if mol_flag == "H2O" else 1.0
        c.ewald_self_per_q2 = pref * alpha / np.sqrt(np.pi)

        if mol_flag == "ABC":
            c.site_charges[0] = 0.0
            c.site_charges[1] = float(ptypes.get("B", {}).get("q", 0.382))
            c.site_charges[2] = float(ptypes.get("C", {}).get("q", -0.382))
        elif mol_flag == "H2O":
            c.site_charges[0] = float(ptypes.get("O", {}).get("q", -0.8476))
            c.site_charges[1] = float(ptypes.get("H", {}).get("q", ptypes.get("H1", {}).get("q", 0.4238)))
            c.site_charges[2] = float(ptypes.get("H", {}).get("q", ptypes.get("H2", {}).get("q", 0.4238)))
        elif len(ptypes) == 2:
            keys = list(ptypes.keys())
            c.site_charges[0] = float(ptypes.get(keys[0], {}).get("q", 1.0))
            c.site_charges[1] = float(ptypes.get(keys[1], {}).get("q", -1.0))
            c.site_charges[2] = 0.0

        if enable_lr:
            vol = c.box_x * c.box_y * c.box_z
            two_pi_lx = 2.0 * np.pi / c.box_x
            two_pi_ly = 2.0 * np.pi / c.box_y
            two_pi_lz = 2.0 * np.pi / c.box_z
            k_idx = 0
            for nx in range(-kmax, kmax + 1):
                for ny in range(-kmax, kmax + 1):
                    for nz in range(0, kmax + 1):
                        if nz == 0 and ny < 0:
                            continue
                        if nz == 0 and ny == 0 and nx <= 0:
                            continue
                        if nx * nx + ny * ny + nz * nz > kmax * kmax:
                            continue
                        if k_idx >= 128:
                            break
                        kx = nx * two_pi_lx
                        ky = ny * two_pi_ly
                        kz = nz * two_pi_lz
                        k_sq = kx * kx + ky * ky + kz * kz
                        if k_sq < 1e-12:
                            continue
                        w = pref * (4.0 * np.pi / (vol * k_sq)) * np.exp(-k_sq / (4.0 * alpha * alpha))
                        c.k_vectors[k_idx].kx = kx
                        c.k_vectors[k_idx].ky = ky
                        c.k_vectors[k_idx].kz = kz
                        c.k_vectors[k_idx].weight = w
                        k_idx += 1
            c.num_k_vectors = k_idx

    _lib.run_cuda_batch_gcmc(num_boxes, num_steps, equilibration_steps, c_configs, c_outputs, seed)

    results = []
    for b in range(num_boxes):
        out = c_outputs[b]
        results.append(
            {
                "box_id": b,
                "final_N": out.final_num_molecules,
                "final_N1": out.final_num1,
                "final_N2": out.final_num2,
                "avg_N": out.avg_num_molecules,
            }
        )
    return results
