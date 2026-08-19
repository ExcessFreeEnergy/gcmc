# Comprehensive Physics & Statistical Mechanics Audit: `gcmc`

This audit evaluates the physical rigor of the grand canonical Monte Carlo (`gcmc`) engine, classical Density Functional Theory (`cDFT`) solvers, and reinforcement learning environments. It systematically identifies and provides first-principles replacements for **phenomenological forcing functions**, **thermodynamic disconnects**, **static initial guesses & phase biases**, **brittle spatial extractions**, and **fake boundary asymptotic limits**.

---

## 1. File-by-File Audit of `src/gcmc/v2/`

### 1.1 `src/gcmc/v2/core_types.h`

#### Issue 1: Phenomenological Multi-Harmonic & Piecewise Linear Spline Potential
* **Lines of Code**: `core_types.h:306-342`
```cpp
double ratio = 2.0 * PI * x_coord / L;
double sine_terms = A1 * std::sin(ratio * 1.0 + phi1) +
                    A2 * std::sin(ratio * 2.0 + phi2) +
                    A3 * std::sin(ratio * 3.0 + phi3) +
                    A4 * std::sin(ratio * 4.0 + phi4);
...
for (const auto& seg : linear_potentials) {
    if (x_coord >= seg.xa && x_coord <= seg.xb && (seg.xb > seg.xa)) {
        double vlin = seg.Va + (seg.Vb - seg.Va) * (x_coord - seg.xa) / (seg.xb - seg.xa);
        linear_terms += vlin;
    }
}
return sine_terms + linear_terms + (sine_terms_q + linear_terms_q) * q;
```
* **Statistical Mechanics Violation**: Phenomenological forcing function. Handcrafted trigonometric Fourier series ($A_1 \dots A_4, \phi_1 \dots \phi_4$) and arbitrary piecewise linear segments ($V_a, V_b, x_a, x_b$) are empirical shortcuts that bypass the physical electrostatic boundary-value problem. External electrostatic potentials must satisfy Poisson's equation $\nabla \cdot (\varepsilon(\mathbf{r}) \nabla \phi(\mathbf{r})) = -\rho_{\rm charge}(\mathbf{r})$ with physical boundary conditions (Dirichlet $\phi(0) = \Phi_1, \phi(L) = \Phi_2$ or Neumann $\left.\partial_z \phi\right|_{\text{wall}} = -4\pi \sigma_{\rm wall}$).
* **Rigorous Replacement**:
  $$\nabla \cdot \left[ \varepsilon(\mathbf{r}) \nabla \phi(\mathbf{r}) \right] = -4\pi \sum_i q_i \rho_i(\mathbf{r}), \quad \phi(z=0) = \Phi_{\rm left}, \quad \phi(z=L) = \Phi_{\rm right}$$
  The external field energy exerted on a molecule with partial charges $\{q_\alpha\}$ at positions $\{\mathbf{r}_\alpha\}$ is:
  $$V_{\rm ext}(\mathbf{R}, \mathbf{\Omega}) = \sum_{\alpha} q_\alpha \phi(\mathbf{r}_\alpha) + \sum_{\alpha} V_{\rm wall}^{\rm steric}(\mathbf{r}_\alpha)$$

---

#### Issue 2: Hard-Coded Cutoff and Potential Defaults
* **Lines of Code**: `core_types.h:180-186, 260-267`
```cpp
double rc = 10.0;
double kappa_inv = 4.5;
double diameter = 2.76;
double cutoff = 0.858374218933;
```
* **Statistical Mechanics Violation**: Thermodynamic & structural disconnect. Screening lengths ($\kappa^{-1}$) and cutoff radii ($r_c$) are hardcoded to fixed values ($4.5\,\text{Å}$, $2.76\,\text{Å}$, $0.858\,\sigma$). The screening parameter $\kappa$ in Local Molecular Field Theory (LMFT) must scale with the physical Debye-Hückel / Stillinger-Lovett screening length $\kappa_{\rm D} = \sqrt{4\pi \lambda_B \sum_i \rho_{i, \text{bulk}} z_i^2}$, where $\lambda_B = \frac{e^2}{4\pi \varepsilon_0 \varepsilon_r k_B T}$.
* **Rigorous Replacement**: Dynamically compute $\kappa$ from bulk ionic strength $I$ or dielectric permittivity $\varepsilon_r(T)$:
  $$\kappa(T, \rho_{\rm bulk}) = \sqrt{\frac{4\pi e^2}{\varepsilon_0 \varepsilon_r(T) k_B T} \sum_i \rho_{i, \text{bulk}} z_i^2}$$

---

### 1.2 `src/gcmc/v2/simulation_engine.h`

#### Issue 3: Hardcoded Thermodynamic State Variables ($\mu$, $T$, $V$)
* **Lines of Code**: `simulation_engine.h:71-74, 123-124`
```cpp
double T = 500.0;
double kB = KB_DEFAULT;
double beta = 1.0 / (KB_DEFAULT * 500.0);
double mu = -8.0;
...
double mu1 = -8.0;
double mu2 = -8.0;
```
* **Statistical Mechanics Violation**: Thermodynamic disconnect. Hardcoding $\mu = -8.0$ (arbitrary reduced units) disconnects the open grand canonical ensemble $(\mu, V, T)$ from the physical equation of state (EOS) and liquid-vapor coexistence $\mu_{\rm coex}(T)$. In a physical simulation, $\mu$ must be derived dynamically from the desired bulk density $\rho_{\rm bulk}$ and temperature $T$ using the bulk Equation of State:
  $$\mu(\rho_{\rm bulk}, T) = \mu^{\rm id}(\rho_{\rm bulk}, T) + \mu^{\rm ex}(\rho_{\rm bulk}, T)$$
* **Rigorous Replacement**:
  $$\mu^{\rm id}(\rho_{\rm bulk}, T) = k_B T \ln(\rho_{\rm bulk} \Lambda^3)$$
  $$\mu^{\rm ex}(\rho_{\rm bulk}, T) = -k_B T \ln \left\langle \exp\left(-\beta \Delta U_{\rm test}\right) \right\rangle_{\rm bulk} \quad \text{(Widom Insertion)}$$
  or via the Carnahan-Starling / Johnson-Zollweg-Gubbins (JZG) Lennard-Jones EOS:
  $$\mu_{\rm LJ}(\rho, T) = k_B T \left[ \ln\rho + \frac{8\eta - 9\eta^2 + 3\eta^3}{(1-\eta)^3} \right] + \sum_{n=1}^8 \frac{n}{n-1} a_n(T) \rho^{n-1}$$

---

### 1.3 `src/gcmc/v2/simulation_engine.cpp`

#### Issue 4: Anisotropic Rotational Sampling Bias (Non-Haar Measure)
* **Lines of Code**: `simulation_engine.cpp:258-265`
```cpp
double d_theta = rng.uniform_range(-maxrot, maxrot);
double d_phi = rng.uniform_range(-maxrot, maxrot);
double d_psi = rng.uniform_range(-maxrot, maxrot);
Quaternion dq(std::cos(d_theta), std::sin(d_phi), std::sin(d_psi), std::sin(d_theta));
double norm = std::sqrt(dq.w * dq.w + dq.x * dq.x + dq.y * dq.y + dq.z * dq.z);
dq = Quaternion(dq.w / norm, dq.x / norm, dq.y / norm, dq.z / norm);
```
* **Statistical Mechanics Violation**: Static Phase / Orientational Bias. Constructing a random quaternion as $(\cos(d\theta), \sin(d\phi), \sin(d\psi), \sin(d\theta))$ with $w = \cos(d\theta)$ and $z = \sin(d\theta)$ couples the real scalar component $w$ directly to the vector component $z$. This generates an anisotropic probability distribution on $SO(3)$, violating rotational invariance (Haar measure $d\mathbf{\Omega} = \frac{1}{8\pi^2}\sin\theta \, d\theta \, d\phi \, d\psi$).
* **Rigorous Replacement**: Uniform isotropic angular displacement using Marsaglia/Shoemake random axis-angle perturbation:
  Sample a uniform unit vector $\hat{\mathbf{u}} \in S^2$ and angle $\delta\theta \sim \text{Uniform}(-\delta\theta_{\max}, \delta\theta_{\max})$:
  $$u_1 \in [-1, 1], \quad \phi \in [0, 2\pi), \quad \hat{\mathbf{u}} = \left(\sqrt{1-u_1^2}\cos\phi, \sqrt{1-u_1^2}\sin\phi, u_1\right)$$
  $$\delta \mathbf{q} = \left(\cos\frac{\delta\theta}{2}, \, \hat{\mathbf{u}}_x \sin\frac{\delta\theta}{2}, \, \hat{\mathbf{u}}_y \sin\frac{\delta\theta}{2}, \, \hat{\mathbf{u}}_z \sin\frac{\delta\theta}{2}\right)$$

---

#### Issue 5: Artificial Overflow Truncation in Particle Deletion Acceptance
* **Lines of Code**: `simulation_engine.cpp:327-329, 460-462`
```cpp
double log_prob = -beta * (delta_E + mu) + std::log((double)number) - std::log(volume);
double prob = (log_prob < 700.0) ? std::exp(log_prob) : 0.0;
if (rng.uniform() < prob) { ... }
```
* **Statistical Mechanics Violation**: Artificial phase suppression / acceptance bias. Setting `prob = 0.0` when `log_prob >= 700.0` completely rejects transitions that have an overwhelmingly favorable free-energy decrease ($\Delta \ln \mathcal{P} \ge 700$). In Metropolis-Hastings, the acceptance probability is $\alpha = \min(1, \mathcal{P})$. If $\ln \mathcal{P} > 0$, the move must be accepted with probability $1.0$, not rejected (`0.0`). This creates severe unphysical particle accumulation during sudden density drops or expansions.
* **Rigorous Replacement**:
  ```cpp
  if (log_prob >= 0.0 || rng.uniform() < std::exp(log_prob)) {
      // Accept particle deletion with exact detailed balance
  }
  ```

---

### 1.4 `src/gcmc/v2/cuda_gcmc.h` & `src/gcmc/v2/cuda_gcmc_kernels.cu`

#### Issue 6: Low-Entropy 32-Bit PRNG Exhaustion in Parallel GPU Markov Chains
* **Lines of Code**: `cuda_gcmc_kernels.cu:49-67`
```cuda
struct DevRNG {
    uint32_t state;
    __device__ uint32_t next_u32() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }
};
```
* **Statistical Mechanics Violation**: Statistical independence violation & pseudo-random cycle bias. A 32-bit Xorshift PRNG has a period of only $2^{32} - 1 \approx 4.29 \times 10^9$. At $1.1 \times 10^8$ steps/s, a GPU simulation cycles through the entire state space in under 40 seconds. This causes spurious spatial-temporal correlations and violates the ergodic hypothesis $\lim_{t \to \infty} \frac{1}{t}\int_0^t A(x(t'))dt' = \langle A \rangle_{\mu, V, T}$.
* **Rigorous Replacement**: 64-bit counter-based PRNG (Philox4x32-10 or SplitMix64 / Xoroshiro128+) with period $\ge 2^{128}$.

---

#### Issue 7: Fixed Sub-Sampling Interval for Ensemble Averages
* **Lines of Code**: `cuda_gcmc_kernels.cu:493-496`
```cuda
if (step >= equilibration_steps && (step % 50 == 0)) {
    s_accum_N += s_num_molecules;
    s_accum_samples++;
}
```
* **Statistical Mechanics Violation**: Brittle spatial & temporal grid extraction. Hardcoding a stride of `50` steps assumes a fixed autocorrelation time $\tau_{\rm auto}$. Near critical points ($T \approx T_c$) or dense liquid condensation, critical slowing down causes $\tau_{\rm auto} \sim \xi^z \to \infty$. Fixed sampling introduces biased variance estimates.
* **Rigorous Replacement**: Integrated autocorrelation time estimation via Flyvbjerg-Petersen block averaging:
  $$\tau_{\rm int} = \frac{1}{2} + \sum_{t=1}^{t_{\max}} \frac{\langle (N(0) - \langle N \rangle)(N(t) - \langle N \rangle) \rangle}{\sigma_N^2}, \quad N_{\rm eff} = \frac{N_{\rm total}}{2 \tau_{\rm int}}$$

---

### 1.5 `src/gcmc/v2/c_api.cpp` & `src/gcmc/v2/bindings.py`

#### Issue 8: Hardcoded Inconsistent Energy Unit Conversions & Inelastic Fallbacks
* **Lines of Code**: `c_api.cpp:92-96`, `bindings.py:278-287, 333-350`
```python
# Chemical potential fallback in bindings.py
if mol_flag == "ABC":
    mu = float(particle_types.get("ABC", {}).get("mu", -8.0))
else:
    mu = -8.0

# Hardcoded partial charges in bindings.py
q1 = float(particle_types.get("B", {}).get("q", 0.382))
q2 = float(particle_types.get("C", {}).get("q", -0.382))
eps_c = 1.0
pref = (1.602176634e-19**2) / (4.0 * np.pi * 8.8541878128e-12 * 1.0e-10 * eps_c) if mol_flag == "H2O" else 1.0
```
```cpp
// In c_api.cpp
if (p.kind == PotentialKind::LJ_C) {
    p.epsilon_lj = eps_lj * 4184.0 / AVOGADRO;
}
```
* **Statistical Mechanics Violation**: Hidden Unit Inconsistencies & Hardcoded Molecular Polarities. The code intermixes reduced Lennard-Jones units ($\varepsilon, \sigma$), SI Joules ($4184 / N_A$), and Gaussian/CGS electrostatic units depending on branching conditions.
* **Rigorous Replacement**: Enforce an explicit internal Hamiltonian unit system (e.g. Gromacs reduced units $\text{kJ}\cdot\text{mol}^{-1}, \text{nm}, e$ or standard LJ reduced units $\varepsilon_0, \sigma_0, m_0, q_0$) with all unit conversions handled at the I/O interface rather than inside the physics engine.

---

## 2. Audit of Surrounding Physics Modules

### 2.1 `src/gcmc/lmft_baseline/baseline_solver.py`

#### Issue 9: Static Initial Guess Ignoring Interfacial Scaling
* **Lines of Code**: `baseline_solver.py:181`
```python
self.rho = np.full(grid_size, self.rho_bulk)
```
* **Statistical Mechanics Violation**: Static initial guess / phase bias. Initializing an inhomogeneous slit pore with a flat bulk profile $\rho(z) = \rho_{\rm bulk}$ ignores the steric exclusion zone ($V_{\rm ext} \to \infty$) at walls and lacks the temperature-dependent interfacial profile width $\xi(T) \propto |1 - T/T_c|^{-\nu}$. In subcritical fluids ($T < T_c$), this causes Picard iteration instability or trapping in metastable vapor/liquid branches.
* **Rigorous Replacement**: Initial guess generated from exact contact value theorem and mean-field capillary profile:
  $$\rho_0(z) = \rho_{\rm bulk} \exp\left[-\beta V_{\rm ext}(z)\right] \cdot \left[ \frac{1}{2} (1 + \tanh((z - z_0)/\xi(T))) \right]$$
  where $\xi(T) = \xi_0 \left|1 - \frac{T}{T_c}\right|^{-0.63}$ and $\rho(z_{\rm wall}^+) = \beta P_{\rm bulk}$.

---

#### Issue 10: Artificial Inflation of Molecular Polarizability
* **Lines of Code**: `baseline_solver.py:236-237`
```python
alpha_pol = 0.008
c1_diel = 0.5 * self.beta * alpha_pol * (e_field**2)
```
* **Statistical Mechanics Violation**: Phenomenological forcing shortcut. The polarizability $\alpha_{\rm pol} = 0.008$ is hardcoded as an ad-hoc parameter. From first principles, the linear dielectric susceptibility and quadratic dielectrophoretic coupling for a dipolar fluid emerge directly from the high-temperature expansion of the Langevin function:
  $$\mathcal{L}(u) = \frac{u}{3} - \frac{u^3}{45} + \mathcal{O}(u^5), \quad u = \beta \mu_0 |E|$$
  $$\alpha_{\rm pol}(T) = \frac{\beta \mu_0^2}{3} = \frac{\mu_0^2}{3 k_B T}$$
  Hardcoding $\alpha_{\rm pol} = 0.008$ artificially inflates the polarization response by ~82x compared to the true physical dipole ($\mu_0 = 0.382\,\text{D}$ at $T = 500\,\text{K}$).
* **Rigorous Replacement**:
  $$c_{\rm diel}^{(1)}(z) = \ln \left[ \frac{\sinh(\beta \mu_0 |\mathbf{E}_R(z)|)}{\beta \mu_0 |\mathbf{E}_R(z)|} \right]$$
  whose small-field Taylor limit is $\frac{1}{2}\beta \left(\frac{\mu_0^2}{3 k_B T}\right) |\mathbf{E}_R(z)|^2$.

---

### 2.2 `src/gcmc/envs/cdft_puffer/cdft_env.c`

#### Issue 11: Single Fourier Mode Truncation of Restructuring Field
* **Lines of Code**: `cdft_env.c:128-141`
```c
float km = 2.0f * PI_F * env->mode_m / env->L_slit;
float n_km_re = 0.0f;
for (int i = 0; i < CDFT_GRID_SIZE; ++i) {
    float z = (i + 0.5f) * dz;
    n_km_re += env->charge_n[i] * cosf(km * z) * dz;
}
float km_sq = km * km;
float v1_km = (4.0f * PI_F / (km_sq + 1e-6f)) * expf(-km_sq / (4.0f * kappa * kappa));
float phi_restruct_amp = (2.0f / env->L_slit) * n_km_re * v1_km;
float phi_R_amp = env->phi0 + phi_restruct_amp;
```
* **Statistical Mechanics Violation**: Truncation of the Long-Range Coulomb Green's Function. Projecting the polarization charge density $n(z)$ onto only a single harmonic `mode_m` discards the full Fourier spectrum $\{k_n = 2\pi n / L_z\}_{n=1}^{N/2}$. Dielectric boundary charges at the slit walls have high-frequency Fourier components that are completely lost, violating Poisson-LMFT consistency $\phi_R(z) = \phi_{\rm ext}(z) + \int dz' n(z') v_1(|z - z'|)$.
* **Rigorous Replacement**: Exact 1D discrete Fourier transform (FFT) convolution over all reciprocal modes $k_n$:
  $$\tilde{\phi}_R(k_n) = \tilde{\phi}_{\rm ext}(k_n) + \frac{4\pi}{k_n^2} \exp\left(-\frac{k_n^2}{4\kappa^2}\right) \tilde{n}(k_n), \quad \forall k_n \neq 0$$
  $$\phi_R(z) = \mathcal{F}^{-1}\left[\tilde{\phi}_R(k_n)\right](z)$$

---

### 2.3 `src/gcmc/v1/utils/get_density_profile.py` & `src/gcmc/v1/utils/get_profiles.py`

#### Issue 12: Fixed Mid-Domain Index Window for Bulk Density Extraction
* **Lines of Code**: `get_density_profile.py:179-183`
```python
mid = len(density_profile) // 2
bulk_density = float(np.median(density_profile[max(0, mid - 10) : min(len(density_profile), mid + 10)]))
density_std = float(np.std(density_profile[max(0, mid - 10) : min(len(density_profile), mid + 10)]))
```
* **Statistical Mechanics Violation**: Brittle spatial & grid extraction. Sampling the bulk density strictly within `[mid - 10 : mid + 10]` fails if an external field gradient or capillary meniscus shifts the interface to the center of the pore ($z \approx L_z/2$). The bulk density must be extracted from flat plateau regions where $\nabla \rho(z) \to 0$ and $\nabla^2 \rho(z) \to 0$.
* **Rigorous Replacement**: Algorithmic plateau detection identifying flat domains outside the interfacial region $\left\{z \mid |\partial_z \rho(z)| < \epsilon_{\rm tol} \land |\partial_z^2 \rho(z)| < \epsilon_{\rm tol}/\Delta z\right\}$:
  $$\rho_{\rm bulk} = \frac{\int_{\text{plateau}} \rho(z) dz}{\int_{\text{plateau}} dz}$$

---

### 2.4 `src/gcmc/v1/external_potentials.py` & `src/gcmc/v1/constants.py`

#### Issue 13: Floating-Point Fake Boundary Representations (`1.0e30` vs `np.inf`)
* **Lines of Code**: `constants.py:24`, `external_potentials.py:48, 61, 93, 133`
```python
very_large_number = 1.0e30
...
return np.where((position[0] > self.width), very_large_number, 0.0)
```
* **Statistical Mechanics Violation**: Fake boundaries & asymptotic limits. Using $1.0\times 10^{30}$ instead of IEEE `+inf` causes catastrophic floating-point cancellation in energy differences $\Delta E = E_{\rm new} - E_{\rm old}$ when two overlapping configurations are subtracted ($10^{30} - 10^{30} = 0.0$), falsely accepting steric core overlaps.
* **Rigorous Replacement**: Use standard IEEE 754 positive infinity `np.inf` / `std::numeric_limits<double>::infinity()`, which satisfies $\exp(-\beta \cdot \infty) \equiv 0.0$ and $\infty - x \equiv \infty$.

---

## 3. Synthesis: Anti-Pattern Kill List & First-Principles Mapping

| Anti-Pattern Category | Files Affected | Exact Line References | Statistical Mechanics Violation | Rigorous First-Principles Replacement |
|---|---|---|---|---|
| **Phenomenological Forcing** | `v2/core_types.h`<br>`v2/cuda_gcmc_kernels.cu`<br>`lmft_baseline/baseline_solver.py` | `core_types.h:306-342`<br>`cuda_gcmc_kernels.cu:137-164`<br>`baseline_solver.py:236-237` | Hardcoded sinusoidal Fourier amplitudes and inflated polarizability $\alpha_{\rm pol} = 0.008$. | Exact boundary value Poisson-cDFT $\nabla \cdot (\varepsilon \nabla \phi) = -4\pi \rho_q$ with Langevin-Debye $\alpha_{\rm pol}(T) = \frac{\mu_0^2}{3 k_B T}$ and non-linear Langevin free energy $c_{\rm diel}^{(1)} = \ln(\sinh(u)/u)$. |
| **Thermodynamic Disconnects** | `v2/simulation_engine.h`<br>`v2/bindings.py`<br>`v2/c_api.cpp` | `simulation_engine.h:71-74`<br>`bindings.py:278-287`<br>`c_api.cpp:92-96` | Arbitrary hardcoded $\mu = -8.0$ and mixed unit conversions. | Derive $\mu(\rho_{\rm bulk}, T)$ directly from Carnahan-Starling / JZG EOS or Widom test-particle insertion. |
| **Static Guesses & Phase Biases** | `v2/simulation_engine.cpp`<br>`lmft_baseline/baseline_solver.py`<br>`envs/cdft_puffer/cdft_env.c` | `simulation_engine.cpp:258-265`<br>`baseline_solver.py:181`<br>`cdft_env.c:81-97` | Anisotropic quaternion generation and flat bulk initial profiles ignoring wall exclusion. | Haar-measure $SO(3)$ uniform quaternion generation and initial profiles scaled by correlation length $\xi(T) \sim |1 - T/T_c|^{-\nu}$. |
| **Brittle Spatial Extraction** | `v1/utils/get_density_profile.py`<br>`v2/cuda_gcmc_kernels.cu` | `get_density_profile.py:179-183`<br>`cuda_gcmc_kernels.cu:493-497` | Fixed index slices `[mid - 10 : mid + 10]` and hardcoded 50-step sampling. | Algorithmic plateau detection ($\nabla \rho \to 0$) and Flyvbjerg-Petersen block averaging $\tau_{\rm int}$. |
| **Fake Boundaries & Limits** | `v1/external_potentials.py`<br>`v1/constants.py`<br>`v2/simulation_engine.cpp`<br>`v2/cuda_gcmc_kernels.cu` | `constants.py:24`<br>`external_potentials.py:48, 61`<br>`simulation_engine.cpp:327-329`<br>`cuda_gcmc_kernels.cu:395-396` | Use of `1e30` instead of `inf`, and artificial rejection (`prob = 0.0`) for `log_p >= 700.0`. | Enforce IEEE `std::numeric_limits<double>::infinity()` and exact Metropolis criterion $\min(1, \exp(\Delta \ln \mathcal{P}))$. |
