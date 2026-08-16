#ifndef GCMC_V2_CORE_TYPES_H
#define GCMC_V2_CORE_TYPES_H

#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

namespace gcmc_v2 {

constexpr double VERY_LARGE_NUMBER = 1.0e30;
constexpr double PI = 3.14159265358979323846;
constexpr double ELEMENTARY_CHARGE = 1.602176634e-19;
constexpr double EPSILON_0 = 8.8541878128e-12;
constexpr double AVOGADRO = 6.02214076e23;
constexpr double KB_DEFAULT = 1.380649e-23;

struct Vec3 {
    double x, y, z;

    inline Vec3() : x(0.0), y(0.0), z(0.0) {}
    inline Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

    inline Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    inline Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    inline Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
    inline Vec3 operator/(double s) const { return Vec3(x / s, y / s, z / s); }
    inline Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    inline Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }

    inline double norm_sq() const { return x * x + y * y + z * z; }
    inline double norm() const { return std::sqrt(norm_sq()); }

    inline Vec3 minimum_image(double lx, double ly, double lz) const {
        return Vec3(
            x - lx * std::round(x / lx),
            y - ly * std::round(y / ly),
            z - lz * std::round(z / lz)
        );
    }

    inline Vec3 wrap_pbc(double lx, double ly, double lz) const {
        return Vec3(
            x - lx * std::floor(x / lx),
            y - ly * std::floor(y / ly),
            z - lz * std::floor(z / lz)
        );
    }
};

struct Quaternion {
    double w, x, y, z;

    inline Quaternion() : w(1.0), x(0.0), y(0.0), z(0.0) {}
    inline Quaternion(double _w, double _x, double _y, double _z) : w(_w), x(_x), y(_y), z(_z) {}

    inline Quaternion operator*(const Quaternion& q) const {
        return Quaternion(
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w
        );
    }

    inline Vec3 rotate(const Vec3& v) const {
        // v' = q * (0, v) * q_conj
        Quaternion p(0.0, v.x, v.y, v.z);
        Quaternion q_conj(w, -x, -y, -z);
        Quaternion res = (*this) * p * q_conj;
        return Vec3(res.x, res.y, res.z);
    }
};

enum class MoleculeType {
    NONE = 0,
    SINGLE_TYPE,
    TWO_TYPE_RPM,
    ABC_DIPOLE,
    H2O_SPCE,
    CO2
};

enum class PotentialKind {
    NONE = 0,
    LJ,
    WCA,
    HS,
    HS_C,
    LJ_C
};

struct PairPotentialParams {
    PotentialKind kind = PotentialKind::NONE;
    double epsilon_lj = 0.0;
    double sigma_lj = 1.0;
    double rc = 10.0;
    double epsilon_c = 1.0;
    double q1 = 0.0;
    double q2 = 0.0;
    double kappa_inv = 4.5;
    double diameter = 2.76;
    double prefactor = 0.0;
    double shift_lj = 0.0;

    void init() {
        if (kind == PotentialKind::LJ || kind == PotentialKind::LJ_C) {
            double r6 = std::pow(sigma_lj / rc, 6.0);
            double r12 = r6 * r6;
            shift_lj = 4.0 * epsilon_lj * (r12 - r6);
        }
        if (kind == PotentialKind::HS_C || kind == PotentialKind::LJ_C) {
            prefactor = (ELEMENTARY_CHARGE * ELEMENTARY_CHARGE) /
                        (4.0 * PI * EPSILON_0 * 1.0e-10 * epsilon_c);
        }
    }

    inline double calculate(double r) const {
        if (r >= rc) return 0.0;
        switch (kind) {
            case PotentialKind::LJ: {
                double r6 = std::pow(sigma_lj / r, 6.0);
                double r12 = r6 * r6;
                return 4.0 * epsilon_lj * (r12 - r6) - shift_lj;
            }
            case PotentialKind::WCA: {
                double r_wca = std::pow(2.0, 1.0 / 6.0) * sigma_lj;
                if (r >= r_wca) return 0.0;
                double r6 = std::pow(sigma_lj / r, 6.0);
                double r12 = r6 * r6;
                return 4.0 * epsilon_lj * (r12 - r6) + epsilon_lj;
            }
            case PotentialKind::HS: {
                return (r < sigma_lj) ? VERY_LARGE_NUMBER : 0.0;
            }
            case PotentialKind::HS_C: {
                if (r < diameter) return VERY_LARGE_NUMBER;
                return prefactor * q1 * q2 * std::erfc(r / kappa_inv) / r;
            }
            case PotentialKind::LJ_C: {
                double r6 = std::pow(sigma_lj / r, 6.0);
                double r12 = r6 * r6;
                double u_lj = 4.0 * epsilon_lj * (r12 - r6) - shift_lj;
                double u_c = prefactor * q1 * q2 * std::erfc(r / kappa_inv) / r;
                return u_lj + u_c;
            }
            default:
                return 0.0;
        }
    }
};

enum class ExternalPotentialKind {
    NONE = 0,
    WALL,
    SLIT,
    SLIT_LJ,
    SLIT_LJ93,
    TRAINING_POTENTIAL_WITH_CHARGE_COS,
    TRAINING_POTENTIAL_WITH_WALLS,
    GENERIC
};

struct LinearPotentialSegment {
    double Va, Vb, xa, xb;
};

struct ExternalPotentialParams {
    ExternalPotentialKind kind = ExternalPotentialKind::NONE;
    double low = 2.0;
    double high = 18.0;
    double width = 2.0;
    double L = 20.0;
    double epsilon = 1.0;
    double sigma = 1.0;
    double cutoff = 0.858374218933;
    double q = 0.0;
    double A1 = 0.0, A2 = 0.0, A3 = 0.0, A4 = 0.0;
    double phi1 = 0.0, phi2 = 0.0, phi3 = 0.0, phi4 = 0.0;
    double q_A1 = 0.0, q_A2 = 0.0, q_A3 = 0.0, q_A4 = 0.0;
    double q_phi1 = 0.0, q_phi2 = 0.0, q_phi3 = 0.0, q_phi4 = 0.0;
    double shift = 0.0;

    std::vector<LinearPotentialSegment> linear_potentials;
    std::vector<LinearPotentialSegment> q_linear_potentials;

    void init() {
        if (cutoff > 0.0) {
            double r9 = std::pow(sigma / cutoff, 9.0);
            double r3 = std::pow(sigma / cutoff, 3.0);
            shift = epsilon * ((2.0 / 15.0) * r9 - r3);
        }
    }

    inline double calculate(const Vec3& pos) const {
        if (kind == ExternalPotentialKind::NONE) return 0.0;

        double x_coord = pos.x;

        if (kind == ExternalPotentialKind::WALL) {
            if (x_coord < width || x_coord > L - width) return VERY_LARGE_NUMBER;
            return 0.0;
        }

        if (kind == ExternalPotentialKind::SLIT) {
            if (x_coord < low || x_coord > high) return VERY_LARGE_NUMBER;
            return 0.0;
        }

        if (kind == ExternalPotentialKind::TRAINING_POTENTIAL_WITH_CHARGE_COS) {
            double bound_lo = (low > 0.0) ? low : (width / 2.0);
            double bound_hi = (high < L) ? high : (L - width / 2.0);
            if (x_coord < bound_lo || x_coord > bound_hi) return VERY_LARGE_NUMBER;

            double ratio = 2.0 * PI * x_coord / L;

            double sine_terms = A1 * std::sin(ratio * 1.0 + phi1) +
                                A2 * std::sin(ratio * 2.0 + phi2) +
                                A3 * std::sin(ratio * 3.0 + phi3) +
                                A4 * std::sin(ratio * 4.0 + phi4);

            double sine_terms_q = q_A1 * std::sin(ratio * 1.0 + q_phi1) +
                                  q_A2 * std::sin(ratio * 2.0 + q_phi2) +
                                  q_A3 * std::sin(ratio * 3.0 + q_phi3) +
                                  q_A4 * std::sin(ratio * 4.0 + q_phi4);

            double linear_terms = 0.0;
            for (const auto& seg : linear_potentials) {
                if (x_coord >= seg.xa && x_coord <= seg.xb && (seg.xb > seg.xa)) {
                    double vlin = seg.Va + (seg.Vb - seg.Va) * (x_coord - seg.xa) / (seg.xb - seg.xa);
                    linear_terms += vlin;
                }
            }

            double linear_terms_q = 0.0;
            for (const auto& seg : q_linear_potentials) {
                if (x_coord >= seg.xa && x_coord <= seg.xb && (seg.xb > seg.xa)) {
                    double vlin_q = seg.Va + (seg.Vb - seg.Va) * (x_coord - seg.xa) / (seg.xb - seg.xa);
                    linear_terms_q += vlin_q;
                }
            }

            return sine_terms + linear_terms + (sine_terms_q + linear_terms_q) * q;
        }

        return 0.0;
    }
};

struct Molecule {
    Vec3 sites[3];
    int num_sites = 3;
    int species_id = 0; // 0=A/O/H1, 1=B/H1/O, 2=C/H2
};

} // namespace gcmc_v2

#endif // GCMC_V2_CORE_TYPES_H
