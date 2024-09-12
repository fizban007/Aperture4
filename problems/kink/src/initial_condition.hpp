#include "core/math.hpp"
#include "core/random.h"
// #include "data/curand_states.h"
#include "data/fields.h"
#include "data/particle_data.h"
#include "data/rng_states.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/physics/lorentz_transform.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "utils/kernel_helper.hpp"

using namespace std;
namespace Aperture {


template <typename Conf>
void
kink_pressure_supported(vector_field<Conf> &B, particle_data_t &ptc,
                        rng_states_t<exec_tags::device> &states) {
  using value_t = typename Conf::value_t;
  value_t B0 = sim_env().params().get_as<double>("B0", 100.0);
  value_t B_z = sim_env().params().get_as<double>("B_z_ratio", 0.0);
  value_t sigma = sim_env().params().get_as<double>("sigma", 5.0);
  // value_t R_c = sim_env().params().get_as<double>("Radius", 1.0);
  // always assume R_c = 1 since it's our unit

  value_t n_c = B0 * B0 / sigma;
  value_t n_0 = sim_env().params().get_as<double>("n_0", n_c / 10.0);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);
  value_t ppc = sim_env().params().get_as<int64_t>("ppc", 4);
  
  auto &grid = B.grid();
  auto ext = grid.extent();

  // Initialize the magnetic field values
  auto Bphi = [B0] (auto r) {
    return B0 * r * std::exp(1 - r);
  };


  B.set_values(0, [Bphi](auto x, auto y, auto z) {
    // return -B0 * y * std::exp(1 - std::pow((x * x + y * y), 0.5) / R_c)/R_c;
    auto r = math::sqrt(x*x + y*y);
    return -y * Bphi(r) / r;
  });
  B.set_values(1, [Bphi](auto x, auto y, auto z) {
    // return B0 * x * std::exp(1 - std::pow((x * x + y * y), 0.5) / R_c)/R_c;
    auto r = math::sqrt(x*x + y*y);
    return x * Bphi(r) / r;
  });
  B.set_values(2, [B0, B_z](auto x, auto y, auto z) { return B0 * B_z; });

  // profiles of physical quantities
  auto jz_profile = [B0] LAMBDA (auto r) {
    return B0 * (2.0 - r) * math::exp(1.0 - r);
  };

  auto n_profile = [n_c, n_0] LAMBDA (auto r) {
    return n_0 + (n_c - n_0) / square(std::cosh(2.0 * r));
  };

  auto P_profile = [B0] LAMBDA (auto r) {
    return square(B0) * (0.25 * math::exp(2.0 - 2.0 * r) * (1.0 + 2.0 * r - 2.0 * r * r) + math::exp(-2.0));
  };
  
  ptc_injector_dynamic<Conf> injector(grid);

  // Jet particles
  injector.inject_pairs( 
      // Injection criterion
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) {return 2 * ppc; },
      [jz_profile, n_profile, P_profile] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        auto x = x_global[0];
        auto y = x_global[1];
        auto r = math::sqrt(x*x + y*y);
        // value_t kT = -500.0 * std::exp(- 2.0 * std::sqrt(x * x + y * y))
        //                  * (-39.0 * std::exp(2.0 * std::sqrt(x * x + y * y)) + 5.0 * std::exp(2.0 + 2.0 * std::sqrt(x * x + y * y)) + 5.0 * std::exp(2.0) * (-1 -2.0 * std::sqrt(x * x + y * y) + 2.0 * (x * x + y * y)))  
        //                         / (n_0 + (n_c - n_0) / square(std::cosh(2.0 * std::sqrt(x * x + y * y))));
        value_t kT = P_profile(r) / n_profile(r) / 2.0;

        // value_t beta_d =  100.0 * std::exp(1.0 - std::sqrt(x * x + y * y)) * (2.0 - std::sqrt(x * x + y * y))
        //                         / (n_0 + (n_c - n_0) / square(std::cosh(2.0 * std::sqrt(x * x + y * y))));
        value_t beta_d = jz_profile(r) / n_profile(r) / 2.0;

        vec_t<value_t, 3> u_d = rng_maxwell_juttner_drifting(state, kT, beta_d);
        value_t sign = 1.0f;
        if (type == PtcType::electron) sign *= -1.0f;

        auto p1 = u_d[1] * sign;
        auto p2 = u_d[2] * sign;
        auto p3 = u_d[0] * sign;
        return vec_t<value_t, 3>(p1, p2, p3);
      },
      // Particle weight
      [n_profile, ppc, q_e] LAMBDA(auto &x_global, PtcType type) {
        auto x = x_global[0];
        auto y = x_global[1];
        double n = n_profile(math::sqrt(x*x + y*y));
        return n / ppc / q_e;
      });

  Logger::print_info("After initial condition, there are {} particles",
                     ptc.number());
}

template <typename Conf>
void
kink_force_free(vector_field<Conf> &B, particle_data_t &ptc,
                rng_states_t<exec_tags::device> &states) {
  using value_t = typename Conf::value_t;
  value_t B0 = sim_env().params().get_as<double>("B0", 100.0);
  value_t zeta = sim_env().params().get_as<double>("pitch_zeta", 1.44);
  value_t n_0 = sim_env().params().get_as<double>("n_0", 400.0);
  value_t kT = sim_env().params().get_as<double>("kT", 1.0e-2);
  // value_t R_c = sim_env().params().get_as<double>("Radius", 1.0);
  // always assume R_c = 1 since it's our unit

  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);
  value_t ppc = sim_env().params().get_as<int64_t>("ppc", 5);

  auto &grid = B.grid();
  auto ext = grid.extent();

  // Initialize the magnetic field values
  auto Bz_profile = [B0, zeta] (auto r) {
    return B0 / std::pow(1 + r*r, zeta);
  };

  auto Bphi_profile = [B0, zeta, Bz_profile] (auto r) {
    return Bz_profile(r) * math::sqrt((std::pow(1 + r*r, 2.0 * zeta) - 1.0 - 2.0 * zeta * r*r)
                                   / (2.0 * zeta - 1)) / r;
  };

  auto Jz_profile = [B0, zeta] LAMBDA (auto r) {
    if (r < 2e-2) {
      return B0 * (2.0 * math::sqrt(zeta) - 4.0 / 3.0 * math::sqrt(zeta) * r*r * (1.0 + 2.0 * zeta));
    } else {
      return B0 * 2.0 * r*r * zeta * std::pow(1.0 + r*r, -1.0 - zeta) / 
             math::sqrt((std::pow(1 + r*r, 2.0 * zeta) - 1.0 - 2.0 * zeta * r*r)
                                   / (2.0 * zeta - 1));
    }
  };

  auto Jphi_profile = [B0, zeta] LAMBDA (auto r) {
    return 2.0 * r * zeta / std::pow(1.0 + r*r, 1 + zeta);
  };


  B.set_values(0, [Bphi_profile](auto x, auto y, auto z) {
    auto r = math::sqrt(x*x + y*y);
    return -y * Bphi_profile(r) / r;
  });
  B.set_values(1, [Bphi_profile](auto x, auto y, auto z) {
    auto r = math::sqrt(x*x + y*y);
    return x * Bphi_profile(r) / r;
  });
  B.set_values(2, [Bz_profile](auto x, auto y, auto z) {
    auto r = math::sqrt(x*x + y*y);
    return Bz_profile(r);
  });

  ptc_injector_dynamic<Conf> injector(grid);

  // Jet particles
  injector.inject_pairs( 
      // Injection criterion
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) {return 2 * ppc; },
      [Jz_profile, Jphi_profile, n_0, kT] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        auto x = x_global[0];
        auto y = x_global[1];
        auto r = math::sqrt(x*x + y*y);

        value_t beta_z = Jz_profile(r) / n_0 / 2.0;
        value_t beta_phi = Jphi_profile(r) / n_0 / 2.0;
        value_t beta = math::sqrt(beta_z * beta_z + beta_phi * beta_phi);
        value_t gamma = 1.0 / math::sqrt(1.0 - beta*beta);

        vec_t<value_t, 3> u = rng_maxwell_juttner_3d(state, kT);
        value_t sign = 1.0f;
        if (type == PtcType::electron) sign *= -1.0f;

        auto p1 = -y * beta_phi / (r + 1e-4) * gamma * sign + u[0];
        auto p2 = x * beta_phi / (r + 1e-4) * gamma * sign + u[1];
        auto p3 = beta_z * gamma * sign + u[2];
        if (gamma != gamma) {
          printf("NaN in pairs! beta_z is %f, beta_phi is %f, jz is %f\n", beta_z,
          beta_phi, Jz_profile(r));
        }
        return vec_t<value_t, 3>(p1, p2, p3);
      },
      // Particle weight
      [n_0, ppc, q_e] LAMBDA(auto &x_global, PtcType type) {
        return n_0 / ppc / q_e;
      });

  Logger::print_info("After initial condition, there are {} particles",
                     ptc.number());
}

template void kink_pressure_supported<Config<3>>(vector_field<Config<3>> &B,
                                                particle_data_t &ptc,
                                                rng_states_t<exec_tags::device> &states);

template void kink_force_free<Config<3>>(vector_field<Config<3>> &B, particle_data_t &ptc,
                              rng_states_t<exec_tags::device> &states);

}  // namespace Aperture