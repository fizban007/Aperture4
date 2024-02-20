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
kink_deriven(vector_field<Conf> &B, particle_data_t &ptc,
                     rng_states_t<exec_tags::device> &states) {
  using value_t = typename Conf::value_t;
  value_t B_z = sim_env().params().get_as<double>("B_z_ratio", 0.0);
  //value_t sigma = sim_env().params().get_as<double>("sigma", 4.0);
  value_t R_c = sim_env().params().get_as<double>("Radius", 1.0);

  value_t n_0 = sim_env().params().get_as<double>("background_n", 100.0);
  value_t n_c = sim_env().params().get_as<double>("core_n", 5000.0);
  value_t q_e = sim_env().params().get_as<double>("q_e", 1.0);
  value_t ppc = sim_env().params().get_as<double>("ppc", 8.0);
  

  
  value_t B0 = 100.0;
  auto &grid = B.grid();
  auto ext = grid.extent();

  // Initialize the magnetic field values
  B.set_values(0, [B0, R_c](auto x, auto y, auto z) {
    return -B0 * y * std::exp(1 - std::pow((x * x + y * y), 0.5) / R_c)/R_c;
  });
  B.set_values(1, [B0, R_c](auto x, auto y, auto z) {
    return B0 * x * std::exp(1 - std::pow((x * x + y * y), 0.5) / R_c)/R_c;
  });
  B.set_values(2, [B0, B_z](auto x, auto y, auto z) { return B0 * B_z; });

  
  ptc_injector_dynamic<Conf> injector(grid);

  // Jet particles
  injector.inject_pairs( 
      // Injection criterion
      [] LAMBDA(auto &pos, auto &grid, auto &ext) { return true; },
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) {return 2 * ppc; },
      [n_0, n_c] LAMBDA(auto &x_global,
                                      rand_state &state, PtcType type) {
        auto x = x_global[0];
        auto y = x_global[1];
        value_t kT = -500.0 * std::exp(- 2.0 * std::sqrt(x * x + y * y))
                         * (-39.0 * std::exp(2.0 * std::sqrt(x * x + y * y)) + 5.0 * std::exp(2.0 + 2.0 * std::sqrt(x * x + y * y)) + 5.0 * std::exp(2.0) * (-1 -2.0 * std::sqrt(x * x + y * y) + 2.0 * (x * x + y * y)))  
                                / (n_0 + (n_c - n_0) / square(std::cosh(2.0 * std::sqrt(x * x + y * y))));

        value_t beta_d =  100.0 * std::exp(1.0 - std::sqrt(x * x + y * y)) * (2.0 - std::sqrt(x * x + y * y))
                                / (n_0 + (n_c - n_0) / square(std::cosh(2.0 * std::sqrt(x * x + y * y))));

        vec_t<value_t, 3> u_d = rng_maxwell_juttner_drifting(state, kT, beta_d);
        value_t sign = 1.0f;
        if (type == PtcType::electron) sign *= -1.0f;

        auto p1 = u_d[1] * sign;
        auto p2 = u_d[2] * sign;
        auto p3 = u_d[0] * sign;
        return vec_t<value_t, 3>(p1, p2, p3);
      },
      // Particle weight
      [n_0, n_c, R_c, ppc] LAMBDA(auto &x_global, PtcType type) {
        auto x = x_global[0];
        auto y = x_global[1];
        double n = n_0 + (n_c - n_0) /  square(std::cosh(2.0 * std::sqrt(x * x + y * y)));
        return n/2.0/ppc;
      });

  Logger::print_info("After initial condition, there are {} particles",
                     ptc.number());
}



template void kink_deriven<Config<3>>(vector_field<Config<3>> &B,
                                                     particle_data_t &ptc,
                                                     rng_states_t<exec_tags::device> &states);
                                      

}  // namespace Aperture
