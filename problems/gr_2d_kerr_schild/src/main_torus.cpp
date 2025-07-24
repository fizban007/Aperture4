/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "core/cuda_control.h"
#include "core/enum_types.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "injector.h"
#include "systems/data_exporter.h"
#include "systems/domain_comm.h"
#include "systems/field_solver_gr_ks.h"
#include "systems/gather_tracked_ptc.h"
#include "systems/compute_moments_gr_ks.h"
#include "systems/grid_ks.h"
#include "systems/radiative_transfer_impl.hpp"
#include "systems/radiation/gr_ks_ic_radiation_scheme_fido.hpp"
#include "systems/policies/coord_policy_gr_ks_sph.hpp"
#include "systems/policies/exec_policy_dynamic.hpp"
#include "systems/ptc_injector_new.h"
#include "systems/ptc_updater.h"
#include "utils/util_functions.h"
#include "utils/hdf_wrapper.h"
#include "utils/interpolation.hpp"

using namespace std;

namespace Aperture {

  template <typename Conf>
  void initial_vacuum_wald(vector_field<Conf> &B0, vector_field<Conf> &D0,
                          const grid_ks_t<Conf> &grid);

  HOST_DEVICE Scalar Delta(Scalar r, Scalar a){
    return r * r - 2.0 * r + a * a;
  } 

  HOST_DEVICE Scalar Sigma( Scalar r, Scalar th,Scalar a){
    return r * r + a * a * math::cos(th) * math::cos(th);
  }

  // HOST_DEVICE Scalar blnormLower(Scalar r, Scalar th, Scalar a, Scalar p_t, Scalar p_r, Scalar p_th, Scalar Lz){
  //   Scalar Delta = blDelta(r,a);
  //   Scalar Sigma = blSigma(r,th,a);
  //   Scalar sin2 =  math::sin(th) * math::sin(th);
  //   Scalar cos2 =  math::cos(th) * math::cos(th);
  //   Scalar numerator =  p_t * p_t * (a * a * Delta * sin2 - (a * a + r * r) * (a * a + r * r)) + 
  //                     p_r * p_r * Delta * Delta + 
  //                     p_th * p_th * Delta + 
  //                     Lz * Lz * (Sigma - 2.0 * r)/sin2
  //                      - 4.0 * a * r * Lz * p_t;
  //   return numerator/(Delta * Sigma);
  // }
  
  

  HOST_DEVICE Scalar A(Scalar r, Scalar th, Scalar a){ 
    return (a * a + r * r) * (a * a + r * r) - a * a * Delta(r,a) * math::sin(th) * math::sin(th);
  }

  HOST_DEVICE Scalar blbetaphi(Scalar r, Scalar th, Scalar a){
    return  - 2.0 * a * r/A(r, th, a);
  }

  HOST_DEVICE Scalar blalpha(Scalar r, Scalar th, Scalar a){
    return math::sqrt(Delta(r,a) * Sigma(r,th,a)/A(r, th, a));
  }

  HOST_DEVICE Scalar ksalpha(Scalar r, Scalar th, Scalar a){
    return math::sqrt(Delta(r,a) * Sigma(r,th,a)/(r * r + a * a * math::cos(th) * math::cos(th)));
  }

  HOST_DEVICE Scalar ksDetGamma(Scalar r, Scalar th, Scalar a) {
    Scalar costh = math::cos(th);
    Scalar sinth = math::sin(th);
    Scalar common = r * r + a * a * costh * costh;
    return common * (r * r + 2.0 * r + a * a * costh * costh) * sinth * sinth;
  } 

  // HOST_DEVICE Scalar energy(Scalar r, Scalar th, Scalar a, Scalar p_r, Scalar p_th, Scalar lz){
  //   Scalar Delta = blDelta(r,a);
  //   Scalar Sigma = blSigma(r,th,a);
  //   Scalar sinth =  math::sin(th);
  //   Scalar discriminant = p_r * p_r * Delta * Delta + 
  //                     p_th * p_th * Delta + 
  //                     Lz * Lz * (Sigma - 2.0 * r)/sin2+1;
  //   return  blalpha(r, th, a)*math::sqrt(discriminant)- blbetaphi(r, th, a) * Lz;
  // }

  //Emin is in in the middle
  HOST_DEVICE Scalar Emin(Scalar r, Scalar th, Scalar a, Scalar Lz){
    Scalar Av = A(r, th, a);
    Scalar C = math::sqrt(Lz * Lz * Sigma(r,th,a)/(Av * math::sin(th) * math::sin(th)) + 1.0);//unsure what to call this
    Scalar alpha = blalpha(r, th, a);
    Scalar beta3 = blbetaphi(r, th, a);
    return  - beta3 * Lz + alpha * C;
  }

  HOST_DEVICE Scalar globalEmin(Scalar a, Scalar r0){
    Scalar r03= r0 * r0 * r0;
    Scalar term= a*a*a + a * r0 * (r0 - 2.0);
    Scalar sqrtterm = r03 * term * term;
    Scalar num = a*a * r0*r0 * (5.0-3.0 * r0) + r03 * (r0-3.0) * (r0-2.0) * (r0-2.0) - 2.0 * math::sqrt(sqrtterm);
    Scalar denom = r03 * ( (r0-3.0) * (r0-3.0) * r0 - 4.0 * a*a );
    return math::sqrt(num/denom);
  }

  HOST_DEVICE Scalar AngularMomentum(Scalar a,  Scalar r0, Scalar E0){
    Scalar a3= a * a * a;
    Scalar term= a3+ a * r0 * (r0 - 2.0);
    Scalar sqrtterm = r0*r0*r0 * term * term;
    Scalar num = a * a3 + a*a * r0 *(3.0* r0 - 4.0) - math::sqrt(sqrtterm);
    Scalar denom = a * (a*a - (r0 - 2.0)*(r0 - 2.0) * r0);
    return E0*num/denom;
  }

  HOST_DEVICE Scalar energyMax(Scalar a, Scalar rmin, Scalar Lz){
    return Emin(rmin,M_PI/2, a, Lz); 
  }


  // HOST_DEVICE Scalar blp_r_max(Scalar r, Scalar th, Scalar a, Scalar Emax, Scalar Lz) {
  //   Scalar Delta = Metric_KS::Delta(a, r);
  //   Scalar Sigma = Metric_KS::rho2(a, r, th);
  //   return math::sqrt(Sigma/Delta * invp_t(r, th, a, Emax, Lz));
  // }

  // HOST_DEVICE Scalar blp_th_max(Scalar r, Scalar th, Scalar a, Scalar Emax, Scalar Lz) {
  //   Scalar Sigma = Metric_KS::rho2(a, r, th);
  //   return math::sqrt(Sigma * invp_t(r, th, a, Emax, Lz));
  // }
 
  




  HOST_DEVICE vec_t<Scalar,3> torus_momentum(Scalar r, Scalar th, Scalar a, Scalar Emax, Scalar Temp, Scalar Lz, rand_state& state) {
    // Minimum allowed energy at this position
    Scalar Sigmav = Sigma(r,th,a);
    Scalar Av = A(r, th, a);
    Scalar C = math::sqrt(Lz * Lz * Sigmav/(Av * math::sin(th) * math::sin(th)) + 1);//unsure what to call this
    Scalar alpha = blalpha(r, th, a);
    Scalar beta3 = blbetaphi(r, th, a);
    Scalar Emin_val =  - beta3 * Lz + alpha * C;
    if ( Emin_val >=   Emax) {
      // printf("Emin >=  Emax: r - >%f, th - >%f, Emin - >%f, Emax - >%f\n", r, th, Emin_val,Emax);
      return vec_t<Scalar, 3>{0, 0, 0};
    }
    Scalar Deltav = Delta(r,a);
    Scalar That = Temp/(alpha * C);
    Scalar phatval = (Emax + beta3 * Lz)/(alpha * C);
    Scalar phatmax2 =  phatval * phatval - 1;
    //if T<<1
    Scalar u1 =  rng_uniform<Scalar>(state);
    Scalar Z = 1 - math::exp( - phatmax2/(2.0 * That));
    Scalar phat = math::sqrt( - 2.0 * That * math::log(1 - u1 * Z));
    Scalar energy =  - beta3 * Lz + alpha * C * math::sqrt(1 + phat * phat);
    Scalar u2 =  rng_uniform<Scalar>(state);
    Scalar blp_r = C * math::sqrt(Sigmav/Deltav) * phat * math::cos(2.0 * M_PI * u2);
    Scalar blp_th = C * math::sqrt(Sigmav) * phat * math::sin(2.0 * M_PI * u2);
    Scalar ksp_r = blp_r - (a * Lz - 2.0 * r * energy) / Deltav;
    return vec_t<Scalar, 3>{ksp_r, blp_th, Lz};
  }
  
  HOST_DEVICE Scalar torus_Density(Scalar r, Scalar th, Scalar a, Scalar E0, Scalar Emax, Scalar Temp, Scalar Lz) {
    Scalar Deltav = Delta(r,a);
    Scalar Sigmav = Sigma(r, th, a);
    Scalar Av = A(r, th, a);
    Scalar sinth = math::sin(th);
    Scalar prefactor = 2.0 * M_PI * Temp/math::sqrt(Deltav* sinth * sinth);
    Scalar Eminv = Emin(r, th, a, Lz);
    Scalar blS_t = prefactor * ( (Temp+Emax) * math::exp( -(Emax-E0)/Temp) - (Temp+Eminv) * math::exp( -(Eminv-E0)/Temp) );
    Scalar blS_phi = Lz * prefactor * (math::exp( -(Eminv-E0)/Temp) - math::exp( -(Emax-E0)/Temp) );
    Scalar blSt= - (Av * blS_t + 2.0 * a * r * blS_phi)/(Deltav * Sigmav);
    return Metric_KS::alpha(r, th, a) * blSt;
  }
  

}  // namespace Aperture



using namespace Aperture;

int
main(int argc, char  * argv[]) {
  typedef Config<2> Conf;
  using value_t = Conf::value_t;

  auto &env = sim_environment::instance(&argc, &argv, true);

  // env.params().add("log_level", (int64_t)LogLevel::debug);

  domain_comm<Conf, exec_policy_dynamic> comm;
  grid_ks_t<Conf> grid(comm);

  auto pusher = env.register_system<ptc_updater<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph>>(grid, &comm);
  auto moments = env.register_system<compute_moments_gr_ks<Conf, exec_policy_dynamic>>(grid);
  // auto injector = env.register_system<bh_injector<Conf>>(grid);
  auto tracker  =  env.register_system<gather_tracked_ptc<Conf, exec_policy_dynamic>>(grid);
  // auto radiation = env.register_system<
  //   radiative_transfer<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph,
  //                     //  default_radiation_scheme_gr>>(grid, &comm);
  //                      gr_ks_ic_radiation_scheme_fido>>(grid, &comm);
  auto solver = env.register_system<field_solver<Conf, exec_policy_dynamic, coord_policy_gr_ks_sph>>(grid, &comm);
  auto exporter  = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(grid, &comm);

  env.init();

  int ppc = 20;
  env.params().get_value("ppc", ppc);
  double set_max_density = 1.0;
  env.params().get_value("torus_max_density", set_max_density);
  //this is the radius of the inner edge of the torus
  double r_lower = 10.0;
  env.params().get_value("torus_r_lower", r_lower);
  //this is an upper bound on the radius of maximum density
  double r_max = 14.0;
  env.params().get_value("torus_r_max", r_max);
  double spin = 0.999;
  env.params().get_value("bh_spin", spin);
  double temp = 0.001;
  env.params().get_value("torus_temp", temp);
  double Bp = 1.0;
  env.params().get_value("Bp", Bp);
  //some constants
  double E0= globalEmin(spin, r_max);
  double Lz = AngularMomentum(spin, r_max, E0);
  double Emax = energyMax(spin, r_lower, Lz);
  double max_density = torus_Density(r_max, M_PI/2, spin, E0, Emax, temp, Lz);//This is wrong

  // Prepare initial field 
  vector_field<Conf> *B, *D, *B0, *D0;
  env.get_data("B0", &B0);
  env.get_data("E0", &D0);
  env.get_data("Bdelta", &B);
  env.get_data("Edelta", &D);

//magnetic field loop
exec_policy_dynamic<Conf>::launch(
    [Bp, E0, Emax, Lz, temp, spin, r_lower ,max_density] LAMBDA(auto B, auto D, auto a) {
      auto &grid = exec_policy_dynamic<Conf>::grid();
      auto ext = grid.extent();
      exec_policy_dynamic<Conf>::loop(
          Conf::begin(ext), Conf::end(ext), [&] LAMBDA(auto idx) {
            auto pos = get_pos(idx, ext);
            auto r = grid_ks_t<Conf>::radius(
                grid.template coord<0>(pos[0], false));
            auto r_s =
                grid_ks_t<Conf>::radius(grid.template coord<0>(pos[0], true));
            auto th =
                grid_ks_t<Conf>::theta(grid.template coord<1>(pos[1], false));
            auto th_s =
                grid_ks_t<Conf>::theta(grid.template coord<1>(pos[1], true));
            if (math::abs(th_s) < TINY){th_s = (th_s < 0.0f ? -1.0f : 1.0f) * 0.01 * grid.delta[1];}
            Scalar rho=torus_Density(r_s, th_s, spin, E0,Emax, temp, Lz); 
            Scalar eps = 1e-6;//come up with a better standard for this. 
            Scalar prefactor =  Bp / max_density /(2.0 * eps)/math::sqrt( ksDetGamma(r_s, th_s, spin));
            if(rho/max_density > 0.2 && r_s>r_lower){
              Scalar dAdr = torus_Density(r_s + eps, th_s, spin, E0, Emax, temp, Lz)-
                          torus_Density(r_s - eps, th_s, spin, E0, Emax, temp, Lz);          
              Scalar dAdth = torus_Density(r_s, th_s+eps, spin, E0, Emax, temp, Lz)-
                            torus_Density(r_s, th_s-eps, spin, E0, Emax, temp, Lz);
                                      
              B[0][idx] = prefactor * dAdth;
              B[1][idx] = - prefactor * dAdr;
            }
            else{
              B[0][idx] = 0.0;
              B[1][idx] = 0.0;
              
            }
            B[2][idx] = 0.0;
            D[0][idx] = 0.0;
            D[1][idx] = 0.0;
            D[2][idx] = 0.0;
          });
      },
        *B, *D, grid.a);
  exec_policy_dynamic<Conf>::sync();

  ptc_injector_dynamic<Conf> ptc_inj(grid);
  ptc_inj.inject_pairs(
      // First function is the injection criterion for each cell. pos is an
      // index_t<Dim> object marking the cell in the grid. Returns true for
      // cells that inject and false for cells that do nothing.
      [spin,Lz,Emax,r_lower] LAMBDA(auto &pos, auto &grid, auto &ext) {
        auto r = grid_ks_t<Conf>::radius( grid.template coord<0>(pos[0], false));
        auto th = grid_ks_t<Conf>::theta(grid.template coord<1>(pos[1], false));
        if (r < r_lower) {return false;}
        if( Emin(r,th, spin, Lz) >=  Emax) {return false;}
        return true;
      },
      // Second function returns the number of particles injected in each cell.
      // This includes all species
      [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) {
        return ppc;
      },
      // Third function is the momentum distribution of the injected particles.
      // Returns a vec_t<value_t, 3> object encoding the 3D momentum of this
      // particular particle
      [spin,Emax,temp,Lz] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
        value_t r = grid_ks_t<Conf>::radius(x_global[0]);  
        value_t theta = grid_ks_t<Conf>::theta(x_global[1]); 
        return torus_momentum(r, theta, spin, Emax, temp, Lz, state);
      },
      // Fourth function is the particle weight, which can depend on the global
      // coordinate.
      [ppc,spin,Lz,E0,Emax,temp,set_max_density,max_density,r_lower] LAMBDA(auto &x_global, PtcType type) {
        value_t r = grid_ks_t<Conf>::radius(x_global[0]);
        value_t th = grid_ks_t<Conf>::theta(x_global[1]);
        if (r < r_lower) {return 0.0;}
        if( Emin(r,th, spin, Lz)  >=   Emax) {return 0.0;}
        value_t density= torus_Density(r, th, spin, E0, Emax, temp, Lz);
        value_t sqrt_gamma = Metric_KS::sqrt_gamma(spin, r, th);
        return set_max_density/max_density * (density * r * sqrt_gamma) / ppc; 
      });

  env.run();

  return 0;
}
