/*
 * Copyright (c) 2023 Alex Chen.
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

 #pragma once

 #include "data/fields.h"
 #include "data/rng_states.h"
 #include "framework/environment.h"
 #include "framework/system.h"
 #include "systems/domain_comm.h"
 #include "systems/grid_sph.hpp"
 #include "utils/nonown_ptr.hpp"
 #include "utils/util_functions.h"
 
 #include "systems/ptc_injector_new.h"
 #include <memory>
 
 namespace Aperture {
 
 template <typename Conf, template <class> class ExecPolicy>
 class constant_injection : public system_t {
  public:
   typedef typename Conf::value_t value_t;
 
   static std::string name() { return "constant_injection"; }
 
   constant_injection(const grid_sph_t<Conf>& grid,
                      const domain_comm<Conf, ExecPolicy>* comm = nullptr)
       : m_grid(grid), m_comm(comm) {
     if (comm != nullptr) {
       m_track_rank = comm->rank();
       m_track_rank <<= 32;
     }
   }
   ~constant_injection() = default;
 
   void init() override {
     sim_env().get_data("particles", ptc);
     sim_env().get_data("rng_states", rng_states);
 
     sim_env().params().get_value("atm_time", m_atm_time);
     
     sim_env().params().get_value("ramp_time", ramp_time);
     sim_env().params().get_value("twist_omega", m_twist_omega);
     sim_env().params().get_value("inject_interval", m_inject_interval);
     sim_env().params().get_value("twist_rmax_1", m_twist_rmax_1);
     sim_env().params().get_value("twist_rmax_2", m_twist_rmax_2);
     sim_env().params().get_value("q_e", m_qe);
     sim_env().params().get_value("tracked_fraction", m_tracked_fraction);
    
   }
 
   void update(double dt, uint32_t step) override {
     // Check for multi_rank and when we are at r=r_* boundary
    //  if (m_comm != nullptr && !m_comm->domain_info().is_boundary[0]) {
    //    return;
    //  }
 
     auto ext = m_grid.extent();
     typedef typename Conf::idx_t idx_t;
     
        // domain_comm<Conf, exec_policy_dynamic> comm;
        // grid_sph_t<Conf> grid(comm);
        auto* comm = m_comm;
        auto& grid = m_grid;

     // Impart twist on the stellar surface
     value_t twist_th1 = math::asin(math::sqrt(1.0 / m_twist_rmax_1));
     value_t twist_th2 = math::asin(math::sqrt(1.0 / m_twist_rmax_2));
     Logger::print_debug("time is {}, th1 is {}, th2 is {}", time,
                          twist_th1, twist_th2);
 
        // printf("line 80");
     // Inject particles at the surface according to surface electric field
     // Define a variable to hold the moving position in the ptc array where we
     // insert new particles
     buffer<unsigned long long int> pos(1, ExecPolicy<Conf>::data_mem_type());
    //  pos[0] = ptc->number();
    //  pos.copy_to_device();
    //  auto ptc_num_orig = pos[0];
    //  value_t qe = m_qe;
    //  value_t tracked_fraction = m_tracked_fraction;
    //  auto track_rank = m_track_rank;
        auto inject_interval = m_inject_interval;
    value_t time = dt * step;
    value_t ppc = 10;
    // printf("line 94");
    // only running if we are on a good step
    if (std::fmod(time, inject_interval) > 0.5 * dt) {return;}
    ptc_injector_dynamic<Conf> injector1(m_grid);
     injector1.inject_pairs(
       // Injection criterion (in flux tube)
         [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) { 
          //  return true; // Inject everywhere
          //  TODO check if this is the actual r we are trying to input
           auto r = math::exp(grid.template coord<0>(pos, 0.5f));// Since r_grid = log(r)
           auto th = grid.template coord<1>(pos, 0.5f);
          //  auto r_max_check = r / (sin(th) * sin(th));
           if (r>1.01 && r < 1.1  ){//((th < 0.7853 && th > 0.46364) || (th < M_PI - 0.4634 && th > M_PI - 0.7853) )) { //&& r_max_check > tube_rmax_1 && r_max_check < tube_rmax_2) {
               //printf("r_max_check is %f\n tube_rmax_1/2 is %f,%f", r_max_check, tube_rmax_1, tube_rmax_2);
             return true;
           }else {
             return false;
           }
           },
         // Number injected per cell
         [ppc] LAMBDA(auto &pos, auto &grid, auto &ext) { return 2* ppc; },
           // Initialize particle momentum
         [] LAMBDA(auto &x_global, rand_state &state, PtcType type) {
           auto &grid = static_cast<const grid_sph_t<Conf> &>(
               exec_policy_dynamic<Conf>::grid());
           auto r =grid.radius(x_global[0]);
           auto th = grid.theta(x_global[1]);
           // From Belobodorov 2013 gamma is 100b where b = B/BQ
           //TODO check if this is the correct conversion to momentum
           value_t gamma = 300; //100 * 10 / cube(r) * math::sqrt( 1 + 3*cos(th)*cos(th));
           if (gamma < 1.0) gamma = 1.0; // Lower LImit since the previous equation does not gaurantee gamma > 1
           if (gamma > 2000.0) gamma = 2000.0;// Upper limit since numerical issues ariese at high gamma
          //  gamma = 1000;
           value_t p0 = math::sqrt(gamma*gamma - 1.0); // gamma = sqrt(1+p^2)
          //  p0 = p0 / math::sqrt( 1.0 + 3.0 * cos(th)*cos(th) ); // Normalization
          // printf("Injecting at r,th,gamma,p0 = %f,%f,%f,%f \n",r,th,gamma,p0);
           if (th <= M_PI / 2.0) {
             // return vec_t<value_t, 3>(p0 * 2.0 * cos(th), p0 * sin(th), 0);
             return vec_t<value_t, 3>(p0, 0, 0); // Inject at the gamma for p_para
           } else {
             // return vec_t<value_t, 3>(-p0 * 2.0 * cos(th), -p0 * sin(th), 0);
             return vec_t<value_t, 3>(-p0, 0, 0); // Inject at the gamma for p_para
           }
           //return vec_t<value_t, 3>(p0 * 2.0 * cos(th), p0 * sin(th), 0);
         },
           // Initialize particle weight (i.e makes each particle worth a different amount of "actual" particles)
         [] LAMBDA(auto &x_global, PtcType type) {
           // auto &grid = static_cast<const grid_sph_t<Conf> &>(
           //     exec_policy_dynamic<Conf>::grid());
           //auto r = grid_sph_t<Conf>::radius(x_global[0]);
           auto th = grid_sph_t<Conf>::theta(x_global[1]);
           return 1.0 * math::sin(th);
         }
         );
    //  ExecPolicy<Conf>::launch(
    //      [ext, qe, tracked_fraction, track_rank,time,dt, inject_interval,twist_th1,twist_th2] LAMBDA(
    //          auto e, auto ptc, auto ptc_pos, auto ptc_id, auto states) {
    //        auto& grid = ExecPolicy<Conf>::grid();
    //        printf("rng_t");
    //        rng_t<typename ExecPolicy<Conf>::exec_tag> rng(states);
    //        ExecPolicy<Conf>::loop(0, grid.dims[1], [&] LAMBDA(auto n1) {
    //          value_t theta =
    //              grid_sph_t<Conf>::theta(grid.template coord<1>(n1, false));
    //          int n0 = grid.guard[0] + 3;
    //          auto idx = idx_t(index_t<2>(n0, n1), ext);
    //         printf("if statement");
    //          // I have inject_interval, I want to inject every inject_interval
    //          if (theta < twist_th1 && theta > twist_th2 && std::fmod(time, inject_interval) < 0.5 * dt) {
    //         //    value_t w = math::abs(E_surface) / qe * 0.1;
    //             value_t w = 0.1;
    //            size_t ptc_offset = atomic_add(&ptc_pos[0], 2);
    //             printf("line 112");
    //            if (ptc.cell[ptc_offset] != empty_cell) {
    //              return;
    //            }
    //            value_t gamma = 100 * 10 / cube(1.0f) * math::sqrt( 1 + 3*cos(theta)*cos(theta));
    //             if (gamma < 1.0) gamma = 1.0; // Lower Limit since the previous equation does not gaurantee gamma > 1
    //             if (gamma > 2000.0) gamma = 2000.0;// Upper limit since numerical issues ariese at high gamma
    //             value_t p0 = math::sqrt(gamma*gamma - 1.0);
    //            float u = rng.template uniform<float>();
    //            // ptc_offset is electron and ptc_offset + 1 is positron
    //            printf("line 121");
    //            ptc.x1[ptc_offset] = ptc.x1[ptc_offset + 1] = 0.0f;
    //            ptc.x2[ptc_offset] = ptc.x2[ptc_offset + 1] = u;
    //            ptc.x3[ptc_offset] = ptc.x3[ptc_offset + 1] = 0.0f;
    //            // Initializing the particle at rest
    //            ptc.p1[ptc_offset] = ptc.p1[ptc_offset + 1] = p0;
    //            ptc.p2[ptc_offset] = ptc.p2[ptc_offset + 1] = 0.0f;
    //            ptc.p3[ptc_offset] = ptc.p3[ptc_offset + 1] = 0.0f;
    //            ptc.E[ptc_offset] = ptc.E[ptc_offset + 1] = gamma;
 
    //            ptc.weight[ptc_offset] = ptc.weight[ptc_offset + 1] =
    //                w * math::sin(theta);
    //            ptc.cell[ptc_offset] = ptc.cell[ptc_offset + 1] = idx.linear;
    //            u = rng.template uniform<float>();
    //            uint32_t flag = 0;
    //            if (u < tracked_fraction) {
    //              flag = flag_or(PtcFlag::tracked);
    //              ptc.id[ptc_offset] = track_rank + atomic_add(ptc_id, 1);
    //              ptc.id[ptc_offset + 1] = track_rank + atomic_add(ptc_id, 1);
    //            }
    //            ptc.flag[ptc_offset] = set_ptc_type_flag(flag, PtcType::electron);
    //            ptc.flag[ptc_offset + 1] =
    //                set_ptc_type_flag(flag, PtcType::positron);
    //          }
    //        });
    //      },
    //      E, ptc, pos, ptc->ptc_id(), rng_states);
    //  ExecPolicy<Conf>::sync();
 
    //  pos.copy_to_host();
    //  ptc->set_num(pos[0]);
    //  Logger::print_info("{} particles are injected!", pos[0] - ptc_num_orig);
   }
 
  private:
   const grid_sph_t<Conf>& m_grid;
   const domain_comm<Conf, ExecPolicy>* m_comm = nullptr;
   value_t m_twist_omega = 0.0;
   value_t m_atm_time = 5.0;
   value_t m_inject_interval = 20.0;
   value_t m_twist_rmax_1 = 3.0;
   value_t m_twist_rmax_2 = 5.0;
   value_t m_qe = 1.0;
   value_t m_tracked_fraction = 0.1;
   value_t m_min_surface_E = 0.01;
   uint64_t m_track_rank = 0;
   value_t Bp = 1000;
   value_t BQ = 100;
 
   value_t ramp_time = 1.0;
 
   nonown_ptr<vector_field<Conf>> E, B, E0, B0;
   nonown_ptr<particles_t> ptc;
   nonown_ptr<rng_states_t<typename ExecPolicy<Conf>::exec_tag>> rng_states;
 };
 
 }  // namespace Aperture
 