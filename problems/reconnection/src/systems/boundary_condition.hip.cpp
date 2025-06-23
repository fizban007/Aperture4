/*
 * Copyright (c) 2021 Alex Chen.
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

#include "boundary_condition.h"
#include "core/math.hpp"
#include "framework/config.h"
#include "systems/grid.h"
#include "systems/physics/lorentz_transform.hpp"
#include "systems/policies/exec_policy_gpu.hpp"
#include "systems/ptc_injector_new.h"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/util_functions.h"

namespace Aperture {

HOST_DEVICE Scalar
pml_sigma(Scalar x, Scalar xh, Scalar pmlscale, Scalar sig0) {
  if (x > xh)
    return sig0 * square((x - xh) / pmlscale);
  else
    return 0.0;
}

template <typename Conf>
boundary_condition<Conf>::boundary_condition(
    const grid_t<Conf> &grid, const domain_comm<Conf, exec_policy_gpu> *comm)
    : m_grid(grid), m_comm(comm) {
  using multi_array_t = typename Conf::multi_array_t;

  sim_env().params().get_value("damping_coef", m_damping_coef);
  value_t sigma = 10.0;
  sim_env().params().get_value("sigma", sigma);
  m_Bp = math::sqrt(sigma);
  sim_env().params().get_value("damping_length", m_damping_length);
  sim_env().params().get_value("upstream_kT", m_upstream_kT);
  sim_env().params().get_value("upstream_n", m_upstream_n);
  sim_env().params().get_value("guide_field", m_Bg);
  sim_env().params().get_value("boost_beta", m_boost_beta);
  sim_env().params().get_value("upstream_rho", m_rho_upstream);
  sim_env().params().get_value("rho_floor", m_rho_floor);
  // m_inj_length = m_damping_length / 2;
  m_inj_length = m_damping_length;

  // bool m_periodic[Conf::dim];

  // auto m_periodic = sim_env().params().template get_as<std::vector<bool>>(
  //     "periodic_boundary");
  // if (m_periodic.size() < Conf::dim) {
  //   m_periodic.resize(Conf::dim, false);
  // }
  // for (int i = 0; i < Conf::dim; i++) {
  //   m_periodic[i] = periodic[i];
  // }

  Logger::print_info("Boundary condition Bp is {}", m_Bp);
  Logger::print_info(
      "m_rho_upstream = {}; m_rho_floor = {}",
      m_rho_upstream, m_rho_floor
  );
  Logger::print_info(
      "2 * m_rho_floor - 2 * m_rho_upstream / upstream_n = {}",
      2.0f * m_rho_floor - 2.0f * m_rho_upstream / m_upstream_n
  );
  // m_prev_E1 = std::make_unique<multi_array_t>(
  //     extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  // m_prev_E2 = std::make_unique<multi_array_t>(
  //     extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  // m_prev_E3 = std::make_unique<multi_array_t>(
  //     extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  // m_prev_B1 = std::make_unique<multi_array_t>(
  //     extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  // m_prev_B2 = std::make_unique<multi_array_t>(
  //     extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  // m_prev_B3 = std::make_unique<multi_array_t>(
  //     extent(m_damping_length, m_grid.dims[1]), MemType::device_only);

  // m_prev_E1->assign_dev(0.0f);
  // m_prev_E2->assign_dev(0.0f);
  // m_prev_E3->assign_dev(0.0f);
  // m_prev_B1->assign_dev(0.0f);
  // m_prev_B2->assign_dev(0.0f);
  // m_prev_B3->assign_dev(0.0f);

  // m_prev_E.set_memtype(MemType::host_device);
  // m_prev_B.set_memtype(MemType::host_device);
  // m_prev_E.resize(3);
  // m_prev_B.resize(3);
  // m_prev_E[0] = m_prev_E1->dev_ptr();
  // m_prev_E[1] = m_prev_E2->dev_ptr();
  // m_prev_E[2] = m_prev_E3->dev_ptr();
  // m_prev_B[0] = m_prev_B1->dev_ptr();
  // m_prev_B[1] = m_prev_B2->dev_ptr();
  // m_prev_B[2] = m_prev_B3->dev_ptr();
  // m_prev_E.copy_to_device();
  // m_prev_B.copy_to_device();

  extent_t<Conf::dim> ext_inj(grid.reduced_dim(0), m_inj_length);
  m_dens_e1 = std::make_unique<multi_array_t>(ext_inj, MemType::device_only);
  m_dens_p1 = std::make_unique<multi_array_t>(ext_inj, MemType::device_only);
  m_dens_e2 = std::make_unique<multi_array_t>(ext_inj, MemType::device_only);
  m_dens_p2 = std::make_unique<multi_array_t>(ext_inj, MemType::device_only);

  // injector =
  //     sim_env().register_system<ptc_injector<Conf, exec_policy_gpu>>(grid);
}

template <typename Conf>
void
boundary_condition<Conf>::init() {
  sim_env().get_data("Edelta", E);
  sim_env().get_data("E0", E0);
  sim_env().get_data("Bdelta", B);
  sim_env().get_data("B0", B0);
  // sim_env().get_data("rand_states", &rand_states);
  sim_env().get_data("particles", ptc);
  sim_env().get_data("rng_states", rng_states);
  sim_env().get_data("Rho_e", rho_e);
  sim_env().get_data("Rho_p", rho_p);

  injector = std::make_unique<ptc_injector<Conf, exec_policy_gpu>>(m_grid);
}

template <typename Conf>
void
boundary_condition<Conf>::update(double dt, uint32_t step) {
  damp_fields();
  inject_plasma();
}

template <typename Conf>
void
boundary_condition<Conf>::damp_fields() {
  typedef typename Conf::idx_t idx_t;
  value_t Bp = m_Bp;
  value_t Bg = m_Bg;
  bool is_boundary[Conf::dim * 2];
  for (int i = 0; i < Conf::dim * 2; i++) {
    if (m_comm != nullptr) {
      is_boundary[i] = m_comm->domain_info().is_boundary[i];
    } else {
      is_boundary[i] = true;
    }
  }

  bool is_periodic[Conf::dim];
  for (int i = 0; i < Conf::dim; i++) {
    if (m_comm != nullptr) {
      is_periodic[i] = m_comm->domain_info().is_periodic[i];
      // Logger::print_info("m_comm NOT null; i = {}", i);
    } else {
      // Logger::print_info("m_comm null; i = {}", i);
      is_periodic[i] = false;
    }
  }

  // Logger::print_info("is_periodic = ({}, {})", is_periodic[0], is_periodic[1]);

  // bool is_periodic[Conf::dim];
  // for (int i = 0; i < Conf::dim; i++) {
  //     is_periodic[i] = m_periodic[i];
  // }

  if (!is_periodic[0]) {
      // Apply damping boundary condition on both X boundaries
      kernel_launch(
          [Bp, Bg, is_boundary] __device__(auto e, auto b, auto prev_e, auto prev_b,
                                           auto damping_length, auto damping_coef) {
            auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
            auto ext = grid.extent();
            // auto ext_damping = extent(damping_length, grid.dims[0]);
            for (auto n1 : grid_stride_range(0, grid.dims[1])) {
              // x = 0 boundary
              if (is_boundary[0] == true) {
                for (int i = 0; i < damping_length; i++) {
                  int n0 = i;
                  auto idx = idx_t(index_t<2>(n0, n1), ext);
                  value_t lambda =
                      1.0f - damping_coef * square((value_t)(damping_length - i) /
                                                 (damping_length - 1));
                  e[0][idx] *= lambda;
                  e[1][idx] *= lambda;
                  e[2][idx] *= lambda;
                  // b[0][idx] = lambda * (b[0][idx] + Bp) - Bp;
                  // b[0][idx] *= lambda;
                  b[1][idx] *= lambda;
                  b[2][idx] *= lambda;
                  // b[2][idx] = lambda * (b[2][idx] - Bp * Bg) + Bp * Bg;
                }
              }
              // x = x_max boundary
              if (is_boundary[1] == true) {
                for (int i = 0; i < damping_length; i++) {
                  int n0 = grid.dims[0] - damping_length + i;
                  auto idx = idx_t(index_t<2>(n0, n1), ext);
                  value_t lambda =
                      1.0f - damping_coef * square((value_t)i / (damping_length - 1));
                  e[0][idx] *= lambda;
                  e[1][idx] *= lambda;
                  e[2][idx] *= lambda;
                  // b[0][idx] = lambda * (b[0][idx] - Bp) + Bp;
                  // b[0][idx] *= lambda;
                  b[1][idx] *= lambda;
                  // b[2][idx] = lambda * (b[2][idx] - Bp * Bg) + Bp * Bg;
                  b[2][idx] *= lambda;
                }
              }
            }
          },
          E->get_ptrs(), B->get_ptrs(), m_prev_E.dev_ptr(), m_prev_B.dev_ptr(),
          m_damping_length, m_damping_coef);
  }
  GpuSafeCall(gpuDeviceSynchronize());
  GpuCheckError();
  
  if (!is_periodic[1]) {
      // Apply damping boundary condition on both Y boundaries
      kernel_launch(
          [Bp, Bg, is_boundary] __device__(auto e, auto b, auto prev_e, auto prev_b,
                                           auto damping_length, auto damping_coef) {
            auto &grid = dev_grid<Conf::dim, typename Conf::value_t>();
            auto ext = grid.extent();
            // auto ext_damping = extent(damping_length, grid.dims[0]);
            for (auto n0 : grid_stride_range(0, grid.dims[0])) {
              // y = -ymax boundary
              if (is_boundary[2] == true) {
                for (int i = 0; i < damping_length; i++) {
                  int n1 = i;
                  auto idx = idx_t(index_t<2>(n0, n1), ext);
                  value_t lambda =
                      1.0f - damping_coef * square((value_t)(damping_length - i) /
                                                 (damping_length - 1));
                  e[0][idx] *= lambda;
                  e[1][idx] *= lambda;
                  e[2][idx] *= lambda;
                  // b[0][idx] = lambda * (b[0][idx] + Bp) - Bp;
                  b[0][idx] *= lambda;
                  b[1][idx] *= lambda;
                  b[2][idx] *= lambda;
                  // b[2][idx] = lambda * (b[2][idx] - Bp * Bg) + Bp * Bg;
                }
              }
              // y = y_max boundary
              if (is_boundary[3] == true) {
                for (int i = 0; i < damping_length; i++) {
                  int n1 = grid.dims[1] - damping_length + i;
                  auto idx = idx_t(index_t<2>(n0, n1), ext);
                  value_t lambda =
                      1.0f - damping_coef * square((value_t)i / (damping_length - 1));
                  e[0][idx] *= lambda;
                  e[1][idx] *= lambda;
                  e[2][idx] *= lambda;
                  // b[0][idx] = lambda * (b[0][idx] - Bp) + Bp;
                  b[0][idx] *= lambda;
                  b[1][idx] *= lambda;
                  // b[2][idx] = lambda * (b[2][idx] - Bp * Bg) + Bp * Bg;
                  b[2][idx] *= lambda;
                }
              }
            }
          },
          E->get_ptrs(), B->get_ptrs(), m_prev_E.dev_ptr(), m_prev_B.dev_ptr(),
          m_damping_length, m_damping_coef);
  }
  GpuSafeCall(gpuDeviceSynchronize());
  GpuCheckError();
}

template <typename Conf>
void
boundary_condition<Conf>::inject_plasma() {
  // m_dens_e1->assign_dev(0.0f);
  // m_dens_p1->assign_dev(0.0f);
  // m_dens_e2->assign_dev(0.0f);
  // m_dens_p2->assign_dev(0.0f);
  bool is_boundary[Conf::dim * 2];
  for (int i = 0; i < Conf::dim * 2; i++) {
    if (m_comm != nullptr) {
      is_boundary[i] = m_comm->domain_info().is_boundary[i];
    } else {
      is_boundary[i] = true;
    }
  }

  bool is_periodic[Conf::dim];
  for (int i = 0; i < Conf::dim; i++) {
    if (m_comm != nullptr) {
      is_periodic[i] = m_comm->domain_info().is_periodic[i];
    } else {
      is_periodic[i] = false;
    }
  }

  auto inj_length = m_inj_length;
  value_t upstream_kT = m_upstream_kT;
  auto upstream_n = m_upstream_n;
  auto num = ptc->number();
  auto boost_beta = m_boost_beta;

  auto rho_e_ptr = rho_e->dev_ndptr();
  auto rho_p_ptr = rho_p->dev_ndptr();
  auto ext_inj = extent_t<2>(m_grid.reduced_dim(0), inj_length);
  auto inj_size = inj_length * m_grid.delta[1];
  auto rho_upstream = m_rho_upstream;
  auto rho_floor = m_rho_floor;

  // rho_upstream is the (mass) density from one species; upstream_n is # ppc per species
  // rho_floor is the density floor for one species (half the total density floor)
  auto weight_per_upstream_ptc = rho_upstream / (1.0f * upstream_n);
  injector->inject_pairs(
      [rho_e_ptr, rho_p_ptr, inj_size, is_periodic, is_boundary, rho_floor, weight_per_upstream_ptc] __device__(
          auto &pos, auto &grid, auto &ext) {
        auto x_global = grid.coord_global(pos, {0.5f, 0.5f, 0.0f});
        auto idx = Conf::idx(pos, ext);
        auto result = rho_p_ptr[idx] - rho_e_ptr[idx] <
            2.0f * rho_floor - 2.0f * weight_per_upstream_ptc;
        if (!is_periodic[0]) {
            // Apply density floor everywhere if not periodic in x
            return result;
        } else if ((x_global[1] >= grid.lower[1] + grid.sizes[1] - inj_size &&
             is_boundary[3]) ||
            (x_global[1] < grid.lower[1] + inj_size && is_boundary[2])) {
            // If periodic in x, just apply density floor near y boundaries
            // where plasma will probably get depleted.
            return result;
        } else {
            // This is when x is periodic and we're far from y boundaries.
            // Don't inject plasma.
            return false;
        }
      },
      [] __device__(auto &pos, auto &grid, auto &ext) { return 2; },
      [upstream_kT, boost_beta] __device__(auto &x_global,
                                           rand_state &state, PtcType type) {
        // vec_t<value_t, 3> u_d =
        // rng.maxwell_juttner_drifting<value_t>(upstream_kT, 0.995f);

        // auto p1 = u_d[0];
        // auto p2 = u_d[1];
        // auto p3 = u_d[2];
        // auto p = vec_t<value_t, 3>(p1, p2, p3);
        // // return lorentz_transform_momentum(p, {boost_beta, 0.0, 0.0});
        // value_t gamma = math::sqrt(1.0f + p.dot(p));
        // // return lorentz_transform_momentum(p, {boost_beta, 0.0, 0.0});
        // vec_t<value_t, 4> p_prime = lorentz_transform_vector(gamma, p,
        // {boost_beta, 0.0, 0.0}); return p_prime.template subset<1, 4>();
        return rng_maxwell_juttner_3d(state, upstream_kT);
        // auto p1 = rng.gaussian<value_t>(2.0f * upstream_kT);
        // auto p2 = rng.gaussian<value_t>(2.0f * upstream_kT);
        // auto p3 = rng.gaussian<value_t>(2.0f * upstream_kT);
        // // value_t gamma = math::sqrt(1.0f + p1*p1 + p2*p2 + p3*p3);
        // // value_t beta = p1 / gamma;
        // // return vec_t<value_t, 3>(beta / math::sqrt(1.0f - beta*beta),
        // 0.0f, 0.0f); return vec_t<value_t, 3>(p1, p2, p3);
      },
      // [upstream_n] __device__(auto &pos, auto &grid, auto &ext) {
      [weight_per_upstream_ptc] __device__(auto &x_global, PtcType type) { return weight_per_upstream_ptc; });

  // injector->inject_pairs(
  //     [rho_e_ptr, rho_p_ptr, inj_size, upstream_n, is_boundary] __device__(
  //         auto &pos, auto &grid, auto &ext) {
  //       auto x_global = grid.coord_global(pos, {0.5f, 0.5f, 0.0f});

  //       if ((x_global[1] >= grid.lower[1] + grid.sizes[1] - inj_size &&
  //            is_boundary[3]) ||
  //           (x_global[1] < grid.lower[1] + inj_size && is_boundary[2])) {
  //         auto idx = Conf::idx(pos, ext);
  //         return rho_p_ptr[idx] - rho_e_ptr[idx] < 2.0f - 3.0f / upstream_n;
  //       }
  //       return false;
  //     },
  //     [] __device__(auto &pos, auto &grid, auto &ext) { return 2; },
  //     [upstream_kT, boost_beta] __device__(auto &x_global,
  //                                          rand_state &state, PtcType type) {
  //       // vec_t<value_t, 3> u_d =
  //       // rng.maxwell_juttner_drifting<value_t>(upstream_kT, 0.995f);

  //       // auto p1 = u_d[0];
  //       // auto p2 = u_d[1];
  //       // auto p3 = u_d[2];
  //       // auto p = vec_t<value_t, 3>(p1, p2, p3);
  //       // // return lorentz_transform_momentum(p, {boost_beta, 0.0, 0.0});
  //       // value_t gamma = math::sqrt(1.0f + p.dot(p));
  //       // // return lorentz_transform_momentum(p, {boost_beta, 0.0, 0.0});
  //       // vec_t<value_t, 4> p_prime = lorentz_transform_vector(gamma, p,
  //       // {boost_beta, 0.0, 0.0}); return p_prime.template subset<1, 4>();
  //       return rng_maxwell_juttner_3d(state, upstream_kT);
  //       // auto p1 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //       // auto p2 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //       // auto p3 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //       // // value_t gamma = math::sqrt(1.0f + p1*p1 + p2*p2 + p3*p3);
  //       // // value_t beta = p1 / gamma;
  //       // // return vec_t<value_t, 3>(beta / math::sqrt(1.0f - beta*beta),
  //       // 0.0f, 0.0f); return vec_t<value_t, 3>(p1, p2, p3);
  //     },
  //     // [upstream_n] __device__(auto &pos, auto &grid, auto &ext) {
  //     [upstream_n] __device__(auto &x_global, PtcType type) { return 1.0 / upstream_n; });

  // buffer<int> offset(1, MemType::host_device);
  // offset[0] = num;
  // offset.copy_to_device();
  // auto ext_inj = m_dens_e1->extent();

  // policy::launch(
  //     [inj_length, upstream_kT, upstream_n, ext_inj] __device__(
  //         auto ptc, auto dens_e1, auto dens_p1, auto dens_e2, auto dens_p2,
  //         auto states, auto offset) {
  //       auto &grid = policy::grid();
  //       auto ext = grid.extent();
  //       rng_t rng(states);

  //       for (auto idx :
  //            grid_stride_range(Conf::begin(ext_inj), Conf::end(ext_inj))) {
  //         auto pos_inj = get_pos(idx, ext_inj);
  //         // First check lower boundary
  //         value_t dens = dens_e1[idx] + dens_p1[idx];
  //         if (dens < 2.0f - 2.0f / upstream_n) {
  //           auto n = atomic_add(offset, 2);
  //           index_t<2> pos = index_t<2>(pos_inj[0] + grid.guard[0],
  //                                       pos_inj[1] + grid.guard[1]);

  //           ptc.x1[n] = ptc.x1[n + 1] = rng.uniform<value_t>();
  //           ptc.x2[n] = ptc.x2[n + 1] = rng.uniform<value_t>();
  //           ptc.x3[n] = ptc.x3[n + 1] = rng.uniform<value_t>();
  //           auto p1 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //           auto p2 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //           auto p3 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //           ptc.p1[n] = p1;
  //           ptc.p2[n] = p2;
  //           ptc.p3[n] = p3;
  //           ptc.E[n] = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

  //           p1 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //           p2 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //           p3 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //           ptc.p1[n + 1] = p1;
  //           ptc.p2[n + 1] = p2;
  //           ptc.p3[n + 1] = p3;
  //           ptc.E[n + 1] = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

  //           ptc.weight[n] = ptc.weight[n + 1] = 1.0f / upstream_n;
  //           auto idx_p = Conf::idx(pos, ext);
  //           ptc.cell[n] = ptc.cell[n + 1] = idx_p.linear;
  //           ptc.flag[n] = set_ptc_type_flag(0, PtcType::electron);
  //           ptc.flag[n + 1] = set_ptc_type_flag(0, PtcType::positron);
  //         }

  //         // Then check upper boundary
  //         dens = dens_e2[idx] + dens_p2[idx];
  //         if (dens < 2.0f - 2.0f / upstream_n) {
  //           auto n = atomic_add(offset, 2);
  //           index_t<2> pos = index_t<2>(
  //               pos_inj[0] + grid.guard[0],
  //               pos_inj[1] + (grid.dims[1] - grid.guard[1] - inj_length));

  //           ptc.x1[n] = ptc.x1[n + 1] = rng.uniform<value_t>();
  //           ptc.x2[n] = ptc.x2[n + 1] = rng.uniform<value_t>();
  //           ptc.x3[n] = ptc.x3[n + 1] = rng.uniform<value_t>();
  //           auto p1 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //           auto p2 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //           auto p3 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //           ptc.p1[n] = p1;
  //           ptc.p2[n] = p2;
  //           ptc.p3[n] = p3;
  //           ptc.E[n] = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

  //           p1 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //           p2 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //           p3 = rng.gaussian<value_t>(2.0f * upstream_kT);
  //           ptc.p1[n + 1] = p1;
  //           ptc.p2[n + 1] = p2;
  //           ptc.p3[n + 1] = p3;
  //           ptc.E[n + 1] = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

  //           ptc.weight[n] = ptc.weight[n + 1] = 1.0f / upstream_n;
  //           auto idx_p = Conf::idx(pos, ext);
  //           ptc.cell[n] = ptc.cell[n + 1] = idx_p.linear;
  //           ptc.flag[n] = set_ptc_type_flag(0, PtcType::electron);
  //           ptc.flag[n + 1] = set_ptc_type_flag(0, PtcType::positron);
  //         }
  //       }
  //     },
  //     ptc, *m_dens_e1, *m_dens_p1, *m_dens_e2, *m_dens_p2, rng_states,
  //     offset);
  // policy::sync();

  // offset.copy_to_host();
  // ptc->set_num(offset[0]);
  // Logger::print_info("Injected {} particles", offset[0] - num);
}

template class boundary_condition<Config<2>>;

}  // namespace Aperture
