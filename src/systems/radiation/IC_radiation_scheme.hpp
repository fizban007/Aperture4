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

#pragma once

#include "core/cuda_control.h"
#include "core/particle_structs.h"
#include "core/random.h"
#include "data/fields.h"
#include "data/multi_array_data.hpp"
#include "data/phase_space.hpp"
#include "framework/environment.h"
#include "systems/grid.h"
#include "systems/inverse_compton.h"
#include "systems/physics/ic_scattering.hpp"
#include "systems/physics/spectra.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
struct IC_radiation_scheme {
  using value_t = typename Conf::value_t;

  const grid_t<Conf> &m_grid;
  ic_scatter_t m_ic_module;
  int m_num_bins = 256;
  value_t m_lim_lower = 1.0e-4;
  value_t m_lim_upper = 1.0e4;
  int m_downsample = 16;
  value_t m_IC_compactness = 0.0;
  mutable ndptr<value_t, Conf::dim> m_IC_loss;
  mutable ndptr<value_t, Conf::dim> m_IC_loss_total;
  mutable ndptr<value_t, 3> m_angle_dist_ptr;
  ndptr<value_t, Conf::dim + 1> m_spec_ptr;
  int m_ph_nth = 32;
  int m_ph_nphi = 64;
  extent_t<3> m_ext_ph_dist;

  // JM: Adding a flexible Lorentz factor
  // For particles with gamma < m_gmin_emit, particles are cooled according
  // to the Thomson IC formula and they do not emit photons (presumably because
  // we expect their photons to be all too low-energy to produce pairs).
  value_t m_gmin_emit = 2.0;

  // JM: Adding a flexible 4-velocity, u = gamma v, below which radiation is
  // artificially turned off.
  // Specifically, radiation is suppressed by the factor
  //     min(max((u - m_urad_cut)/(m_urad_cut - m_urad_cut/2), 0), 1)
  // which takes radiative cooling to zero between m_urad_cut and
  // m_urad_cut / 2.
  // Set m_urad_cut = 0.0 to suppress this feature.
  value_t m_urad_cut = 0.0;

  IC_radiation_scheme(const grid_t<Conf> &grid) : m_grid(grid) {}

  void init() {
    value_t emin = 1.0e-5;
    sim_env().params().get_value("IC_emin", emin);
    value_t ic_alpha = 1.25;
    sim_env().params().get_value("IC_alpha", ic_alpha);
    value_t bb_kT = 1e-5;
    sim_env().params().get_value("IC_bb_kT", bb_kT);
    value_t emono = 1e-5;
    sim_env().params().get_value("IC_monochrome_en", emono);
    std::string spec_type = "soft_power_law";
    sim_env().params().get_value("IC_bg_spectrum", spec_type);
    sim_env().params().get_value("IC_compactness", m_IC_compactness);

    // JM: Reading in these parameters if they've been specified
    if (sim_env().params().has("IC_Thomson_switch")) {
        sim_env().params().get_value("IC_Thomson_switch", m_gmin_emit);
    }
    if (sim_env().params().has("IC_urad_cut")) {
        sim_env().params().get_value("IC_urad_cut", m_urad_cut);
    }

    // If the config file specifies an IC gamma_rad, then we
    // use that to determine IC_compactness.
    if (sim_env().params().has("IC_gamma_rad") && sim_env().params().has("sigma")) {
      value_t IC_gamma_rad, sigma;
      sim_env().params().get_value("IC_gamma_rad", IC_gamma_rad);
      sim_env().params().get_value("sigma", sigma);
      m_IC_compactness = 0.3f * math::sqrt(sigma) / (square(IC_gamma_rad) * 4.0f);
    }

    // Configure the spectrum here and initialize the ic module
    // Spectra::black_body spec(bb_kT);

    auto ic = sim_env().register_system<inverse_compton_t>();
    // ic->compute_coefficients(spec, spec.emin(), spec.emax(), 1.5e24 /
    // ic_path);
    Logger::print_info("Using background spectrum {}", spec_type);
    value_t emean = 0.0;
    if (spec_type == "soft_power_law") {
      Spectra::power_law_soft spec(ic_alpha, emin, 1.0);
      ic->compute_coefficients(spec, spec.emin(), spec.emax());
      emean = spec.emean();
    } else if (spec_type == "black_body") {
      Spectra::black_body spec(bb_kT);
      ic->compute_coefficients(spec, spec.emin(), spec.emax());
      emean = spec.emean();
    // } else if (spec_type == "quasi_monochrome") {
    //   Spectra::very_narrow_gaussian spec(emono);
    //   ic->compute_coefficients(spec, spec.emin(), spec.emax());
    //   emean = spec.emean();
    } else {
      Logger::err("Spectrum type {} is not recognized!", spec_type);
      exit(1);
    }

    // If the config file specifies an IC opacity, then we
    // use that to determine IC_compactness.
    if (sim_env().params().has("IC_opacity")) {
      value_t IC_opacity;
      sim_env().params().get_value("IC_opacity", IC_opacity);
      m_IC_compactness = IC_opacity * emean;
    }

    Logger::print_info("IC compactness is {}", m_IC_compactness);

    m_ic_module = ic->get_ic_module();

    sim_env().params().get_value("ph_num_bins", m_num_bins);
    sim_env().params().get_value("ph_spec_lower", m_lim_lower);
    sim_env().params().get_value("ph_spec_upper", m_lim_upper);
    sim_env().params().get_value("momentum_downsample", m_downsample);

    // Always use logarithmic bins
    m_lim_lower = math::log(m_lim_lower);
    m_lim_upper = math::log(m_lim_upper);

    // Scalar field to tally up the low energy IC loss not accounted for using
    // photon spectrum
    auto IC_loss = sim_env().register_data<scalar_field<Conf>>(
        "IC_loss", this->m_grid, field_type::cell_centered,
        MemType::host_device);
    m_IC_loss = IC_loss->dev_ndptr();
    IC_loss->reset_after_output(true);
    auto IC_loss_total = sim_env().register_data<scalar_field<Conf>>(
        "IC_loss_total", this->m_grid, field_type::cell_centered,
        MemType::host_device);
    m_IC_loss_total = IC_loss_total->dev_ndptr();
    IC_loss_total->reset_after_output(true);

    extent_t<Conf::dim + 1> ext;
    for (int i = 0; i < Conf::dim; i++) {
      ext[i + 1] = m_grid.N[i] / m_downsample;
    }
    ext[0] = m_num_bins;
    ext.get_strides();

    auto photon_dist = sim_env().register_data<phase_space<Conf, 1>>(
        "IC_spectrum", m_grid, m_downsample, &m_num_bins, &m_lim_lower,
        &m_lim_upper, true, MemType::host_device);
    m_spec_ptr = photon_dist->data.dev_ndptr();
    photon_dist->reset_after_output(true);

    // IC angular distribution
    sim_env().params().get_value("ph_dist_n_th", m_ph_nth);
    sim_env().params().get_value("ph_dist_n_phi", m_ph_nphi);
    int ph_dist_interval = 10;
    sim_env().params().get_value("fld_output_interval", ph_dist_interval);
    // If no "ph_dist_interval" specified, we use fld_output_interval
    sim_env().params().get_value("ph_dist_interval", ph_dist_interval);

    auto photon_angular_dist =
        sim_env().register_data<multi_array_data<value_t, 3>>(
            "IC_dist", m_ph_nth, m_ph_nphi, m_num_bins);
    m_angle_dist_ptr = photon_angular_dist->dev_ndptr();
    photon_angular_dist->reset_after_output(true);
    photon_angular_dist->m_special_output_interval = ph_dist_interval;
    m_ext_ph_dist = photon_angular_dist->extent();
  }

  HOST_DEVICE size_t emit_photon(const Grid<Conf::dim, value_t> &grid,
                                 const extent_t<Conf::dim> &ext, ptc_ptrs &ptc,
                                 size_t tid, ph_ptrs &ph, size_t ph_num,
                                 unsigned long long int *ph_pos,
                                 rand_state &state, value_t dt) {
    using idx_t = default_idx_t<Conf::dim + 1>;
    value_t gamma = ptc.E[tid];
    value_t p1 = ptc.p1[tid];
    value_t p2 = ptc.p2[tid];
    value_t p3 = ptc.p3[tid];
    value_t p = math::sqrt(p1*p1 + p2*p2 + p3*p3);
    auto flag = ptc.flag[tid];

    value_t ic_cutoff_fac;

    ic_cutoff_fac = 1.0;
    if (m_urad_cut > 0.0) {
      ic_cutoff_fac = min(max(0.0, (p - 0.5 * m_urad_cut) / (0.5 * m_urad_cut)), 1.0);
    }

    if (check_flag(flag, PtcFlag::ignore_radiation)) {
      return 0;
    }

    // Regardless of how energy loss is accounted, we add this to total IC loss
    auto idx = Conf::idx(ptc.cell[tid], ext);
    value_t IC_loss_dt = ptc.weight[tid] * (4.0f / 3.0f) * m_IC_compactness * gamma * p * dt;
    atomic_add(&m_IC_loss_total[idx], IC_loss_dt);
    // printf("gamma is %f, IC loss is %f, weight is %f\n", gamma, IC_loss_dt, ptc.weight[tid]);
    // We don't care too much about the radiation from lowest energy particles.
    // Just cool them using usual formula
    if (gamma < m_gmin_emit) {
    // if (true) {
      // if (gamma < 1.0001) return 0;

      ptc.p1[tid] -=
          (4.0f / 3.0f) * ic_cutoff_fac * m_IC_compactness * gamma * ptc.p1[tid] * dt;
      ptc.p2[tid] -=
          (4.0f / 3.0f) * ic_cutoff_fac * m_IC_compactness * gamma * ptc.p2[tid] * dt;
      ptc.p3[tid] -=
          (4.0f / 3.0f) * ic_cutoff_fac * m_IC_compactness * gamma * ptc.p3[tid] * dt;

      ptc.E[tid] = math::sqrt(1.0f + square(ptc.p1[tid]) + square(ptc.p2[tid]) +
                              square(ptc.p3[tid]));

      // Since this part of cooling is not accounted for in the photon spectrum,
      // we need to accumulate it separately
      if (!check_flag(ptc.flag[tid], PtcFlag::exclude_from_spectrum)) {
        atomic_add(&m_IC_loss[idx],
                   ptc.weight[tid] * max(gamma - ptc.E[tid], 0.0));
      }

      return 0;
    }

    value_t p_i = math::sqrt(square(p1) + square(p2) + square(p3));
    auto pos = get_pos(idx, ext);

    auto ext_out = grid.extent_less() / m_downsample;
    auto ext_spec = extent_t<Conf::dim + 1>(m_num_bins, ext_out);
    index_t<Conf::dim + 1> pos_out(0, (pos - grid.guards()) / m_downsample);

    value_t lambda = ic_cutoff_fac * m_ic_module.ic_scatter_rate(gamma) * dt;
    // if (tid == 0) {
    //   printf("ic_prob is %f, gamma is %f\n", lambda, gamma);
    // }
    // int num_scattering = rng_poisson(state, lambda);
    int num_scattering = (rng_uniform<value_t>(state) < lambda ? 1 : 0);

    value_t e_ph = 0.0;
    for (int i = 0; i < num_scattering; i++) {
      e_ph = m_ic_module.gen_photon_e(gamma, state) * gamma;
      // printf("num_scattering is %d, e_ph is %f, gamma is %f\n", num_scattering, e_ph, gamma);
      // value_t e_ph = m_ic_module.e_mean * gamma * gamma;
      if (e_ph <= 0.0) {
        e_ph = 0.0;
        continue;
      }
      if (e_ph > gamma - 1.0001) {
        e_ph = math::abs(gamma - 1.0001);
        num_scattering = 0;
      }
      gamma -= math::abs(e_ph);
      // Account for the energy loss regardless of particle flag
      atomic_add(&m_IC_loss[idx], ptc.weight[tid] * max(e_ph, 0.0));

      auto flag = ptc.flag[tid];
      if (!check_flag(flag, PtcFlag::exclude_from_spectrum)) {
        value_t log_e_ph = clamp(math::log(max(e_ph, math::exp(m_lim_lower))),
                                 m_lim_lower, m_lim_upper);
        int bin = round((log_e_ph - m_lim_lower) / (m_lim_upper - m_lim_lower) *
                        (m_num_bins - 1));
        pos_out[0] = bin;
        atomic_add(&m_spec_ptr[idx_t(pos_out, ext_spec)], ptc.weight[tid]);

        // Simply deposit the photon direction along the particle direction,
        // without computing the 1/gamma cone
        value_t th = math::acos(p3 / p_i);
        value_t phi = math::atan2(p2, p1) + M_PI;
        int th_bin = round(th / M_PI * (m_ph_nth - 1));
        int phi_bin = round(phi * 0.5 / M_PI * (m_ph_nphi - 1));
        index_t<3> pos_ph_dist(th_bin, phi_bin, bin);
        atomic_add(
            &m_angle_dist_ptr[default_idx_t<3>(pos_ph_dist, m_ext_ph_dist)],
            ptc.weight[tid]);
      }
    }

    // value_t e_ph = m_ic_module.gen_photon_e(gamma, rng) * gamma;
    // gamma -= e_ph * num_scattering;

    value_t new_p = math::sqrt(max(square(gamma) - 1.0f, 0.0f));

    ptc.p1[tid] = p1 * new_p / p_i;
    ptc.p2[tid] = p2 * new_p / p_i;
    ptc.p3[tid] = p3 * new_p / p_i;
    ptc.E[tid] = gamma;

    // JM: I am enforcing photon censoring to the emission stage.
    // Do not create a photon if the energy is too small or if the pair-
    // production probability is too small.
    if (e_ph > 2.0 && m_ic_module.gg_scatter_rate(e_ph) * dt > 1e-5) {
      size_t offset = ph_num + atomic_add(ph_pos, 1);
      ph.x1[offset] = ptc.x1[tid];
      ph.x2[offset] = ptc.x2[tid];
      ph.x3[offset] = ptc.x3[tid];
      ph.cell[offset] = ptc.cell[tid];
      ph.weight[offset] = ptc.weight[tid];

      ph.p1[offset] = p1 * e_ph / p_i;
      ph.p2[offset] = p2 * e_ph / p_i;
      ph.p3[offset] = p3 * e_ph / p_i;
      // ph.E[offset] = Metric_KS::u0(m_a, r, th, u.template subset<1, 4>(), true);
      ph.E[offset] = e_ph;
      ph.flag[offset] = 0;

      return offset;
    }

    return 0;
  }

  HOST_DEVICE size_t produce_pair(const Grid<Conf::dim, value_t> &grid,
                                  const extent_t<Conf::dim> &ext, ph_ptrs &ph,
                                  size_t tid, ptc_ptrs &ptc, size_t ptc_num,
                                  unsigned long long int *ptc_pos,
                                  rand_state &state, value_t dt) {
    if (ph.cell[tid] == empty_cell) {
      return 0;
    }
    // Censor photons that have too low energy
    value_t eph = ph.E[tid];
    if (eph < 2.0f) {
      ph.cell[tid] = empty_cell;
      return 0;
    }

    // Compute the probability of producing a pair
    value_t gg_prob = m_ic_module.gg_scatter_rate(eph) * dt;
    if (gg_prob <= 1e-5) {
      // censor photons that are unlikely to produce pairs
      ph.cell[tid] = empty_cell;
      return 0;
    }
    if (rng_uniform<value_t>(state) >= gg_prob) {
      return 0;  // Does not produce a pair
    }

    // Produce a pair
    size_t offset = ptc_num + atomic_add(ptc_pos, 2);
    size_t offset_e = offset;
    size_t offset_p = offset + 1;

    ptc.x1[offset_e] = ptc.x1[offset_p] = ph.x1[tid];
    ptc.x2[offset_e] = ptc.x2[offset_p] = ph.x2[tid];
    ptc.x3[offset_e] = ptc.x3[offset_p] = ph.x3[tid];

    value_t p = math::sqrt(square(0.5 * eph) - 1.0);
    
    ptc.p1[offset_e] = ptc.p1[offset_p] = p * ph.p1[tid] / eph;
    ptc.p2[offset_e] = ptc.p2[offset_p] = p * ph.p2[tid] / eph;
    ptc.p3[offset_e] = ptc.p3[offset_p] = p * ph.p3[tid] / eph;
    ptc.E[offset_e] = ptc.E[offset_p] = 0.5 * eph;

#ifdef PARA_PERP
    ptc.work_para[offset_e] = ptc.work_para[offset_p] = 0.0;
    ptc.work_perp[offset_e] = ptc.work_perp[offset_p] = 0.0;
    ptc.work_curv[offset_e] = ptc.work_curv[offset_p] = 0.0;
#endif
// #ifndef NDEBUG
//     assert(ptc.cell[offset_e] == empty_cell);
//     assert(ptc.cell[offset_p] == empty_cell);
// #endif
    ptc.weight[offset_e] = ptc.weight[offset_p] = ph.weight[tid];
    ptc.cell[offset_e] = ptc.cell[offset_p] = ph.cell[tid];
    uint32_t base_flag = flag_or(PtcFlag::secondary);
    ptc.flag[offset_e] =
        set_ptc_type_flag(base_flag, PtcType::electron);
    ptc.flag[offset_p] =
        set_ptc_type_flag(base_flag, PtcType::positron);

    return offset;
  }
};

}  // namespace Aperture
