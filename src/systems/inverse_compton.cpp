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

#include "inverse_compton.h"
#include "core/math.hpp"
#include "framework/config.h"
#include "framework/environment.h"
#include "systems/physics/spectra.hpp"
#include "utils/logger.h"

namespace Aperture {

namespace {

constexpr double r_e_square = 7.91402e-26;

template <typename Scalar>
HOST_DEVICE Scalar
beta(Scalar gamma) {
  return 1.0 / sqrt(1.0 - 1.0 / square(gamma));
}

template <typename Scalar>
HOST_DEVICE Scalar
sigma_ic(Scalar x) {
  if (x < 1.0e-3) {
    return 1.0f - 2.0f * x + 26.0f * x * x / 5.0f;
  } else {
    Scalar l = std::log(1.0 + 2.0 * x);
    return 0.75 *
           ((1.0 + x) * (2.0 * x * (1.0 + x) / (1.0 + 2.0 * x) - l) / cube(x) +
            0.5 * l / x - (1.0 + 3.0 * x) / square(1.0 + 2.0 * x));
  }
}

template <typename Scalar>
HOST_DEVICE Scalar
sigma_gg(Scalar beta) {
  return (1.0 - square(beta)) *
         ((3.0 - beta * beta * beta * beta) * log((1.0 + beta) / (1.0 - beta)) -
          2.0 * beta * (2.0 - beta * beta));
}

template <typename Scalar>
HOST_DEVICE Scalar
x_ic(Scalar gamma, Scalar e, Scalar mu) {
  return gamma * e * (1.0 - mu * beta(gamma));
}

template <typename Scalar>
HOST_DEVICE Scalar
sigma_lab(Scalar q, Scalar ge) {
  return 2.0 * q * log(q) + (1.0 + 2.0 * q) * (1.0 - q) +
         0.5 * square(ge * q) * (1.0 - q) / (1.0 + q * ge);
}

template <typename Scalar>
HOST_DEVICE Scalar
sigma_rest(Scalar ep, Scalar e1p) {
  return (ep / e1p + e1p / ep - (1.0 - square(1.0 - 1.0 / e1p + 1.0 / ep)));
}

}  // namespace

inverse_compton::inverse_compton() {
  int n_gamma = 512;
  int n_ep = 512;
  constexpr value_t max_gamma = 1.0e12;

  sim_env().params().get_value("n_gamma", n_gamma);
  sim_env().params().get_value("n_ep", n_ep);

  m_min_ep = 1.0e-10;
  m_dgamma = math::log(max_gamma) / (n_gamma - 1.0);
  m_dep = 1.0 / (n_ep - 1.0);
  m_dlep = -math::log(m_min_ep) / (n_ep - 1.0);

  auto spec_ext = extent(n_ep, n_gamma);
  m_dNde.resize(spec_ext);
  m_dNde_thomson.resize(spec_ext);
}

inverse_compton::~inverse_compton() = default;

ic_scatter_t
inverse_compton::get_ic_module() {
  ic_scatter_t result;
  result.dNde = m_dNde.cref();
  result.dNde_thomson = m_dNde_thomson.cref();
#ifdef CUDA_ENABLED
  result.ic_rate = m_ic_rate.dev_ptr();
  result.gg_rate = m_gg_rate.dev_ptr();
#else
  result.ic_rate = m_ic_rate.host_ptr();
  result.gg_rate = m_gg_rate.host_ptr();
#endif

  result.min_ep = m_min_ep;
  result.dep = m_dep;
  result.dlep = m_dlep;
  result.dgamma = m_dgamma;
  return result;
}

template <typename Spectrum>
void
inverse_compton::compute_coefficients(const Spectrum& n_e, value_t emin,
                                      value_t emax, value_t n0) {
  // These are the parameters of the cross-section integration
  constexpr int N_mu = 100;
  constexpr int N_e = 800;
  auto spec_ext = m_dNde.extent();
  int n_ep = spec_ext[0];
  int n_gamma = spec_ext[1];

  // Compute the gammas and rates for IC scattering
  Logger::print_info("Pre-calculating the scattering rate");
  double dmu = 2.0 / (N_mu - 1.0);
  double de = (log(emax) - log(emin)) / (N_e - 1.0);

  for (uint32_t n = 0; n < m_ic_rate.size(); n++) {
    // double gamma = m_gammas[n];
    double gamma = math::exp(m_dgamma * n);
    // Logger::print_info("gamma is {}", gamma);
    double result = 0.0;
    for (int i_mu = 0; i_mu < N_mu; i_mu++) {
      double mu = i_mu * dmu - 1.0;
      for (int i_e = 0; i_e < N_e; i_e++) {
        double e = exp(log(emin) + i_e * de);
        double x = x_ic(gamma, e, mu);
        double sigma = sigma_ic(x);
        result += 0.5f * n_e(e) * sigma * (1.0f - beta(gamma) * mu) * e;
      }
    }
    m_ic_rate[n] = result * dmu * de * (r_e_square * n0) * 8.0 * M_PI / 3.0;
    if (n % 10 == 0)
      Logger::print_info(
          "IC rate at gamma {} is {}, result is {}, factor is {}", gamma,
          m_ic_rate[n], result,
          (dmu * de * r_e_square * n0) * 8.0 * M_PI / 3.0);
  }
  m_ic_rate.copy_to_device();

  Logger::print_info("Pre-calculating the gamma-gamma pair creation rate");
  for (uint32_t n = 0; n < m_gg_rate.size(); n++) {
    // double eph = m_gammas[n];
    double eph = math::exp(m_dgamma * n);
    if (eph < 2.0) {
      m_gg_rate[n] = 0.0;
    } else {
      double result = 0.0;
      for (int i_mu = 0; i_mu < N_mu; i_mu++) {
        double mu = i_mu * dmu - 1.0;
        for (int i_e = 0; i_e < N_e; i_e++) {
          double e = exp(log(emin) + i_e * de);
          double s = eph * e * (1.0 - mu) * 0.5;
          if (s <= 1.0) continue;
          double b = sqrt(1.0 - 1.0 / s);
          if (b == 1.0) continue;
          result += sigma_gg(b) * (1.0 - mu) * n_e(e) * e;
          // Logger::print_info("eph is {}, s is {}, b is {}, sigma is
          // {}", eph, s, b, sigma_gg(b));
        }
      }
      m_gg_rate[n] = 0.25 * result * dmu * de * M_PI * (n0 * r_e_square);
      // if (n != 0)
      //   m_gg_rate[n] /= m_gg_rate[0];
      if (n % 10 == 0)
        Logger::print_info("gg rate at gamma {} is {}", eph, m_gg_rate[n]);
    }
  }
  m_gg_rate.copy_to_device();

  Logger::print_info("Pre-calculating the lab-frame spectrum");
  for (uint32_t n = 0; n < n_gamma; n++) {
    // double gamma = m_gammas[n];
    double gamma = math::exp(m_dgamma * n);
    for (uint32_t i = 0; i < n_ep; i++) {
      double e1 = m_dep * i;
      double result = 0.0;
      for (uint32_t i_e = 0; i_e < N_e; i_e++) {
        double e = exp(log(emin) + i_e * de);
        double ne = n_e(e);
        // if (ne < 1.0e-8) continue;
        double ge = gamma * e * 4.0;
        double q = e1 / (ge * (1.0 - e1));
        if (e1 < ge / (1.0 + ge) && e1 > e / gamma)
          // result += n_e(e) * sigma_lab(q, ge) / gamma;
          result += ne * sigma_lab(q, ge) / gamma;
      }
      m_dNde(i, n) = result * de;
    }
    for (uint32_t i = 1; i < n_ep; i++) {
      m_dNde(i, n) += m_dNde(i - 1, n);
    }
    for (uint32_t i = 0; i < n_ep; i++) {
      m_dNde(i, n) /= m_dNde(n_ep - 1, n);
    }
  }

  m_dNde.copy_to_device();
  Logger::print_info("Finished copying m_dNde to device");

  for (uint32_t n = 0; n < n_gamma; n++) {
    // double gamma = m_gammas[n];
    double gamma = math::exp(m_dgamma * n);
    for (uint32_t i = 0; i < n_ep; i++) {
      // double e1 = m_log_ep[i];
      double e1 = m_min_ep * math::exp(m_dlep * i);
      double result = 0.0;
      for (uint32_t i_e = 0; i_e < N_e; i_e++) {
        double e = exp(log(emin) + i_e * de);
        double ne = n_e(e);
        // if (ne < 1.0e-8) continue;
        double ge = gamma * e * 4.0;
        double q = e1 / (ge * (1.0 - e1));
        if (e1 < ge / (1.0 + ge) && e1 > e / gamma)
          result += ne * sigma_lab(q, ge) / gamma;
      }
      m_dNde_thomson(i, n) = result * de * e1;
    }
    for (uint32_t i = 1; i < n_ep; i++) {
      m_dNde_thomson(i, n) += m_dNde_thomson(i - 1, n);
    }
    for (uint32_t i = 0; i < n_ep; i++) {
      m_dNde_thomson(i, n) /= m_dNde_thomson(n_ep - 1, n);
    }
  }

  m_dNde_thomson.copy_to_device();
  Logger::print_info("Finished copying m_dNde_thomson to device");
}

template void inverse_compton::compute_coefficients<Spectra::power_law_hard>(
    const Spectra::power_law_hard& n_e, value_t emin, value_t emax, value_t n0);
template void inverse_compton::compute_coefficients<Spectra::power_law_soft>(
    const Spectra::power_law_soft& n_e, value_t emin, value_t emax, value_t n0);
template void inverse_compton::compute_coefficients<Spectra::black_body>(
    const Spectra::black_body& n_e, value_t emin, value_t emax, value_t n0);
template void inverse_compton::compute_coefficients<Spectra::mono_energetic>(
    const Spectra::mono_energetic& n_e, value_t emin, value_t emax, value_t n0);
template void inverse_compton::compute_coefficients<Spectra::broken_power_law>(
    const Spectra::broken_power_law& n_e, value_t emin, value_t emax,
    value_t n0);

}  // namespace Aperture
