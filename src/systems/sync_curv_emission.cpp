/*
 * Copyright (c) 2022 Alex Chen.
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

#include "sync_curv_emission.h"
#include "framework/environment.h"

namespace Aperture {

namespace {

// This is the approximate form of synchrotron F(x) taken from Aharonian et al
// 2010
constexpr double Fx(double x) {
  double x13 = pow(x, 1.0/3.0);  // x13 is x^1/3
  double x23 = x13 * x13; // x23 is x^2/3
  double x43 = x23 * x23; // x43 is x^4/3

  return 2.15 * x13 * pow(1.0 + 3.06 * x, 1.0/6.0) * (1.0 + 0.884 * x23 + 0.471 * x43)
      / (1.0 + 1.64 * x23 + 0.974 * x43) * exp(-x);
}

}

sync_curv_emission_t::sync_curv_emission_t(MemType type) {
  m_nx = 2048;
  sim_env().params().get_value("n_sync_bins", m_nx);

  m_x_min = 1.0e-8;
  m_x_max = 1.0e3;

  // prepare the lookup table array
  m_Fx_lookup.resize(m_nx);
  m_Fx_cumulative.set_memtype(type);
  m_Fx_cumulative.resize(m_nx);

  // prepare the submodule
  m_sync.nx = m_nx;
  m_sync.logx_max = math::log(m_x_max);
  m_sync.logx_min = math::log(m_x_min);
  m_sync.dlogx = (m_sync.logx_max - m_sync.logx_min) / m_nx;
  if (type == MemType::host_only) {
    m_sync.ptr_lookup = m_Fx_cumulative.host_ptr();
  } else {
    m_sync.ptr_lookup = m_Fx_cumulative.dev_ptr();
  }

  compute_lookup_table();
}

sync_curv_emission_t::~sync_curv_emission_t() {}

void sync_curv_emission_t::compute_lookup_table() {
  value_t dlogx = m_sync.dlogx;
  for (int n = 0; n < m_nx; n++) {
    value_t logx = m_sync.logx_min + n * dlogx;
    value_t x = math::exp(logx);
    // Times an extra factor of x due to log spacing
    m_Fx_lookup[n] = Fx(x) * x;
    // m_Fx_lookup[n] = Fx(x);
    if (n == 0) {
      m_Fx_cumulative[n] = 0.0;
    } else {
      m_Fx_cumulative[n] = m_Fx_cumulative[n - 1] + m_Fx_lookup[n];
    }
  }

  for (int n = 0; n < m_nx; n++) {
    m_Fx_cumulative[n] /= m_Fx_cumulative[m_nx - 1];
  }

  m_Fx_cumulative.copy_to_device();
}

}
