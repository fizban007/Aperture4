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

#ifndef _INVERSE_COMPTON_H_
#define _INVERSE_COMPTON_H_

#include "core/multi_array.hpp"
#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/physics/ic_scattering.hpp"

namespace Aperture {

class inverse_compton_t : public system_t {
 public:
  using value_t = Scalar;
  static std::string name() { return "inverse_compton"; }

  inverse_compton_t();
  ~inverse_compton_t();

  template <typename Spectrum>
  void compute_coefficients(const Spectrum& n_e, value_t emin, value_t emax);

  ic_scatter_t get_ic_module();

  void init() {}

  const buffer<value_t>& ic_rate() const { return m_ic_rate; }
  const buffer<value_t>& gg_rate() const { return m_gg_rate; }
  const multi_array<value_t, 2, idx_col_major_t<2>>& dNde() const {
    return m_dNde;
  }
  const multi_array<value_t, 2, idx_col_major_t<2>>& dNde_thomson() const {
    return m_dNde_thomson;
  }
  value_t min_ep() const { return m_min_ep; }
  value_t dgamma() const { return m_dgamma; }
  value_t dep() const { return m_dep; }
  value_t dlep() const { return m_dlep; }

 private:
  multi_array<value_t, 2, idx_col_major_t<2>> m_dNde;
  multi_array<value_t, 2, idx_col_major_t<2>> m_dNde_thomson;
  buffer<value_t> m_ic_rate, m_gg_rate;
  value_t m_min_ep, m_dgamma, m_dep, m_dlep, m_e_mean;
  value_t m_ic_compactness = 0.01;
};

}  // namespace Aperture

#endif  // _INVERSE_COMPTON_H_
