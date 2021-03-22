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

#ifndef _PTC_INJECTOR_NEW_H_
#define _PTC_INJECTOR_NEW_H_

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
class ptc_injector;
//     : public system_t {
//  public:
//   using value_t = typename Conf::value_t;
//   static std::string name() { return "ptc_injector"; }

//   ptc_injector(const grid_t<Conf>& grid) : m_grid(grid) {}
//   ~ptc_injector() {}

//   void init() override;

//   template <typename FCriteria, typename FDist>
//   void inject(const FCriteria& fc, const FDist& fd);

//  private:
//   const grid_t<Conf>& m_grid;
//   particle_data_t* ptc;

//   multi_array<int, Conf::dim> m_num_per_cell, m_cum_num_per_cell;
// };

}  // namespace Aperture

#endif  // _PTC_INJECTOR_NEW_H_
