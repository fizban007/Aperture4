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

#ifndef _DOMAIN_COMM_ASYNC_H_
#define _DOMAIN_COMM_ASYNC_H_

#include "domain_comm.h"
#ifdef CUDA_ENABLED
#include <cuda_runtime_api.h>
#endif

namespace Aperture {

template <typename Conf>
class domain_comm_async : public domain_comm<Conf> {
 public:
  static std::string name() { return "domain_comm"; }

  domain_comm_async(sim_environment& env);
  virtual ~domain_comm_async();

  // virtual void send_guard_cells(vector_field<Conf>& field) const override;
  // virtual void send_guard_cells(scalar_field<Conf>& field) const override;
  virtual void send_guard_cells(typename Conf::multi_array_t& array,
                                const Grid<Conf::dim>& grid) const override;
  // virtual void send_add_guard_cells(vector_field<Conf>& field) const;
  // virtual void send_add_guard_cells(scalar_field<Conf>& field) const;
  virtual void send_add_guard_cells(typename Conf::multi_array_t& array,
                                    const Grid<Conf::dim>& grid) const override;

 protected:
  cudaStream_t m_copy_stream;

  void send_array_guard_cells_single_dir_async(
      typename Conf::multi_array_t& array, const Grid<Conf::dim>& grid, int dim,
      int dir) const;
  void send_add_array_guard_cells_single_dir_async(
      typename Conf::multi_array_t& array, const Grid<Conf::dim>& grid, int dim,
      int dir) const;
};

}  // namespace Aperture

#endif  // _DOMAIN_COMM_ASYNC_H_
