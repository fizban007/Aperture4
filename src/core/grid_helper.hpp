#ifndef __GRID_HELPER_H_
#define __GRID_HELPER_H_

#include "core/cuda_control.h"
#include "core/grid.hpp"
#include "utils/vec.hpp"
#include "utils/index.hpp"

namespace Aperture {

template <typename Conf>
HD_INLINE typename Conf::idx_t grid_get_idx(Grid<Conf::dim>& grid, uint32_t cell) {
  return Conf::idx(cell, grid.extent());
}

template <typename Conf>
HD_INLINE index_t<Conf::dim> grid_get_pos(Grid<Conf::dim>& grid, uint32_t cell) {
  auto idx = Conf::idx(cell, grid.extent());
  return idx.get_pos();
}

}

#endif // __GRID_HELPER_H_
