#ifndef __CONSTANT_MEM_FUNC_H_
#define __CONSTANT_MEM_FUNC_H_

#include "core/grid.hpp"
#include <cstdint>

namespace Aperture {

struct params_struct;

void init_morton(const uint32_t m2dLUT[256], const uint32_t m3dLUT[256]);
// void init_dev_params(conparams_structams& params);
template <int Dim>
void init_dev_grid(const Grid<Dim>& grid);

void init_dev_charge_mass(const float charge[max_ptc_types],
                          const float mass[max_ptc_types]);
}  // namespace Aperture

#endif
