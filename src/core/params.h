#ifndef __PARAMS_H_
#define __PARAMS_H_

#include "core/enum_types.h"
#include "core/typedefs_and_constants.h"
#include <string>
#include "visit_struct/visit_struct_intrusive.hpp"

namespace Aperture {

/////////////////////////////////////////////////////////////////////////
///  This is the standard simulation parameters class. This class will
///  be maintained in the config and be passed around as reference to
///  determine how the simulation will unfold.
/////////////////////////////////////////////////////////////////////////
struct params_struct {
  BEGIN_VISITABLES(params_struct);
  VISITABLE_INIT(float, dt, 0.1);

  VISITABLE_INIT(uint64_t, max_steps, 10000); ///< Total number of timesteps
  VISITABLE_INIT(uint32_t, data_interval, 100); ///< How many steps between data outputs
  END_VISITABLES;
  // // Grid parameters
  // uint32_t N[3] = {1, 1, 1};
  // uint32_t guard[3] = {0, 0, 0};
  // uint32_t skirt[3] = {0, 0, 0};
  // float lower[3] = {0.0, 0.0, 0.0};
  // float size[3] = {1.0, 1.0, 1.0};
  // uint32_t tile_size[3] = {1, 1, 1};
  // uint32_t nodes[3] = {1, 1, 1};
};

params_struct parse_config(const std::string& filename);

void parse_config(const std::string& filename, params_struct& params);

}  // namespace Aperture

#endif
