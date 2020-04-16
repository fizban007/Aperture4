#ifndef _TYPEDEFS_N_CONSTANTS_H_
#define _TYPEDEFS_N_CONSTANTS_H_

#include <limits>
#include <cstdint>
#include "enum_types.h"

namespace Aperture {

#ifdef CUDA_ENABLED
constexpr MemType default_mem_type = MemType::host_device;
#else
constexpr MemType default_mem_type = MemType::host_only;
#endif
constexpr uint32_t empty_cell = std::numeric_limits<uint32_t>::max();
constexpr float eps_float = 1.0e-8f;
constexpr double eps_double = 1.0e-12;

#ifndef USE_DOUBLE
typedef float Scalar;
typedef float Mom_t;
typedef float Pos_t;
constexpr float TINY = eps_float;
#else
typedef double Scalar;
typedef double Mom_t;
typedef double Pos_t;
constexpr double TINY = eps_double;
#endif

}  // namespace Aperture

#endif  // _TYPEDEFS_N_CONSTANTS_H_
