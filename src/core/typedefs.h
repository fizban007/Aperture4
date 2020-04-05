#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_

#include <cstdint>

namespace Aperture {

#ifndef USE_DOUBLE
typedef float Scalar;
typedef float Mom_t;
typedef float Pos_t;
#else
typedef double Scalar;
typedef double Mom_t;
typedef float Pos_t;
#endif

typedef std::size_t Index_t;

}  // namespace Aperture

#endif  // _TYPEDEFS_H_
