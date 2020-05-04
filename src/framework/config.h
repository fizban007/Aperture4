#ifndef __CONFIG_H_
#define __CONFIG_H_

#include "core/enum_types.h"
#include "core/multi_array.hpp"
#include "core/ndptr.hpp"
#include "core/particles.h"
#include "core/typedefs_and_constants.h"
#include "utils/index.hpp"
#include "utils/interpolation.hpp"
#include "utils/logger.h"

namespace Aperture {

class sim_environment;

////////////////////////////////////////////////////////////////////////////////
///  The `Config` class basically maintains all the compile time type
///  configurations of all modules. Instead of individually specifying default
///  types, indexing schemes, and dimension of the grid, a module can simply use
///  `Config` as the general template parameter. `Config` has no members, no
///  member functions, and only has static `typedef`s and static functions.
////////////////////////////////////////////////////////////////////////////////
template <int Dim, typename FloatT = Scalar,
          template <int> typename Idx_t = idx_col_major_t>
class Config {
 public:
  static constexpr int dim = Dim;
  static constexpr bool is_zorder =
      std::is_same<Idx_t<Dim>, idx_zorder_t<Dim>>::value;

  typedef FloatT value_t;
  typedef Idx_t<Dim> idx_t;
  typedef multi_array<FloatT, Dim, Idx_t<Dim>> multi_array_t;
  typedef ndptr<FloatT, Dim, Idx_t<Dim>> ndptr_t;
  typedef ndptr_const<FloatT, Dim, Idx_t<Dim>> ndptr_const_t;
  typedef buffer<FloatT> buffer_t;
  typedef bspline<3> spline_t;

  template <typename... Args>
  static multi_array_t make_multi_array(Args... args) {
    return multi_array_t(args...);
  }

  static multi_array_t make_multi_array(const extent_t<Dim>& ext,
                                        MemType model = default_mem_type) {
    return multi_array_t(ext, model);
  }

  Config() {}
  Config(const Config& other) = delete;
  Config(Config&& other) = delete;
  Config& operator=(const Config& other) = delete;
  Config& operator=(Config&& other) = delete;
};

}  // namespace Aperture

#endif
