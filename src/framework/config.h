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
///  The Config class basically maintains all the compile time type
///  configurations of all modules. Instead of individually specifying default
///  types, indexing schemes, and dimension of the grid, a module can simply use
///  Config as the general template parameter. Config has no members, no
///  member functions, and only has static typedefs and static functions.
///
///  \tparam Dim              Number of dimensions of the simulation
///  \tparam FloatT           The floating point type, `float` or `double`
///  \tparam InterpOrder      Interpolation order, 0, 1, 2, or 3
///  \tparam Idx_t            Indexing scheme for multiarrays
////////////////////////////////////////////////////////////////////////////////
template <int Dim, typename FloatT = Scalar,
          int InterpOrder = default_interp_order,
          template <int> typename Idx_t = default_idx_t>
class Config {
 public:
  static constexpr int dim = Dim;  //!< Access the dimension of the simulation
  static constexpr bool is_zorder =
      std::is_same<Idx_t<Dim>,
                   idx_zorder_t<Dim>>::value;  //!< Whether this is zorder

  using value_t = FloatT;    //!< The floating point type
  using idx_t = Idx_t<Dim>;  //!< The indexing type
  using coord_t = vec_t<FloatT, Dim>; //!< The floating point coordinate type
  using multi_array_t =
      multi_array<FloatT, Dim, Idx_t<Dim>>;  //!< The multi_array type
  using ndptr_t = ndptr<FloatT, Dim, Idx_t<Dim>>;  //!< The ndptr type
  using ndptr_const_t =
      ndptr_const<FloatT, Dim, Idx_t<Dim>>;  //!< The const ndptr type
  using buffer_t = buffer<FloatT>;           //!< The buffer type
  using spline_t = bspline<InterpOrder>;     //!< The interpolation b-spline type

  /// Construct and return a multi_array.
  /**
   * This is a helper function to construct a multi_array directly without using
   * the typedef. This version takes an extent, and can specify where the array
   * is located in memory:
   *
   *     auto array = Conf::make_multi_array(extent(32, 32))
   *     // The first argument can be abbreviated as an initializer list
   *     auto array = Conf::make_multi_array({32, 32}, MemType::host_device);
   *
   * \param ext   The extent of the resulting multi_array
   * \param type  Location of memory allocation
   * \return The constructed multi_array, by value.
   */
  static multi_array_t make_multi_array(const extent_t<Dim>& ext,
                                        MemType type = default_mem_type) {
    return multi_array_t(ext, type);
  }

  /// Make an idx object from a linear position and an extent.
  /**
   * \param n    The linear index in memory
   * \param ext  The n-dimensional extent
   */
  static HD_INLINE idx_t idx(size_t n, const extent_t<Dim>& ext) {
    return idx_t(n, ext);
  }

  /// Make an idx object from an n-dimensional position and an extent.
  /**
   * \param pos  The n-dimensional position
   * \param ext  The n-dimensional extent
   */
  static HD_INLINE idx_t idx(const index_t<Dim>& pos,
                             const extent_t<Dim>& ext) {
    return idx_t(pos, ext);
  }

  /// Make an idx object with given extent and 0 linear position, equivalent to
  /// calling `idx(0, ext)`.
  static HD_INLINE idx_t begin(const extent_t<Dim>& ext) {
    return idx_t(0, ext);
  }

  /// Make an idx_t with given extent and linear position at the end of the
  /// array, equivalent to calling `idx(ext.size(), ext)`.
  static HD_INLINE idx_t end(const extent_t<Dim>& ext) {
    return idx_t(ext.size(), ext);
  }

  static constexpr value_t value(float x) { return (value_t)x; }
  static constexpr value_t value(double x) { return (value_t)x; }

  Config() {}
  Config(const Config& other) = delete;
  Config(Config&& other) = delete;
  Config& operator=(const Config& other) = delete;
  Config& operator=(Config&& other) = delete;
};

// Define a macro to help instantiate classes with config
#define INSTANTIATE_WITH_CONFIG(class_name) \
  template class class_name<Config<1>>;     \
  template class class_name<Config<2>>;     \
  template class class_name<Config<3>>

}  // namespace Aperture

#endif
