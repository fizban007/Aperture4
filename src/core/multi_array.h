#ifndef __MULTI_ARRAY_H_
#define __MULTI_ARRAY_H_

#include "typedefs_and_constants.h"
#include "buffer.h"
#include "utils/index.h"
#include "utils/ndptr.h"
#include "utils/vec3.h"
#include <cstdlib>

namespace Aperture {

/// The multi_array class is a unified interface for 1D, 2D and 3D
/// arrays with proper index access and memory management. The most
/// benefit is indexing convenience. One-liners are just implemented
/// in the definition. Other functions are implemented in the
/// implementation header file.
template <typename T, MemoryModel Model = default_memory_model,
          typename Index_t = idx_col_major_t<>>
class multi_array : public buffer_t<T, Model> {
 private:
  Extent m_ext;

 public:
  typedef buffer_t<T, Model> base_type;
  typedef multi_array<T> self_type;
  typedef Index_t idx_type;
  typedef ndptr<T, Index_t> ptr_type;
  typedef ndptr_const<T, Index_t> const_ptr_type;

  /// Default constructor, initializes sizes to zero and data
  /// pointers to `nullptr`.
  multi_array();

  /// Main constructor, initializes with given width, height, and
  /// depth of the array. Allocate memory in the initialization.
  explicit multi_array(uint32_t width, uint32_t height = 1, uint32_t depth = 1);

  /// Alternative main constructor, takes in an @ref Extent object and
  /// initializes an array of the corresponding extent.
  explicit multi_array(const Extent& extent);

  /// Disallow copy constructor.
  multi_array(const self_type& other) = delete;

  /// Standard move constructor. The object `other` will become empty
  /// after the move.
  multi_array(self_type&& other) : base_type(std::move(other)) {
    m_ext = other.m_ext;
    other.m_ext = Extent(0, 0, 0);
  };

  /// Destructor. Delete the member data array.
  ~multi_array();

  /// Assignment operators for copying
  self_type& operator=(const self_type& other) = delete;

  /// Move assignment operator
  self_type& operator=(self_type&& other) {
    this->operator=(other);
    m_ext = other.m_ext;
    other.m_ext = Extent(0, 0, 0);
    return *this;
  }

  /// Use the base operator[] for simple uint32_t indexing.
  using base_type::operator[];

  /// Vector indexing operator using an @ref Index object, read only
  inline T operator[](const Index& index) const {
    return this->data()[get_idx(index).key];
  }

  /// Vector indexing operator using an @ref Index object, read and
  /// write
  inline T& operator[](const Index& index) {
    return this->data()[get_idx(index).key];
  }

  /// Vector indexing operator using an @ref Index_t object, read only
  inline T operator[](const idx_type& idx) const {
    return this->data()[idx.key];
  }

  /// Vector indexing operator using an @ref Index_t object, read and
  /// write
  inline T& operator[](const idx_type& idx) {
    return this->data()[idx.key];
  }

  /// Vector indexing operator using 3 indices, read only
  inline T operator()(uint32_t x0) const {
    return this->data()[get_idx(x0).key];
  }

  /// Vector indexing operator using 3 indices, read and
  /// write
  inline T& operator()(uint32_t x0) {
    return this->data()[get_idx(x0).key];
  }

  /// Vector indexing operator using 3 indices, read only
  inline T operator()(uint32_t x0, uint32_t x1) const {
    return this->data()[get_idx(x0, x1).key];
  }

  /// Vector indexing operator using 3 indices, read and
  /// write
  inline T& operator()(uint32_t x0, uint32_t x1) {
    return this->data()[get_idx(x0, x1).key];
  }

  /// Vector indexing operator using 3 indices, read only
  inline T operator()(uint32_t x0, uint32_t x1, uint32_t x2) const {
    return this->data()[get_idx(x0, x1, x2).key];
  }

  /// Vector indexing operator using 3 indices, read and
  /// write
  inline T& operator()(uint32_t x0, uint32_t x1, uint32_t x2) {
    return this->data()[get_idx(x0, x1, x2).key];
  }

  inline idx_type get_idx(uint32_t x0) const {
    return idx_type(x0, m_ext);
  }

  inline idx_type get_idx(uint32_t x0, uint32_t x1) const {
    return idx_type(x0, x1, m_ext);
  }

  inline idx_type get_idx(uint32_t x0, uint32_t x1, uint32_t x2) const {
    return idx_type(x0, x1, x2, m_ext);
  }

  inline idx_type get_idx(const Index& index) const {
    return idx_type(index.x, index.y, index.z, m_ext);
  }

  inline ptr_type get_ptr() { return ptr_type(this->m_data_d); }

  inline const_ptr_type get_const_ptr() const {
    return const_ptr_type(this->m_data_d);
  }

  inline const Extent& extent() const { return m_ext; }
};

}  // namespace Aperture

#include "multi_array_impl.hpp"

#endif  // __MULTI_ARRAY_H_
