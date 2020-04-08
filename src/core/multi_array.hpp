#ifndef __MULTI_ARRAY_H_
#define __MULTI_ARRAY_H_

#include "typedefs_and_constants.h"
#include "ndptr.hpp"
#include "utils/buffer.h"
#include "utils/index.hpp"
#include "utils/range.hpp"
#include "utils/vec.hpp"
#include <exception>
#include <type_traits>

namespace Aperture {

template <typename T, int Rank,
          MemoryModel Model = default_memory_model,
          typename Index_t = default_index_t<Rank>>
class multi_array : public buffer_t<T, Model> {
 private:
  extent_t<Rank> m_ext;

 public:
  typedef buffer_t<T, Model> base_type;
  typedef multi_array<T, Rank, Model, Index_t> self_type;
  typedef extent_t<Rank> extent_type;
  typedef Index_t index_type;
  typedef ndptr<T, Rank, Index_t> ptr_type;
  typedef ndptr_const<T, Rank, Index_t> const_ptr_type;

  multi_array() {}

  template <typename... Args>
  multi_array(Args... args)
      : m_ext(args...), base_type(extent_type(args...).size()) {
    check_dimension();
  }

  multi_array(const extent_t<Rank>& extent)
      : m_ext(extent), base_type(extent.size()) {
    check_dimension();
  }

  // Disallow copy
  multi_array(const self_type& other) = delete;

  multi_array(self_type&& other)
      : m_ext(other.m_ext), base_type(std::move(other)) {
    other.m_ext = extent_type{};
  }

  ~multi_array() {}

  void check_dimension() {
    if (std::is_same<Index_t, idx_zorder_t<Rank>>::value) {
      if (not_power_of_two(m_ext)) {
        throw std::range_error(
            "One of the dimensions is not a power of 2, can't use "
            "zorder "
            "indexing!");
      }
    }
  }

  self_type& operator=(const self_type& other) = delete;

  self_type& operator=(self_type&& other) {
    this->operator=(other);
    m_ext = other.m_ext;
    other.m_ext = extent_type{};
    return *this;
  }

  using base_type::operator[];

  inline T operator[](const Index_t& idx) const {
    return this->data()[idx.linear];
  }

  inline T& operator[](const Index_t& idx) {
    return this->data()[idx.linear];
  }

  template <typename... Args>
  inline T operator()(Args... args) const {
    auto idx = get_idx(args...);
    return this->data()[idx.linear];
  }

  template <typename... Args>
  inline T& operator()(Args... args) {
    auto idx = get_idx(args...);
    return this->data()[idx.linear];
  }

  template <typename... Args>
  inline Index_t get_idx(Args... args) {
    auto idx = Index_t(index_t<Rank>(args...), m_ext);
    return idx;
  }

  inline Index_t idx_at(uint64_t idx) const {
    return Index_t(idx, m_ext);
  }

  inline ptr_type get_ptr() { return ptr_type(this->m_data_d); }

  inline const_ptr_type get_const_ptr() const {
    return const_ptr_type(this->m_data_d);
  }

  extent_t<Rank> extent() const { return m_ext; }

  inline range_proxy<Index_t> indices() const {
    return range(idx_at(0), idx_at(this->m_size));
  }
};

template <typename T, MemoryModel Model = default_memory_model,
          typename... Args>
auto
make_multi_array(Args... args) {
  return multi_array<T, sizeof...(Args), Model,
                     default_index_t<sizeof...(Args)>>(args...);
}

template <typename T,
          MemoryModel Model = default_memory_model,
          template <int> class Index_t = default_index_t,
          int Rank>
auto
make_multi_array(const extent_t<Rank>& ext) {
  return multi_array<T, Rank, Model, Index_t<Rank>>(ext);
}

}  // namespace Aperture

#endif  // __MULTI_ARRAY_H_
