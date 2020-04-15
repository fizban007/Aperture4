#ifndef __MULTI_ARRAY_H_
#define __MULTI_ARRAY_H_

#include "ndptr.hpp"
#include "typedefs_and_constants.h"
#include "utils/buffer.h"
#include "utils/index.hpp"
#include "utils/range.hpp"
#include "utils/vec.hpp"
#include <exception>
#include <type_traits>

namespace Aperture {

template <typename T, int Rank,
          // MemType Model = default_mem_type,
          typename Index_t = default_index_t<Rank>>
class multi_array : public buffer_t<T> {
 private:
  extent_t<Rank> m_ext;

 public:
  typedef buffer_t<T> base_type;
  typedef multi_array<T, Rank, Index_t> self_type;
  typedef extent_t<Rank> extent_type;
  typedef Index_t index_type;
  typedef ndptr<T, Rank, Index_t> ptr_type;
  typedef ndptr_const<T, Rank, Index_t> const_ptr_type;
  typedef T value_type;

  multi_array() {}

  template <typename... Args>
  multi_array(Args... args)
      : m_ext(args...), base_type(extent_type(args...).size()) {
    check_dimension();
  }

  multi_array(const extent_t<Rank>& extent,
              MemType model = default_mem_type)
      : m_ext(extent), base_type(extent.size(), model) {
    check_dimension();
  }

  // Disallow copy
  multi_array(const self_type& other) = delete;

  multi_array(self_type&& other)
      : m_ext(other.m_ext), base_type(std::move(other)) {
    other.m_ext = extent_type{};
  }

  ~multi_array() {}

  void assign(const T& value) { base_type::assign(value); }

  void copy_from(const self_type& other) {
    base_type::copy_from(other);
  }

  void resize(const extent_t<Rank>& ext) {
    m_ext = ext;
    base_type::resize(ext.size());
  }

  template <typename... Args>
  void resize(Args... args) {
    resize(extent(args...));
  }

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
    base_type::operator=(std::move(other));
    m_ext = other.m_ext;
    other.m_ext = extent_type{};
    return *this;
  }

  using base_type::operator[];

  // template <MemType M = Model>
  // inline std::enable_if_t<M != MemType::device_only, T>
  inline T
  operator[](const Index_t& idx) const {
    // Logger::print_info("in operator [], typeof idx is {}", typeid(idx).name());
    return this->m_data_h[idx.linear];
  }

  // template <MemType M = Model>
  // inline std::enable_if_t<M != MemType::device_only, T&>
  inline T&
  operator[](const Index_t& idx) {
    return this->m_data_h[idx.linear];
  }

  template <typename... Args>
  // inline std::enable_if_t<M != MemType::device_only, T>
  inline T
  operator()(Args... args) const {
    auto idx = get_idx(args...);
    return this->m_data_h[idx.linear];
  }

  template <typename... Args>
  // inline std::enable_if_t<M != MemType::device_only, T&>
  // template <typename... Args>
  inline T&
  operator()(Args... args) {
    auto idx = get_idx(args...);
    return this->m_data_h[idx.linear];
  }

  template <typename... Args>
  inline Index_t get_idx(Args... args) {
    auto idx = Index_t(index_t<Rank>(args...), m_ext);
    return idx;
  }

  inline Index_t idx_at(uint64_t n) const {
    return Index_t(n, m_ext);
  }

  inline Index_t idx_at(index_t<Rank> pos) const {
    return Index_t(pos, m_ext);
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

template <typename T,
          typename... Args>
auto
make_multi_array(Args... args) {
  return multi_array<T, sizeof...(Args),
                     default_index_t<sizeof...(Args)>>(args...);
}

template <typename T,
          template <int> class Index_t = default_index_t, int Rank>
auto
make_multi_array(const extent_t<Rank>& ext, MemType model) {
  return multi_array<T, Rank, Index_t<Rank>>(ext, model);
}

}  // namespace Aperture

#endif  // __MULTI_ARRAY_H_
