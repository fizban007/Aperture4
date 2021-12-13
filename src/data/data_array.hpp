/*
 * Copyright (c) 2021 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __DATA_ARRAY_H_
#define __DATA_ARRAY_H_

#include "core/buffer.hpp"
#include "core/data_adapter.h"
#include "utils/nonown_ptr.hpp"
#include <vector>

namespace Aperture {

template <typename T>
class data_array {
 public:
#ifdef CUDA_ENABLED
  typedef typename cuda_adapter<T>::type adapted_type;
#else
  typedef typename host_adapter<T>::type adapted_type;
#endif

  void init() {
    for (int i = 0; i < m_data.size(); i++) {
      m_data[i]->init();
    }
  }

  void resize(int size) {
    m_data.resize(size);
    m_ptrs.set_memtype(MemType::host_device);
    m_ptrs.resize(size);
  }

  void set(int i, const nonown_ptr<T> &p) {
    m_data[i] = p;
#ifdef CUDA_ENABLED
    m_ptrs[i] = cuda_adapter<T>::apply(*p);
#else
    m_ptrs[i] = host_adapter<T>::apply(*p);
#endif
  }

  void copy_to_device() {
    for (int i = 0; i < m_data.size(); i++) {
#ifdef CUDA_ENABLED
      m_ptrs[i] = cuda_adapter<T>::apply(*m_data[i]);
#else
      m_ptrs[i] = host_adapter<T>::apply(*m_data[i]);
#endif
    }
    m_ptrs.copy_to_device();
  }

  int size() const { return m_data.size(); }

  nonown_ptr<T> &operator[](int i) { return m_data[i]; }
  const nonown_ptr<T> &operator[](int i) const { return m_data[i]; }

  std::vector<nonown_ptr<T>> &data() { return m_data; }
  const std::vector<nonown_ptr<T>> &data() const { return m_data; }

  adapted_type *dev_ptrs() { return m_ptrs.dev_ptr(); }
  const adapted_type *dev_ptrs() const { return m_ptrs.dev_ptr(); }
  adapted_type *host_ptrs() { return m_ptrs.host_ptr(); }
  const adapted_type *host_ptrs() const { return m_ptrs.host_ptr(); }

 private:
  std::vector<nonown_ptr<T>> m_data;
  buffer<adapted_type> m_ptrs;
};

template <typename T>
struct host_adapter<data_array<T>> {
  typedef typename host_adapter<T>::type *type;

  static inline type apply(data_array<T> &array) { return array.host_ptrs(); }
};

#ifdef CUDA_ENABLED

template <typename T>
struct cuda_adapter<data_array<T>> {
  typedef typename cuda_adapter<T>::type *type;

  static inline type apply(data_array<T> &array) { return array.dev_ptrs(); }
};

#endif

}  // namespace Aperture

#endif  // __DATA_ARRAY_H_
