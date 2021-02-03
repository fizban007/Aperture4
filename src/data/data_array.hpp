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
  typedef typename cuda_adapter<T>::type adapted_type;

  void resize(int size) {
    m_data.resize(size);
    m_ptrs.set_memtype(MemType::host_device);
    m_ptrs.resize(size);
  }

  void set(int i, nonown_ptr<T>& p) {
    m_data[i] = p;
    m_ptrs[i] = cuda_adapter<T>::apply(*p);
  }

  void copy_to_device() {
    m_ptrs.copy_to_device();
  }

  std::vector<nonown_ptr<T>>& data() { return m_data; }
  const std::vector<nonown_ptr<T>>& data() const { return m_data; }

  adapted_type* ptrs() { return m_ptrs.dev_ptr(); }
  const adapted_type* ptrs() const { return m_ptrs.dev_ptr(); }

 private:
  std::vector<nonown_ptr<T>> m_data;
  buffer<adapted_type> m_ptrs;
};

#ifdef CUDA_ENABLED

template <typename T>
class cuda_adapter<data_array<T>> {
  typedef typename cuda_adapter<T>::type* type;

  static inline type apply(data_array<T>& array) {
    return array.ptrs();
  }
};

#endif

}  // namespace Aperture

#endif  // __DATA_ARRAY_H_
