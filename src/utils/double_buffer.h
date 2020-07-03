/*
 * Copyright (c) 2020 Alex Chen.
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

#ifndef __DOUBLE_BUFFER_H_
#define __DOUBLE_BUFFER_H_

#include "core/cuda_control.h"

namespace Aperture {

/// Many algorithms require iteration and it is beneficial to have two
/// buffers/arrays so that the iteration can bounce back and forth between the
/// two. The `double_buffer` class solves this problem and makes bouncing
/// between two classes of the same type seamless.
template <typename T>
struct double_buffer {
  T* buffers[2];
  int selector = 0;

  HD_INLINE double_buffer() {
    buffers[0] = nullptr;
    buffers[1] = nullptr;
  }

  HD_INLINE double_buffer(T* main, T* alt) {
    buffers[0] = main;
    buffers[1] = alt;
  }

  HD_INLINE T& main() { return *buffers[selector]; }
  HD_INLINE T& alt() { return *buffers[selector ^ 1]; }
  HD_INLINE void swap() { selector ^= 1; }
};

template <typename T>
double_buffer<T> make_double_buffer(T& t1, T& t2) {
  double_buffer<T> result(&t1, &t2);
  return result;
}

}

#endif
