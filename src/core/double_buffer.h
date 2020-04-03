#ifndef __DOUBLE_BUFFER_H_
#define __DOUBLE_BUFFER_H_

#include "cuda_control.h"

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

  HD_INLINE double_buffer(T* current, T* alternative) {
    buffers[0] = current;
    buffers[1] = alternative;
  }

  HD_INLINE T* current() { return buffers[selector]; }
  HD_INLINE T* alternative() { return buffers[selector ^ 1]; }
  HD_INLINE void swap() { selector ^= 1; }
};


}

#endif
