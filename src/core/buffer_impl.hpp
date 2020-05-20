#ifndef __BUFFER_IMPL_H_
#define __BUFFER_IMPL_H_

#include <cstdlib>

namespace Aperture {

template <typename T>
void ptr_assign(T* array, size_t start, size_t end, const T& value);

template <typename T>
void ptr_assign_dev(T* array, size_t start, size_t end, const T& value);

template <typename T>
void ptr_copy(T* src, T* dst, size_t num, size_t src_pos, size_t dst_pos);

template <typename T>
void ptr_copy_dev(T* src, T* dst, size_t num, size_t src_pos, size_t dst_pos);

}

#endif // __BUFFER_IMPL_H_
