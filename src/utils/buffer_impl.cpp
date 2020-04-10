#include "buffer_impl.hpp"
#include <algorithm>

namespace Aperture {

template <typename T>
void
ptr_assign(T* array, size_t start, size_t end, const T& value) {
  std::fill(array + start, array + end, value);
}

template <typename T>
void
ptr_copy(T* src, T* dst, size_t num, size_t src_pos, size_t dst_pos) {
  std::copy(src + src_pos, src + src_pos + num, dst + dst_pos);
}

template void ptr_assign(int*, size_t, size_t, const int&);
template void ptr_assign(long*, size_t, size_t, const long&);
template void ptr_assign(uint32_t*, size_t, size_t, const uint32_t&);
template void ptr_assign(uint64_t*, size_t, size_t, const uint64_t&);
template void ptr_assign(float*, size_t, size_t, const float&);
template void ptr_assign(double*, size_t, size_t, const double&);

template void ptr_copy(int*, int*, size_t, size_t, size_t);
template void ptr_copy(long*, long*, size_t, size_t, size_t);
template void ptr_copy(uint32_t*, uint32_t*, size_t, size_t, size_t);
template void ptr_copy(uint64_t*, uint64_t*, size_t, size_t, size_t);
template void ptr_copy(float*, float*, size_t, size_t, size_t);
template void ptr_copy(double*, double*, size_t, size_t, size_t);

}  // namespace Aperture
