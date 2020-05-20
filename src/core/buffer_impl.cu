#include "buffer_impl.hpp"
#include "core/cuda_control.h"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace Aperture {

template <typename T>
void
ptr_assign_dev(T* array, size_t start, size_t end, const T& value) {
  auto ptr = thrust::device_pointer_cast(array);
  thrust::fill(ptr + start, ptr + end, value);
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename T>
void
ptr_copy_dev(T* src, T* dst, size_t num, size_t src_pos, size_t dst_pos) {
  auto src_ptr = thrust::device_pointer_cast(src);
  auto dst_ptr = thrust::device_pointer_cast(dst);
  thrust::copy(src_ptr + src_pos, src_ptr + src_pos + num,
               dst_ptr + dst_pos);
  CudaSafeCall(cudaDeviceSynchronize());
}


template void ptr_assign_dev(int*, size_t, size_t, const int&);
template void ptr_assign_dev(long*, size_t, size_t, const long&);
template void ptr_assign_dev(uint32_t*, size_t, size_t, const uint32_t&);
template void ptr_assign_dev(uint64_t*, size_t, size_t, const uint64_t&);
template void ptr_assign_dev(float*, size_t, size_t, const float&);
template void ptr_assign_dev(double*, size_t, size_t, const double&);

template void ptr_copy_dev(int*, int*, size_t, size_t, size_t);
template void ptr_copy_dev(long*, long*, size_t, size_t, size_t);
template void ptr_copy_dev(uint32_t*, uint32_t*, size_t, size_t, size_t);
template void ptr_copy_dev(uint64_t*, uint64_t*, size_t, size_t, size_t);
template void ptr_copy_dev(float*, float*, size_t, size_t, size_t);
template void ptr_copy_dev(double*, double*, size_t, size_t, size_t);

}
