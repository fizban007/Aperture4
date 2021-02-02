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

#include "core/multi_array_exp.hpp"
#include "core/ndsubset_dev.hpp"
#include "multi_array_helpers.h"
#include "utils/interpolation.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include <cuda_runtime_api.h>

namespace Aperture {

template <typename T, typename U, int Rank>
void
resample_dev(const multi_array<T, Rank>& from, multi_array<U, Rank>& to,
             const index_t<Rank>& offset_src, const index_t<Rank>& offset_dst,
             stagger_t st_src, stagger_t st_dst, int downsample,
             const cudaStream_t* stream) {
  auto ext = to.extent();
  auto ext_src = from.extent();
  auto resample_kernel = [downsample, ext, ext_src] __device__(
                             auto p_src, auto p_dst, auto offset_src,
                             auto offset_dst, auto st_src, auto st_dst) {
    auto interp = lerp<Rank>{};
    for (auto n : grid_stride_range(0, ext.size())) {
      auto idx = p_dst.idx_at(n, ext);
      // auto pos = idx.get_pos();
      auto pos = get_pos(idx, ext);
      bool in_bound = true;
#pragma unroll
      for (int i = 0; i < Rank; i++) {
        if (pos[i] < offset_dst[i] || pos[i] >= ext[i] - offset_dst[i])
          in_bound = false;
      }
      if (!in_bound) continue;
      auto idx_src =
          p_src.get_idx((pos - offset_dst) * downsample + offset_src, ext_src);
      p_dst[idx] = interp(p_src, idx_src, st_src, st_dst);
    }
  };
  kernel_exec_policy p;
  configure_grid(p, resample_kernel, from.dev_ndptr_const(), to.dev_ndptr(),
                 offset_src, offset_dst, st_src, st_dst);
  if (stream != nullptr) p.set_stream(*stream);
  kernel_launch(p, resample_kernel, from.dev_ndptr_const(), to.dev_ndptr(),
                offset_src, offset_dst, st_src, st_dst);
  CudaSafeCall(cudaDeviceSynchronize());
}

template <typename T, int Rank>
void
add_dev(multi_array<T, Rank>& dst, const multi_array<T, Rank>& src,
        const index_t<Rank>& dst_pos, const index_t<Rank>& src_pos,
        const extent_t<Rank>& ext, T scale, const cudaStream_t* stream) {
  // auto add_kernel = [ext, scale] __device__(auto dst_ptr, auto src_ptr,
  //                                           auto dst_pos, auto src_pos,
  //                                           auto dst_ext, auto src_ext) {
  //   for (auto n : grid_stride_range(0, ext.size())) {
  //     idx_col_major_t<Rank> idx(n, ext);
  //     auto idx_dst = dst_ptr.get_idx(dst_pos + idx.get_pos(), dst_ext);
  //     auto idx_src = src_ptr.get_idx(src_pos + idx.get_pos(), src_ext);
  //     dst_ptr[idx_dst] += src_ptr[idx_src] * scale;
  //   }
  // };
  // kernel_exec_policy p;
  // configure_grid(p, add_kernel, dst.dev_ndptr(), src.dev_ndptr_const(),
  // dst_pos,
  //                src_pos, dst.extent(), src.extent());
  // if (stream != nullptr) p.set_stream(*stream);
  // kernel_launch(p, add_kernel, dst.dev_ndptr(), src.dev_ndptr_const(),
  // dst_pos,
  //               src_pos, dst.extent(), src.extent());
  // CudaSafeCall(cudaDeviceSynchronize());
  if (stream != nullptr) {
    select_dev(dst, dst_pos, ext).with_stream(*stream) +=
        select_dev(scale * src, src_pos, ext);
  } else {
    select_dev(dst, dst_pos, ext) += select_dev(scale * src, src_pos, ext);
  }
}

template <typename T, int Rank>
void
copy_dev(multi_array<T, Rank>& dst, const multi_array<T, Rank>& src,
         const index_t<Rank>& dst_pos, const index_t<Rank>& src_pos,
         const extent_t<Rank>& ext, const cudaStream_t* stream) {
  // auto copy_kernel = [ext] __device__(auto dst_ptr, auto src_ptr, auto
  // dst_pos,
  //                                     auto src_pos, auto dst_ext,
  //                                     auto src_ext) {
  //   for (auto n : grid_stride_range(0, ext.size())) {
  //     // Always use column major inside loop to simplify conversion between
  //     // different indexing schemes
  //     idx_col_major_t<Rank> idx(n, ext);
  //     auto idx_dst = dst_ptr.get_idx(dst_pos + idx.get_pos(), dst_ext);
  //     auto idx_src = src_ptr.get_idx(src_pos + idx.get_pos(), src_ext);
  //     dst_ptr[idx_dst] = src_ptr[idx_src];
  //   }
  // };
  // kernel_exec_policy p;
  // configure_grid(p, copy_kernel, dst.dev_ndptr(), src.dev_ndptr_const(),
  //                dst_pos, src_pos, dst.extent(), src.extent());
  // if (stream != nullptr) p.set_stream(*stream);
  // kernel_launch(p, copy_kernel, dst.dev_ndptr(), src.dev_ndptr_const(),
  // dst_pos,
  //               src_pos, dst.extent(), src.extent());
  // CudaSafeCall(cudaDeviceSynchronize());
  if (stream != nullptr) {
    select_dev(dst, dst_pos, ext).with_stream(*stream) =
        select_dev(src, src_pos, ext);
  } else {
    select_dev(dst, dst_pos, ext) = select_dev(src, src_pos, ext);
  }
}

#define INSTANTIATE_RESAMPLE_DEV_DIM(type1, type2, dim)                 \
  template void resample_dev(                                           \
      const multi_array<type1, dim>& from, multi_array<type2, dim>& to, \
      const index_t<dim>& offset, const index_t<dim>& offset_dst,       \
      stagger_t st_src, stagger_t st_dst, int downsample,               \
      const cudaStream_t* stream)

#define INSTANTIATE_RESAMPLE_DEV(type1, type2)   \
  INSTANTIATE_RESAMPLE_DEV_DIM(type1, type2, 1); \
  INSTANTIATE_RESAMPLE_DEV_DIM(type1, type2, 2); \
  INSTANTIATE_RESAMPLE_DEV_DIM(type1, type2, 3)

INSTANTIATE_RESAMPLE_DEV(float, float);
INSTANTIATE_RESAMPLE_DEV(float, double);
INSTANTIATE_RESAMPLE_DEV(double, float);

#define INSTANTIATE_ADD_DEV(type, dim)                                \
  template void add_dev(                                              \
      multi_array<type, dim>& dst, const multi_array<type, dim>& src, \
      const index_t<dim>& dst_pos, const index_t<dim>& src_pos,       \
      const extent_t<dim>& ext, type scale, const cudaStream_t* stream)

INSTANTIATE_ADD_DEV(float, 1);
INSTANTIATE_ADD_DEV(float, 2);
INSTANTIATE_ADD_DEV(float, 3);
INSTANTIATE_ADD_DEV(double, 1);
INSTANTIATE_ADD_DEV(double, 2);
INSTANTIATE_ADD_DEV(double, 3);

#define INSTANTIATE_COPY_DEV(type, dim)                               \
  template void copy_dev(                                             \
      multi_array<type, dim>& dst, const multi_array<type, dim>& src, \
      const index_t<dim>& dst_pos, const index_t<dim>& src_pos,       \
      const extent_t<dim>& ext, const cudaStream_t* stream)

INSTANTIATE_COPY_DEV(float, 1);
INSTANTIATE_COPY_DEV(float, 2);
INSTANTIATE_COPY_DEV(float, 3);
INSTANTIATE_COPY_DEV(double, 1);
INSTANTIATE_COPY_DEV(double, 2);
INSTANTIATE_COPY_DEV(double, 3);

}  // namespace Aperture
