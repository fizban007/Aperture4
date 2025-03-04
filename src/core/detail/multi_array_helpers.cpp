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

#include "multi_array_helpers.h"
#include "utils/interpolation.hpp"

namespace Aperture {

template <typename T, typename U, int Rank>
void
resample(exec_tags::host, const multi_array<T, Rank>& from, multi_array<U, Rank>& to,
         const index_t<Rank>& offset_src, const index_t<Rank>& offset_dst,
         stagger_t st_src, stagger_t st_dst, int downsample) {
  auto interp = lerp<Rank>{};
  auto ext_from = from.extent();
  auto ext_to = to.extent();
  for (auto idx : to.indices()) {
    auto pos = get_pos(idx, ext_to);
    bool in_bound = true;
    for (int i = 0; i < Rank; i++) {
      if (pos[i] < offset_dst[i] || pos[i] >= to.extent()[i] - offset_dst[i]) {
        in_bound = false;
        break;
      }
    }
    if (!in_bound) continue;
    auto idx_src = from.get_idx((pos - offset_dst) * downsample + offset_src);
    to[idx] = interp(from, idx_src, st_src, st_dst);
  }
}

template <typename T, int Rank>
void
add(exec_tags::host, multi_array<T, Rank>& dst, const multi_array<T, Rank>& src,
    const index_t<Rank>& dst_pos, const index_t<Rank>& src_pos,
    const extent_t<Rank>& ext, T scale) {
  // for (auto n : range(0, ext.size())) {
  using idx_t = idx_col_major_t<Rank>;
  for (auto idx : range(idx_t(0, ext), idx_t(ext.size(), ext))) {
    // idx_col_major_t<Rank> idx(n, ext);
    auto pos_offset = get_pos(idx, ext);
    auto idx_dst = dst.get_idx(dst_pos + pos_offset);
    auto idx_src = src.get_idx(src_pos + pos_offset);
    dst[idx_dst] += src[idx_src] * scale;
  }
}

template <typename T, typename U, int Rank>
void
copy(exec_tags::host, multi_array<T, Rank>& dst, const multi_array<U, Rank>& src,
     const index_t<Rank>& dst_pos, const index_t<Rank>& src_pos,
     const extent_t<Rank>& ext) {
  using idx_t = idx_col_major_t<Rank>;
  for (auto idx : range(idx_t(0, ext), idx_t(ext.size(), ext))) {
    // idx_col_major_t<Rank> idx(n, ext);
    auto pos_offset = get_pos(idx, ext);
    auto idx_dst = dst.get_idx(dst_pos + pos_offset);
    auto idx_src = src.get_idx(src_pos + pos_offset);
    dst[idx_dst] = src[idx_src];
  }
}

#define INSTANTIATE_RESAMPLE_DIM(type1, type2, dim)                     \
  template void resample(exec_tags::host, const multi_array<type1, dim>&,                \
                         multi_array<type2, dim>&, const index_t<dim>&, \
                         const index_t<dim>&, stagger_t, stagger_t, int)

#define INSTANTIATE_RESAMPLE(type1, type2)   \
  INSTANTIATE_RESAMPLE_DIM(type1, type2, 1); \
  INSTANTIATE_RESAMPLE_DIM(type1, type2, 2); \
  INSTANTIATE_RESAMPLE_DIM(type1, type2, 3)

INSTANTIATE_RESAMPLE(float, float);
INSTANTIATE_RESAMPLE(float, double);
INSTANTIATE_RESAMPLE(double, float);
INSTANTIATE_RESAMPLE(double, double);

#define INSTANTIATE_ADD(type, dim)                                            \
  template void add(exec_tags::host, multi_array<type, dim>& dst,                              \
                    const multi_array<type, dim>& src,                        \
                    const index_t<dim>& dst_pos, const index_t<dim>& src_pos, \
                    const extent_t<dim>& ext, type scale)

INSTANTIATE_ADD(float, 1);
INSTANTIATE_ADD(float, 2);
INSTANTIATE_ADD(float, 3);
INSTANTIATE_ADD(double, 1);
INSTANTIATE_ADD(double, 2);
INSTANTIATE_ADD(double, 3);

#define INSTANTIATE_COPY(type1, type2, dim)                                            \
  template void copy(exec_tags::host, multi_array<type1, dim>& dst,                              \
                     const multi_array<type2, dim>& src,                        \
                     const index_t<dim>& dst_pos, const index_t<dim>& src_pos, \
                     const extent_t<dim>& ext)

INSTANTIATE_COPY(float, float, 1);
INSTANTIATE_COPY(float, float, 2);
INSTANTIATE_COPY(float, float, 3);
INSTANTIATE_COPY(float, float, 4);
INSTANTIATE_COPY(double, double, 1);
INSTANTIATE_COPY(double, double, 2);
INSTANTIATE_COPY(double, double, 3);
INSTANTIATE_COPY(double, double, 4);
INSTANTIATE_COPY(float, double, 1);
INSTANTIATE_COPY(float, double, 2);
INSTANTIATE_COPY(float, double, 3);
INSTANTIATE_COPY(float, double, 4);

// #if !defined(CUDA_ENABLED) && !defined(HIP_ENABLED)
// template <typename T, int Rank>
// void
// add_dev(multi_array<T, Rank>& dst, const multi_array<T, Rank>& src,
//         const index_t<Rank>& dst_pos, const index_t<Rank>& src_pos,
//         const extent_t<Rank>& ext, T scale, const gpuStream_t* stream) {}

// template <typename T, typename U, int Rank>
// void
// resample_dev(const multi_array<T, Rank>& from, multi_array<U, Rank>& to,
//              const index_t<Rank>& offset_src, const index_t<Rank>& offset_dst,
//              stagger_t st_src, stagger_t st_dst, int downsample,
//              const gpuStream_t* stream) {}

// template <typename T, int Rank>
// void
// copy_dev(multi_array<T, Rank>& dst, const multi_array<T, Rank>& src,
//          const index_t<Rank>& dst_pos, const index_t<Rank>& src_pos,
//          const extent_t<Rank>& ext, const gpuStream_t* stream) {}

// template void resample_dev(const multi_array<float, 1>&, multi_array<float, 1>&,
//                            const index_t<1>&, const index_t<1>&, stagger_t,
//                            stagger_t, int, const gpuStream_t* stream);
// template void resample_dev(const multi_array<float, 2>&, multi_array<float, 2>&,
//                            const index_t<2>&, const index_t<2>&, stagger_t,
//                            stagger_t, int, const gpuStream_t* stream);
// template void resample_dev(const multi_array<float, 3>&, multi_array<float, 3>&,
//                            const index_t<3>&, const index_t<3>&, stagger_t,
//                            stagger_t, int, const gpuStream_t* stream);
// template void resample_dev(const multi_array<double, 1>&,
//                            multi_array<float, 1>&, const index_t<1>&,
//                            const index_t<1>&, stagger_t, stagger_t, int,
//                            const gpuStream_t* stream);
// template void resample_dev(const multi_array<double, 2>&,
//                            multi_array<float, 2>&, const index_t<2>&,
//                            const index_t<2>&, stagger_t, stagger_t, int,
//                            const gpuStream_t* stream);
// template void resample_dev(const multi_array<double, 3>&,
//                            multi_array<float, 3>&, const index_t<3>&,
//                            const index_t<3>&, stagger_t, stagger_t, int,
//                            const gpuStream_t* stream);
// template void resample_dev(const multi_array<double, 1>&,
//                            multi_array<double, 1>&, const index_t<1>&,
//                            const index_t<1>&, stagger_t, stagger_t, int,
//                            const gpuStream_t* stream);
// template void resample_dev(const multi_array<double, 2>&,
//                            multi_array<double, 2>&, const index_t<2>&,
//                            const index_t<2>&, stagger_t, stagger_t, int,
//                            const gpuStream_t* stream);
// template void resample_dev(const multi_array<double, 3>&,
//                            multi_array<double, 3>&, const index_t<3>&,
//                            const index_t<3>&, stagger_t, stagger_t, int,
//                            const gpuStream_t* stream);

// template void add_dev(multi_array<float, 1>& dst,
//                       const multi_array<float, 1>& src,
//                       const index_t<1>& dst_pos, const index_t<1>& src_pos,
//                       const extent_t<1>& ext, float scale,
//                       const gpuStream_t* stream);
// template void add_dev(multi_array<float, 2>& dst,
//                       const multi_array<float, 2>& src,
//                       const index_t<2>& dst_pos, const index_t<2>& src_pos,
//                       const extent_t<2>& ext, float scale,
//                       const gpuStream_t* stream);
// template void add_dev(multi_array<float, 3>& dst,
//                       const multi_array<float, 3>& src,
//                       const index_t<3>& dst_pos, const index_t<3>& src_pos,
//                       const extent_t<3>& ext, float scale,
//                       const gpuStream_t* stream);
// template void add_dev(multi_array<double, 1>& dst,
//                       const multi_array<double, 1>& src,
//                       const index_t<1>& dst_pos, const index_t<1>& src_pos,
//                       const extent_t<1>& ext, double scale,
//                       const gpuStream_t* stream);
// template void add_dev(multi_array<double, 2>& dst,
//                       const multi_array<double, 2>& src,
//                       const index_t<2>& dst_pos, const index_t<2>& src_pos,
//                       const extent_t<2>& ext, double scale,
//                       const gpuStream_t* stream);
// template void add_dev(multi_array<double, 3>& dst,
//                       const multi_array<double, 3>& src,
//                       const index_t<3>& dst_pos, const index_t<3>& src_pos,
//                       const extent_t<3>& ext, double scale,
//                       const gpuStream_t* stream);

// template void copy_dev(multi_array<float, 1>& dst,
//                        const multi_array<float, 1>& src,
//                        const index_t<1>& dst_pos, const index_t<1>& src_pos,
//                        const extent_t<1>& ext, const gpuStream_t* stream);
// template void copy_dev(multi_array<float, 2>& dst,
//                        const multi_array<float, 2>& src,
//                        const index_t<2>& dst_pos, const index_t<2>& src_pos,
//                        const extent_t<2>& ext, const gpuStream_t* stream);
// template void copy_dev(multi_array<float, 3>& dst,
//                        const multi_array<float, 3>& src,
//                        const index_t<3>& dst_pos, const index_t<3>& src_pos,
//                        const extent_t<3>& ext, const gpuStream_t* stream);
// template void copy_dev(multi_array<double, 1>& dst,
//                        const multi_array<double, 1>& src,
//                        const index_t<1>& dst_pos, const index_t<1>& src_pos,
//                        const extent_t<1>& ext, const gpuStream_t* stream);
// template void copy_dev(multi_array<double, 2>& dst,
//                        const multi_array<double, 2>& src,
//                        const index_t<2>& dst_pos, const index_t<2>& src_pos,
//                        const extent_t<2>& ext, const gpuStream_t* stream);
// template void copy_dev(multi_array<double, 3>& dst,
//                        const multi_array<double, 3>& src,
//                        const index_t<3>& dst_pos, const index_t<3>& src_pos,
//                        const extent_t<3>& ext, const gpuStream_t* stream);
// #endif
}  // namespace Aperture
