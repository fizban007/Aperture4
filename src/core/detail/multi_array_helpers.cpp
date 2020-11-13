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
resample(const multi_array<T, Rank>& from, multi_array<U, Rank>& to,
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
add(multi_array<T, Rank>& dst, const multi_array<T, Rank>& src,
    const index_t<Rank>& dst_pos, const index_t<Rank>& src_pos,
    const extent_t<Rank>& ext, T scale) {
  for (auto n : range(0, ext.size())) {
    idx_col_major_t<Rank> idx(n, ext);
    auto idx_dst = dst.get_idx(dst_pos + idx.get_pos());
    auto idx_src = src.get_idx(src_pos + idx.get_pos());
    dst[idx_dst] += src[idx_src] * scale;
  }
}

template <typename T, int Rank>
void
copy(multi_array<T, Rank>& dst, const multi_array<T, Rank>& src,
     const index_t<Rank>& dst_pos, const index_t<Rank>& src_pos,
     const extent_t<Rank>& ext) {
  for (auto n : range(0, ext.size())) {
    idx_col_major_t<Rank> idx(n, ext);
    auto idx_dst = dst.get_idx(dst_pos + idx.get_pos());
    auto idx_src = src.get_idx(src_pos + idx.get_pos());
    dst[idx_dst] = src[idx_src];
  }
}

template void resample(const multi_array<float, 1>&, multi_array<float, 1>&,
                       const index_t<1>&, const index_t<1>&, stagger_t,
                       stagger_t, int);
template void resample(const multi_array<float, 2>&, multi_array<float, 2>&,
                       const index_t<2>&, const index_t<2>&, stagger_t,
                       stagger_t, int);
template void resample(const multi_array<float, 3>&, multi_array<float, 3>&,
                       const index_t<3>&, const index_t<3>&, stagger_t,
                       stagger_t, int);
template void resample(const multi_array<double, 1>&, multi_array<float, 1>&,
                       const index_t<1>&, const index_t<1>&, stagger_t,
                       stagger_t, int);
template void resample(const multi_array<double, 2>&, multi_array<float, 2>&,
                       const index_t<2>&, const index_t<2>&, stagger_t,
                       stagger_t, int);
template void resample(const multi_array<double, 3>&, multi_array<float, 3>&,
                       const index_t<3>&, const index_t<3>&, stagger_t,
                       stagger_t, int);
template void resample(const multi_array<double, 1>&, multi_array<double, 1>&,
                       const index_t<1>&, const index_t<1>&, stagger_t,
                       stagger_t, int);
template void resample(const multi_array<double, 2>&, multi_array<double, 2>&,
                       const index_t<2>&, const index_t<2>&, stagger_t,
                       stagger_t, int);
template void resample(const multi_array<double, 3>&, multi_array<double, 3>&,
                       const index_t<3>&, const index_t<3>&, stagger_t,
                       stagger_t, int);

template void add(multi_array<float, 1>& dst, const multi_array<float, 1>& src,
                  const index_t<1>& dst_pos, const index_t<1>& src_pos,
                  const extent_t<1>& ext, float scale);
template void add(multi_array<float, 2>& dst, const multi_array<float, 2>& src,
                  const index_t<2>& dst_pos, const index_t<2>& src_pos,
                  const extent_t<2>& ext, float scale);
template void add(multi_array<float, 3>& dst, const multi_array<float, 3>& src,
                  const index_t<3>& dst_pos, const index_t<3>& src_pos,
                  const extent_t<3>& ext, float scale);
template void add(multi_array<double, 1>& dst,
                  const multi_array<double, 1>& src, const index_t<1>& dst_pos,
                  const index_t<1>& src_pos, const extent_t<1>& ext,
                  double scale);
template void add(multi_array<double, 2>& dst,
                  const multi_array<double, 2>& src, const index_t<2>& dst_pos,
                  const index_t<2>& src_pos, const extent_t<2>& ext,
                  double scale);
template void add(multi_array<double, 3>& dst,
                  const multi_array<double, 3>& src, const index_t<3>& dst_pos,
                  const index_t<3>& src_pos, const extent_t<3>& ext,
                  double scale);

template void copy(multi_array<float, 1>& dst, const multi_array<float, 1>& src,
                   const index_t<1>& dst_pos, const index_t<1>& src_pos,
                   const extent_t<1>& ext);
template void copy(multi_array<float, 2>& dst, const multi_array<float, 2>& src,
                   const index_t<2>& dst_pos, const index_t<2>& src_pos,
                   const extent_t<2>& ext);
template void copy(multi_array<float, 3>& dst, const multi_array<float, 3>& src,
                   const index_t<3>& dst_pos, const index_t<3>& src_pos,
                   const extent_t<3>& ext);
template void copy(multi_array<double, 1>& dst,
                   const multi_array<double, 1>& src, const index_t<1>& dst_pos,
                   const index_t<1>& src_pos, const extent_t<1>& ext);
template void copy(multi_array<double, 2>& dst,
                   const multi_array<double, 2>& src, const index_t<2>& dst_pos,
                   const index_t<2>& src_pos, const extent_t<2>& ext);
template void copy(multi_array<double, 3>& dst,
                   const multi_array<double, 3>& src, const index_t<3>& dst_pos,
                   const index_t<3>& src_pos, const extent_t<3>& ext);

#ifndef CUDA_ENABLED
template <typename T, int Rank>
void
add_dev(multi_array<T, Rank>& dst, const multi_array<T, Rank>& src,
        const index_t<Rank>& dst_pos, const index_t<Rank>& src_pos,
        const extent_t<Rank>& ext, T scale, const cudaStream_t* stream) {}

template <typename T, typename U, int Rank>
void
resample_dev(const multi_array<T, Rank>& from, multi_array<U, Rank>& to,
             const index_t<Rank>& offset_src, const index_t<Rank>& offset_dst,
             stagger_t st_src, stagger_t st_dst, int downsample,
             const cudaStream_t* stream) {}

template <typename T, int Rank>
void
copy_dev(multi_array<T, Rank>& dst, const multi_array<T, Rank>& src,
         const index_t<Rank>& dst_pos, const index_t<Rank>& src_pos,
         const extent_t<Rank>& ext, const cudaStream_t* stream) {}

template void resample_dev(const multi_array<float, 1>&, multi_array<float, 1>&,
                           const index_t<1>&, const index_t<1>&, stagger_t,
                           stagger_t, int, const cudaStream_t* stream);
template void resample_dev(const multi_array<float, 2>&, multi_array<float, 2>&,
                           const index_t<2>&, const index_t<2>&, stagger_t,
                           stagger_t, int, const cudaStream_t* stream);
template void resample_dev(const multi_array<float, 3>&, multi_array<float, 3>&,
                           const index_t<3>&, const index_t<3>&, stagger_t,
                           stagger_t, int, const cudaStream_t* stream);
template void resample_dev(const multi_array<float, 1>&,
                           multi_array<double, 1>&, const index_t<1>&,
                           const index_t<1>&, stagger_t, stagger_t, int, const cudaStream_t* stream);
template void resample_dev(const multi_array<float, 2>&,
                           multi_array<double, 2>&, const index_t<2>&,
                           const index_t<2>&, stagger_t, stagger_t, int, const cudaStream_t* stream);
template void resample_dev(const multi_array<float, 3>&,
                           multi_array<double, 3>&, const index_t<3>&,
                           const index_t<3>&, stagger_t, stagger_t, int, const cudaStream_t* stream);
template void resample_dev(const multi_array<double, 1>&,
                           multi_array<double, 1>&, const index_t<1>&,
                           const index_t<1>&, stagger_t, stagger_t, int, const cudaStream_t* stream);
template void resample_dev(const multi_array<double, 2>&,
                           multi_array<double, 2>&, const index_t<2>&,
                           const index_t<2>&, stagger_t, stagger_t, int, const cudaStream_t* stream);
template void resample_dev(const multi_array<double, 3>&,
                           multi_array<double, 3>&, const index_t<3>&,
                           const index_t<3>&, stagger_t, stagger_t, int, const cudaStream_t* stream);

template void add_dev(multi_array<float, 1>& dst,
                      const multi_array<float, 1>& src,
                      const index_t<1>& dst_pos, const index_t<1>& src_pos,
                      const extent_t<1>& ext, float scale, const cudaStream_t* stream);
template void add_dev(multi_array<float, 2>& dst,
                      const multi_array<float, 2>& src,
                      const index_t<2>& dst_pos, const index_t<2>& src_pos,
                      const extent_t<2>& ext, float scale, const cudaStream_t* stream);
template void add_dev(multi_array<float, 3>& dst,
                      const multi_array<float, 3>& src,
                      const index_t<3>& dst_pos, const index_t<3>& src_pos,
                      const extent_t<3>& ext, float scale, const cudaStream_t* stream);
template void add_dev(multi_array<double, 1>& dst,
                      const multi_array<double, 1>& src,
                      const index_t<1>& dst_pos, const index_t<1>& src_pos,
                      const extent_t<1>& ext, double scale, const cudaStream_t* stream);
template void add_dev(multi_array<double, 2>& dst,
                      const multi_array<double, 2>& src,
                      const index_t<2>& dst_pos, const index_t<2>& src_pos,
                      const extent_t<2>& ext, double scale, const cudaStream_t* stream);
template void add_dev(multi_array<double, 3>& dst,
                      const multi_array<double, 3>& src,
                      const index_t<3>& dst_pos, const index_t<3>& src_pos,
                      const extent_t<3>& ext, double scale, const cudaStream_t* stream);

template void copy_dev(multi_array<float, 1>& dst,
                       const multi_array<float, 1>& src,
                       const index_t<1>& dst_pos, const index_t<1>& src_pos,
                       const extent_t<1>& ext, const cudaStream_t* stream);
template void copy_dev(multi_array<float, 2>& dst,
                       const multi_array<float, 2>& src,
                       const index_t<2>& dst_pos, const index_t<2>& src_pos,
                       const extent_t<2>& ext, const cudaStream_t* stream);
template void copy_dev(multi_array<float, 3>& dst,
                       const multi_array<float, 3>& src,
                       const index_t<3>& dst_pos, const index_t<3>& src_pos,
                       const extent_t<3>& ext, const cudaStream_t* stream);
template void copy_dev(multi_array<double, 1>& dst,
                       const multi_array<double, 1>& src,
                       const index_t<1>& dst_pos, const index_t<1>& src_pos,
                       const extent_t<1>& ext, const cudaStream_t* stream);
template void copy_dev(multi_array<double, 2>& dst,
                       const multi_array<double, 2>& src,
                       const index_t<2>& dst_pos, const index_t<2>& src_pos,
                       const extent_t<2>& ext, const cudaStream_t* stream);
template void copy_dev(multi_array<double, 3>& dst,
                       const multi_array<double, 3>& src,
                       const index_t<3>& dst_pos, const index_t<3>& src_pos,
                       const extent_t<3>& ext, const cudaStream_t* stream);
#endif
}  // namespace Aperture
