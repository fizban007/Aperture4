#include "multi_array_helpers.h"
#include "utils/interpolation.hpp"

namespace Aperture {

template <typename T, typename U, int Rank>
void
resample(const multi_array<T, Rank>& from, multi_array<U, Rank>& to,
         const index_t<Rank>& offset, stagger_t st_src, stagger_t st_dst,
         int downsample) {
  auto interp = lerp<Rank>{};
  for (auto idx : to.indices()) {
    auto pos = idx.get_pos();
    bool in_bound = true;
    for (int i = 0; i < Rank; i++) {
      if (pos[i] < offset[i] || pos[i] >= to.extent()[i] - offset[i])
        in_bound = false;
    }
    if (!in_bound) continue;
    auto idx_src = from.get_idx(pos * downsample + offset);
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

template void resample(const multi_array<float, 1>& from,
                       multi_array<float, 1>& to, const index_t<1>& offset,
                       stagger_t st_src, stagger_t st_dst, int downsample);
template void resample(const multi_array<float, 2>& from,
                       multi_array<float, 2>& to, const index_t<2>& offset,
                       stagger_t st_src, stagger_t st_dst, int downsample);
template void resample(const multi_array<float, 3>& from,
                       multi_array<float, 3>& to, const index_t<3>& offset,
                       stagger_t st_src, stagger_t st_dst, int downsample);
template void resample(const multi_array<double, 1>& from,
                       multi_array<float, 1>& to, const index_t<1>& offset,
                       stagger_t st_src, stagger_t st_dst, int downsample);
template void resample(const multi_array<double, 2>& from,
                       multi_array<float, 2>& to, const index_t<2>& offset,
                       stagger_t st_src, stagger_t st_dst, int downsample);
template void resample(const multi_array<double, 3>& from,
                       multi_array<float, 3>& to, const index_t<3>& offset,
                       stagger_t st_src, stagger_t st_dst, int downsample);
template void resample(const multi_array<double, 1>& from,
                       multi_array<double, 1>& to, const index_t<1>& offset,
                       stagger_t st_src, stagger_t st_dst, int downsample);
template void resample(const multi_array<double, 2>& from,
                       multi_array<double, 2>& to, const index_t<2>& offset,
                       stagger_t st_src, stagger_t st_dst, int downsample);
template void resample(const multi_array<double, 3>& from,
                       multi_array<double, 3>& to, const index_t<3>& offset,
                       stagger_t st_src, stagger_t st_dst, int downsample);

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

}  // namespace Aperture
