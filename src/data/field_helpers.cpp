#include "field_helpers.h"
#include "utils/interpolation.hpp"

namespace Aperture {

template <typename T, int Rank>
void
resample(const multi_array<T, Rank>& from, multi_array<T, Rank>& to,
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
    auto idx_out = from.get_idx((pos - offset) * downsample + offset);
    to[idx_out] = interp(from, idx_out, st_src, st_dst);
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
                       multi_array<double, 1>& to, const index_t<1>& offset,
                       stagger_t st_src, stagger_t st_dst, int downsample);
template void resample(const multi_array<double, 2>& from,
                       multi_array<double, 2>& to, const index_t<2>& offset,
                       stagger_t st_src, stagger_t st_dst, int downsample);
template void resample(const multi_array<double, 3>& from,
                       multi_array<double, 3>& to, const index_t<3>& offset,
                       stagger_t st_src, stagger_t st_dst, int downsample);

}  // namespace Aperture
