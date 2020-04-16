#include "field_helpers.h"
#include "utils/interpolation.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"

namespace Aperture {

template <typename T, int Rank>
void
resample_dev(const multi_array<T, Rank>& from, multi_array<T, Rank>& to,
             const index_t<Rank>& offset, stagger_t st_src, stagger_t st_dst,
             int downsample) {
  auto ext = to.extent();
  auto ext_src = from.extent();
  kernel_launch(
      [downsample, ext, ext_src] __device__(auto p_src, auto p_dst, auto offset,
                                            auto st_src, auto st_dst) {
        auto interp = lerp<Rank>{};
        for (auto n : grid_stride_range(0, ext.size())) {
          auto idx = p_dst.idx_at(n, ext);
          auto pos = idx.get_pos();
          bool in_bound = true;
#pragma unroll
          for (int i = 0; i < Rank; i++) {
            if (pos[i] < offset[i] || pos[i] >= ext[i] - offset[i])
              in_bound = false;
          }
          if (!in_bound) continue;
          auto idx_out =
              p_src.get_idx((pos - offset) * downsample + offset, ext_src);
          p_dst[idx_out] = interp(p_src, idx_out, st_src, st_dst);
        }
      },
      from.get_const_ptr(), to.get_ptr(), offset, st_src, st_dst);
}

template void resample_dev(const multi_array<float, 1>& from,
                           multi_array<float, 1>& to, const index_t<1>& offset,
                           stagger_t st_src, stagger_t st_dst, int downsample);
template void resample_dev(const multi_array<float, 2>& from,
                           multi_array<float, 2>& to, const index_t<2>& offset,
                           stagger_t st_src, stagger_t st_dst, int downsample);
template void resample_dev(const multi_array<float, 3>& from,
                           multi_array<float, 3>& to, const index_t<3>& offset,
                           stagger_t st_src, stagger_t st_dst, int downsample);
template void resample_dev(const multi_array<double, 1>& from,
                           multi_array<double, 1>& to, const index_t<1>& offset,
                           stagger_t st_src, stagger_t st_dst, int downsample);
template void resample_dev(const multi_array<double, 2>& from,
                           multi_array<double, 2>& to, const index_t<2>& offset,
                           stagger_t st_src, stagger_t st_dst, int downsample);
template void resample_dev(const multi_array<double, 3>& from,
                           multi_array<double, 3>& to, const index_t<3>& offset,
                           stagger_t st_src, stagger_t st_dst, int downsample);

}  // namespace Aperture
