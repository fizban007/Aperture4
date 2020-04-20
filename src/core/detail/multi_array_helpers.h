#ifndef _MULTI_ARRAY_HELPERS_H_
#define _MULTI_ARRAY_HELPERS_H_

#include "core/multi_array.hpp"
#include "data/fields.hpp"
#include "utils/stagger.h"

namespace Aperture {

template <typename T, typename U, int Rank>
void resample(const multi_array<T, Rank>& from, multi_array<U, Rank>& to,
              const index_t<Rank>& offset, stagger_t st_src, stagger_t st_dst,
              int downsample = 1);

template <typename T, typename U, int Rank>
void resample_dev(const multi_array<T, Rank>& from, multi_array<U, Rank>& to,
                  const index_t<Rank>& offset, stagger_t st_src, stagger_t st_dst,
                  int downsample = 1);

}

#endif  // _MULTI_ARRAY_HELPERS_H_
