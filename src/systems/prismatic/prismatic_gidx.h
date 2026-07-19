#pragma once

#include <cstdint>

namespace Aperture {

// GLOBAL 3D element/cochain index type (checkpoint plan appendix item
// 1b).  The largest cochain, h_edges = (N_r+1)·30·4^L, passes 2^31
// between L8 and L9, so every index into a global 3D cochain range —
// layouts, halo-plan lists, ownership queries, l2g maps, 3D vertex ids
// — is 64-bit.  SPHERE-level indices (tris, sphere edges/vertices:
// O(4^L), < 10^9 through L12) and LOCAL indices (per-rank) stay int.
using gidx_t = int64_t;

}  // namespace Aperture
