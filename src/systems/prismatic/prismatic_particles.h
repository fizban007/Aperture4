#pragma once

#include "core/detail/macro_trickery.h"
#include "core/gpu_translation_layer.h"
#include "core/particles.h"
#include "core/typedefs_and_constants.h"

// Particle struct for the prismatic mesh.
//
// Position encoding:
//   x1 = lambda_1  (barycentric coordinate, lambda_3 = 1 - x1 - x2)
//   x2 = lambda_2
//   x3 = zeta      (normalized radial coordinate within layer, in [0,1])
//
// Momentum: Cartesian (p1, p2, p3)
//
// Cell encoding (shell-major, matches face/edge layout):
//   cell = layer_idx * N_tri + tri_idx
//   Decode: layer_idx = cell / N_tri,  tri_idx = cell % N_tri

DEF_PARTICLE_STRUCT(prism_ptc,
                    (Aperture::Scalar, x1, 0.0)
                    (Aperture::Scalar, x2, 0.0)
                    (Aperture::Scalar, x3, 0.0)
                    (Aperture::Scalar, p1, 0.0)
                    (Aperture::Scalar, p2, 0.0)
                    (Aperture::Scalar, p3, 0.0)
                    (Aperture::Scalar, E, 0.0)
                    (Aperture::Scalar, weight, 0.0)
                    (uint32_t, cell, empty_cell)
                    (uint64_t, id, 0)
                    (uint32_t, flag, 0));

namespace Aperture {

using prismatic_particles_t = particles_base<prism_ptc_buffer>;

// Cell encoding helpers
HD_INLINE uint32_t prism_cell_encode(int tri_idx, int layer_idx, int N_tri) {
  return static_cast<uint32_t>(layer_idx * N_tri + tri_idx);
}

HD_INLINE void prism_cell_decode(uint32_t cell, int N_tri,
                                 int& tri_idx, int& layer_idx) {
  layer_idx = cell / N_tri;
  tri_idx = cell % N_tri;
}

}  // namespace Aperture
