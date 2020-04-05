#ifndef __PARTICLE_STRUCTS_H_
#define __PARTICLE_STRUCTS_H_

#include "typedefs_and_constants.h"
#include "detail/macro_trickery.h"

namespace Aperture {

template <typename SingleType>
struct ptc_array_type;

}

// Here we define particle types through some macro magic. The macro
// DEF_PARTICLE_STRUCT takes in a name, and a sequence of triples, each
// one defines an entry in the particle struct. The macro defines two
// structs at the same time: `particle_data` and `single_particle_t`.
// For example, the following definition:
//
//     DEF_PARTICLE_STRUCT(particle,
//                         (float, x1, 0.0)
//                         (float, x2, 0.0)
//                         (float, x3, 0.0));
//
// will define two structs:
//
//     struct single_particle_t {
//       float x1 = 0.0;
//       float x2 = 0.0;
//       float x3 = 0.0;
//     };
//     struct particle_data {
//       float* x1;
//       float* x2;
//       float* x3;
//       enum { size = 3 * sizeof(float) };
//       single_particle_t operator[](size_t idx) const;
//     };
//
// where `single_particle_t` is a struct representing a single particle,
// and `particle_data` is a struct of arrays that contain the actual
// data. An integral constant `size` is defined in `particle_data` for
// MPI purposes, and an index operator is defined to easily read a
// single particle.

DEF_PARTICLE_STRUCT(ptc,
                    (Aperture::Pos_t, x1, 0.0)
                    (Aperture::Pos_t, x2, 0.0)
                    (Aperture::Pos_t, x3, 0.0)
                    (Aperture::Scalar, p1, 0.0)
                    (Aperture::Scalar, p2, 0.0)
                    (Aperture::Scalar, p3, 0.0)
                    (Aperture::Scalar, E, 0.0)
                    (Aperture::Scalar, weight, 0.0)
                    (uint32_t, cell, empty_cell)
                    (uint64_t, id, 0)
                    (uint32_t, flag, 0));

// We use a 32-bit integer to give every particle a "flag". The highest
// 3 bits are used to represent the particle species (able to represent
// 8 different kinds of particles). The lower bits are given to
// pre-defined `ParticleFlag`s in the `enum_types.h` header.

DEF_PARTICLE_STRUCT(ph,
                    (Aperture::Pos_t, x1, 0.0)
                    (Aperture::Pos_t, x2, 0.0)
                    (Aperture::Pos_t, x3, 0.0)
                    (Aperture::Scalar, p1, 0.0)
                    (Aperture::Scalar, p2, 0.0)
                    (Aperture::Scalar, p3, 0.0)
                    (Aperture::Scalar, E, 0.0)
                    (Aperture::Scalar, weight, 0.0)
                    (Aperture::Scalar, path_left, 0.0)
                    (uint32_t, cell, empty_cell)
                    (uint64_t, id, 0)
                    (uint32_t, flag, 0));


#endif // __PARTICLE_STRUCTS_H_
