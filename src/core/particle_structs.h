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

#ifndef __PARTICLE_STRUCTS_H_
#define __PARTICLE_STRUCTS_H_

#include "core/typedefs_and_constants.h"
#include "core/detail/macro_trickery.h"

namespace Aperture {

template <typename SingleType>
struct ptc_array_type;

}

// Here we define particle types through some macro magic. The macro
// DEF_PARTICLE_STRUCT takes in a name, and a sequence of triples, each
// one defines an entry in the particle struct. The macro defines several
// structs at the same time: `ptc_ptrs`, `ptc_buffer`, and `single_ptc_t`.
// For example, the following definition:
//
//     DEF_PARTICLE_STRUCT(ptc,
//                         (float, x1, 0.0)
//                         (float, x2, 0.0)
//                         (float, x3, 0.0));
//
// will define these structs:
//
//     struct single_ptc_t {
//       float x1 = 0.0;
//       float x2 = 0.0;
//       float x3 = 0.0;
//     };
//     struct ptc_ptrs {
//       float* x1;
//       float* x2;
//       float* x3;
//       enum { size = 3 * sizeof(float) };
//     };
//     struct ptc_buffer {
//       typedef single_ptc_t single_type;
//       typedef ptc_ptrs ptrs_type;
//       buffer<float> x1;
//       buffer<float> x2;
//       buffer<float> x3;
//
//       ptc_ptrs host_ptrs();
//       ptc_ptrs dev_ptrs();
//     };
//
// where `single_ptc_t` is a struct representing a single particle, `ptc_ptrs`
// is a struct of arrays that point to the actual data, and `ptc_buffer` is a
// struct that contains all the buffer objects that are responsible for the
// memory management. An integral constant `size` is defined in `ptc_ptrs` for
// MPI purposes.

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

// We use a 32-bit integer to give every particle a "flag". The highest 3 bits
// are used to represent the particle species (able to represent 8 different
// kinds of particles). This can be changed in the header file `enum_types.h`.
// The lower bits are given to pre-defined `PtcFlag`s in the `enum_types.h`
// header.

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
