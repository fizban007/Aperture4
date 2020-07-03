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

#ifndef _MACRO_TRICKERY_H_
#define _MACRO_TRICKERY_H_

#include "core/cuda_control.h"
#include "visit_struct/visit_struct.hpp"
#include "visit_struct/visit_struct_intrusive.hpp"
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/expand.hpp>
#include <boost/preprocessor/seq/fold_right.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/to_tuple.hpp>

////////////////////////////////////////////////////////////////////
// These are helper macros for defining the particle data structure.
////////////////////////////////////////////////////////////////////

#define ESC(...) __VA_ARGS__

#define EXPAND_ELEMS(macro, elem)                                         \
  macro(BOOST_PP_TUPLE_ELEM(3, 0, elem), BOOST_PP_TUPLE_ELEM(3, 1, elem), \
        BOOST_PP_TUPLE_ELEM(3, 2, elem))

#define DEF_ENTRY_(type, name, dv) type name = (dv);
#define DEF_ENTRY(r, data, elem) EXPAND_ELEMS(DEF_ENTRY_, elem)

#define DEF_PTR_ENTRY_(type, name, dv) type* name;
#define DEF_PTR_ENTRY(r, data, elem) EXPAND_ELEMS(DEF_PTR_ENTRY_, elem)

// #define DEF_BUF_ENTRY_(type, name, dv) buffer<type, Model> name;
#define DEF_BUF_ENTRY_(type, name, dv) buffer<type> name;
#define DEF_BUF_ENTRY(r, data, elem) EXPAND_ELEMS(DEF_BUF_ENTRY_, elem)

#define PTR_ASSIGN_ENTRY_(result, ptr, name) result.name = name.ptr();
#define PTR_ASSIGN_ENTRY(r, data, elem)                                     \
  PTR_ASSIGN_ENTRY_(BOOST_PP_SEQ_ELEM(0, data), BOOST_PP_SEQ_ELEM(1, data), \
                    BOOST_PP_TUPLE_ELEM(3, 1, elem))

#define ASSIGN_ENTRY_(arr1, idx1, arr2, idx2, name) \
  arr1.name[idx1] = arr2.name[idx2];
#define ASSIGN_ENTRY(r, data, elem)                                     \
  ASSIGN_ENTRY_(BOOST_PP_SEQ_ELEM(0, data), BOOST_PP_SEQ_ELEM(1, data), \
                BOOST_PP_SEQ_ELEM(2, data), BOOST_PP_SEQ_ELEM(3, data), \
                BOOST_PP_TUPLE_ELEM(3, 1, elem))

#define ASSIGN_SINGLE_(arr1, idx1, single, name) arr1.name[idx1] = single.name;
#define ASSIGN_SINGLE(r, data, elem)                                     \
  ASSIGN_SINGLE_(BOOST_PP_SEQ_ELEM(0, data), BOOST_PP_SEQ_ELEM(1, data), \
                 BOOST_PP_SEQ_ELEM(2, data), BOOST_PP_TUPLE_ELEM(3, 1, elem))

#define GET_NAME_(type, name, dv) (name)
#define GET_NAME(r, data, elem) EXPAND_ELEMS(GET_NAME_, elem)
#define GET_TYPE_(type, name, dv) (type)
#define GET_TYPE(r, data, elem) EXPAND_ELEMS(GET_TYPE_, elem)
#define GET_TYPE_NAME_(type, name, dv) (type, name)
#define GET_TYPE_NAME(r, data, elem) EXPAND_ELEMS(GET_TYPE_NAME_, elem)
#define GET_PTR_NAME_(type, name, dv) (type*, name)
#define GET_PTR_NAME(r, data, elem) EXPAND_ELEMS(GET_PTR_NAME_, elem)

#define ADD_SIZEOF(s, state, elem) state + sizeof(elem)

#define GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_0(...) \
  ((__VA_ARGS__)) GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_1

#define GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_1(...) \
  ((__VA_ARGS__)) GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_0

#define GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_0_END
#define GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_1_END

// Double the parentheses of a Boost.PP sequence
// I.e. (a, b)(c, d) becomes ((a, b))((c, d))
#define GLK_PP_SEQ_DOUBLE_PARENS(seq) \
  BOOST_PP_CAT(GLK_PP_DETAIL_SEQ_DOUBLE_PARENS_0 seq, _END)

////////////////////////////////////////////////////////////////////

#include "core/buffer.hpp"

#define DEF_PARTICLE_STRUCT(name, content)                                     \
  namespace Aperture {                                                         \
                                                                               \
  struct single_##name##_t {                                                   \
    BOOST_PP_SEQ_FOR_EACH(DEF_ENTRY, _, GLK_PP_SEQ_DOUBLE_PARENS(content))     \
  };                                                                           \
                                                                               \
  struct name##_ptrs {                                                         \
    BOOST_PP_SEQ_FOR_EACH(DEF_PTR_ENTRY, _, GLK_PP_SEQ_DOUBLE_PARENS(content)) \
    enum {                                                                     \
      size = BOOST_PP_SEQ_FOLD_RIGHT(                                          \
          ADD_SIZEOF, 0,                                                       \
          BOOST_PP_SEQ_FOR_EACH(GET_TYPE, _,                                   \
                                GLK_PP_SEQ_DOUBLE_PARENS(content)))            \
    };                                                                         \
  };                                                                           \
                                                                               \
  struct name##_buffer {                                                       \
    typedef single_##name##_t single_type;                                     \
    typedef name##_ptrs ptrs_type;                                             \
                                                                               \
    BOOST_PP_SEQ_FOR_EACH(DEF_BUF_ENTRY, _, GLK_PP_SEQ_DOUBLE_PARENS(content)) \
                                                                               \
    name##_ptrs host_ptrs() {                                                  \
      name##_ptrs ptrs;                                                        \
      BOOST_PP_SEQ_FOR_EACH(PTR_ASSIGN_ENTRY, (ptrs)(host_ptr),                \
                            GLK_PP_SEQ_DOUBLE_PARENS(content))                 \
      return ptrs;                                                             \
    }                                                                          \
                                                                               \
    name##_ptrs dev_ptrs() {                                                   \
      name##_ptrs ptrs;                                                        \
      BOOST_PP_SEQ_FOR_EACH(PTR_ASSIGN_ENTRY, (ptrs)(dev_ptr),                 \
                            GLK_PP_SEQ_DOUBLE_PARENS(content))                 \
      return ptrs;                                                             \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <>                                                                  \
  struct ptc_array_type<single_##name##_t> {                                   \
    typedef name##_ptrs type;                                                  \
  };                                                                           \
                                                                               \
  HD_INLINE void assign_ptc(name##_ptrs array_1, size_t idx_1,                 \
                            name##_ptrs array_2, size_t idx_2){                \
      BOOST_PP_SEQ_FOR_EACH(ASSIGN_ENTRY, (array_1)(idx_1)(array_2)(idx_2),    \
                            GLK_PP_SEQ_DOUBLE_PARENS(content))}                \
                                                                               \
  HD_INLINE void assign_ptc(name##_ptrs array_1, size_t idx_1,                 \
                            single_##name##_t single) {                        \
    BOOST_PP_SEQ_FOR_EACH(ASSIGN_SINGLE, (array_1)(idx_1)(single),             \
                          GLK_PP_SEQ_DOUBLE_PARENS(content))                   \
  }                                                                            \
  }                                                                            \
  VISITABLE_STRUCT(                                                            \
      Aperture::name##_buffer,                                               \
      BOOST_PP_EXPAND(ESC BOOST_PP_SEQ_TO_TUPLE(BOOST_PP_SEQ_FOR_EACH(         \
          GET_NAME, _, GLK_PP_SEQ_DOUBLE_PARENS(content)))));                  \
  VISITABLE_STRUCT(                                                            \
      Aperture::single_##name##_t,                                             \
      BOOST_PP_EXPAND(ESC BOOST_PP_SEQ_TO_TUPLE(BOOST_PP_SEQ_FOR_EACH(         \
          GET_NAME, _, GLK_PP_SEQ_DOUBLE_PARENS(content)))));                  \
  VISITABLE_STRUCT(                                                            \
      Aperture::name##_ptrs,                                                   \
      BOOST_PP_EXPAND(ESC BOOST_PP_SEQ_TO_TUPLE(BOOST_PP_SEQ_FOR_EACH(         \
          GET_NAME, _, GLK_PP_SEQ_DOUBLE_PARENS(content)))))

  // template <MemType Model = MemType::host_only>                        \
    static constexpr MemType model() { return Model; }                     \
  VISITABLE_STRUCT(                                                            \
      Aperture::name##_buffer<Aperture::MemType::host_device>,             \
      BOOST_PP_EXPAND(ESC BOOST_PP_SEQ_TO_TUPLE(BOOST_PP_SEQ_FOR_EACH(         \
          GET_NAME, _, GLK_PP_SEQ_DOUBLE_PARENS(content)))));                  \
  VISITABLE_STRUCT(                                                            \
      Aperture::name##_buffer<Aperture::MemType::device_managed>,          \
      BOOST_PP_EXPAND(ESC BOOST_PP_SEQ_TO_TUPLE(BOOST_PP_SEQ_FOR_EACH(         \
          GET_NAME, _, GLK_PP_SEQ_DOUBLE_PARENS(content)))));                  \
  VISITABLE_STRUCT(                                                            \
      Aperture::name##_buffer<Aperture::MemType::device_only>,             \
      BOOST_PP_EXPAND(ESC BOOST_PP_SEQ_TO_TUPLE(BOOST_PP_SEQ_FOR_EACH(         \
          GET_NAME, _, GLK_PP_SEQ_DOUBLE_PARENS(content)))));
#endif  // _MACRO_TRICKERY_H_
