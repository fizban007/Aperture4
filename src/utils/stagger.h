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

#pragma once

#include "core/cuda_control.h"

namespace Aperture {

class stagger_t {
 private:
  unsigned char stagger;

 public:
  /// Default constructor, initialize the stagger in each direction to be 0.
  HOST_DEVICE stagger_t() : stagger(0) {}

  /// Constructor using an unsigned char. The recommended way to use this is to
  /// do
  ///
  ///     stagger_t st(0b001);
  ///
  /// This will initialize the lowest bit to 1, and upper bits to 0. This is
  /// means staggered in x, but not staggered in y and z directions.
  HOST_DEVICE stagger_t(unsigned char s) : stagger(s){};

  /// Copy constructor, simply copy the stagger of the given input.
  HOST_DEVICE stagger_t(const stagger_t& s) : stagger(s.stagger) {}

  /// Assignment, copy the stagger of the input.
  HD_INLINE stagger_t& operator=(const stagger_t& s) {
    stagger = s.stagger;
    return *this;
  }

  /// Assignment, using an unsigned char. The recommended way to use this is to
  /// do
  ///
  ///     st = 0b001;
  ///
  /// This will initialize the lowest bit to 1, and upper bits to 0. This is
  /// means staggered in x, but not staggered in y and z directions.
  HD_INLINE stagger_t& operator=(const unsigned char s) {
    stagger = s;
    return *this;
  }

  /// Subscript operator. Use this to take the stagger of a given direction. For
  /// example,
  ///
  ///     stagger_t st(0b110);
  ///     assert(st[0] == 1);
  ///     assert(st[1] == 1);
  ///     assert(st[2] == 1);
  ///
  /// Since this is inlined and bit shifts are cheap, feel free to use this
  /// inside a kernel.
  HD_INLINE int operator[](int i) const { return (stagger >> i) & 1UL; }

  /// Set the given bit to true or false.
  HD_INLINE void set_bit(int bit, bool i) {
    unsigned long x = !!i;
    stagger ^= (-x ^ stagger) & (1UL << bit);
  }

  /// Flip the stagger of a given direcion.
  HD_INLINE void flip(int n) { stagger ^= (1UL << n); }

  /// Return the complement of this stagger configuration.
  HD_INLINE stagger_t complement() { return stagger_t(~stagger); }
};

}  // namespace Aperture
