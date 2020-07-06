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

#include "catch.hpp"
#include "framework/config.h"
#include "utils/type_traits.hpp"

using namespace Aperture;

TEST_CASE("Making multi_array", "[config]") {
  // Since Config is a purely compile-time configurator, this test file is meant
  // to contain no output. The test passes as long as this can compile.
  typedef Config<3> Conf;
  auto array = Conf::make_multi_array({32, 32, 32});
}

TEST_CASE("Multi array is indexable", "[type_traits]") {
  typedef Config<3> Conf;
  typedef multi_array_cref<float, 3, idx_col_major_t<3>> mcref;
  static_assert(
      is_indexable<typename Conf::multi_array_t>::value);
  static_assert(
      is_const_indexable<typename Conf::multi_array_t>::value);
  static_assert(
      is_indexable<typename Conf::ndptr_t>::value);
  static_assert(
      is_const_indexable<typename Conf::ndptr_const_t>::value);
}
