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

#include "catch2/catch_all.hpp"
#include "framework/params_store.h"

using namespace Aperture;

TEST_CASE("Using the parameter store in cu", "[param_store]") {
  params_store store;

  store.add("p", 4.0);
  store.add("n", 42l);

  int n = store.get_as<int64_t>("n", 1);
  REQUIRE(n == 42);
  // REQUIRE(store.get<int>("n", 1) == 4);
  // REQUIRE(store.get<std::string>("name", "") == "");
  // REQUIRE(store.get<std::string>("p", "") == "");
}
