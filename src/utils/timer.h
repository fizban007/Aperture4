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

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

namespace Aperture {

class timer {
 public:
  timer() {}
  ~timer() {}

  static void stamp(const std::string& name = "");
  static void show_duration_since_stamp(const std::string& routine_name,
                                        const std::string& unit,
                                        const std::string& stamp_name = "");
  static float get_duration_since_stamp(const std::string& unit,
                                        const std::string& stamp_name = "");

  static std::unordered_map<std::string,
                            std::chrono::high_resolution_clock::time_point>
      t_stamps;
  static std::chrono::high_resolution_clock::time_point t_now;
};  // ----- end of class timer -----

}  // namespace Aperture
