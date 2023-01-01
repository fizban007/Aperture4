/*
 * Copyright (c) 2021 Alex Chen.
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
#include "core/typedefs_and_constants.h"

namespace Aperture {

// Return the index of the first element in the array that is not less than x
template <typename value_t>
HD_INLINE int
lower_bound(value_t x, const value_t* array, int size) {
  int mid;

  // Initialise starting index and
  // ending index
  int low = 0;
  int high = size;

  // Till high is less or equal to low
  while (low < high) {
    mid = low + (high - low) / 2;

    // If x is less than or equal
    // to arr[mid], then find in
    // left subarray
    if (x <= array[mid]) {
      high = mid;
    }

    // If X is greater arr[mid]
    // then find in right subarray
    else {
      low = mid + 1;
    }
  }

  // Return the lower_bound index
  return low;
}

// Return the index of the first element in the array that is greater than x
template <typename value_t>
HD_INLINE int
upper_bound(value_t x, const value_t* array, int size) {
  int mid;

  // Initialise starting index and
  // ending index
  int low = 0;
  int high = size;

  // Till high is less or equal to low
  while (low < high) {
    // Find the middle index
    mid = low + (high - low) / 2;

    // If X is greater than or equal
    // to arr[mid] then find
    // in right subarray
    if (x >= array[mid]) {
      low = mid + 1;
    }

    // If X is less than arr[mid]
    // then find in left subarray
    else {
      high = mid;
    }
  }

  // Return the upper_bound index
  return low;
}

}  // namespace Aperture
