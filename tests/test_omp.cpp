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

#include <iostream>
#include <omp.h>

int
main() {
  int n;

  int N = 100;
  int step = 6;
#pragma omp parallel for
  for (n = 0; n < N - step; n+=step) {
    std::cout << omp_get_thread_num() << ": " << n << std::endl;
// #pragma omp barrier
  }

  std::cout << "iterated " << (N / step) * step << std::endl;
  for (n = (N / step) * step; n < N; n++) {
    std::cout << n << std::endl;
  }

  std::cout << "Finally, n = " << n << std::endl;

  return 0;
}
