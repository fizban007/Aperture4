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

#include <vector>
#include "utils/logger.h"
// #include "core/particles.h"

using namespace Aperture;

// The goal here is to define two systems. Each of them has some requirement in
// data components. Then build a `data` object that contain the required
// components.
//
struct Component1 {
  std::vector<float> v;
  float* ptr_v;

  void init() {
    ptr_v = v.data();
  }
};

struct Component2 {
  std::vector<float> x;
  float* ptr_x;

  void init() {
    ptr_x = x.data();
  }
};

struct Component3 {
  std::vector<uint32_t> cell;
};

// template <class ... Ts>
// struct Mixed : public Ts... {
//   void init() {
//     (Ts::init(), ...);
//   }
// };

template<template<typename> typename... Skills>
class X : public Skills<X<Skills...>>...
{
public:
    void basicMethod();
};

template <typename T, typename ... Ts, typename Func>
void variadic_loop(Func f, T& t, Ts& ... ts) {
  f(t);
  variadic_loop(f, ts...);
}

template <typename T, typename Func>
void variadic_loop(Func f, T& t) {
  f(t);
}

int main(int argc, char *argv[]) {
  return 0;
}
