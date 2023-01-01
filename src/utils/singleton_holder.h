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

#include <cstdlib>
#include <memory>

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  This class adapts the SingletonHolder template described in Modern C++
///  Design by Andrei Alexandrescu. Only a simplified version is implemented,
///  since we don't really need to change creation, lifetime, and threading
///  policies.
////////////////////////////////////////////////////////////////////////////////
template <typename T>
class singleton_holder {
 public:
  template <typename... Args>
  static T& instance(Args&&... args) {
    init(args...);
    return *p_instance;
  }

  template <typename... Args>
  static inline void init(Args&&... args) {
    if (p_instance == nullptr) {
      p_instance = new T(std::forward<Args>(args)...);
      // p_instance.reset(new T(std::forward<Args>(args)...));
    }
  }

  static inline void kill_instance() {
    delete p_instance;
    p_instance = nullptr;
  }

  singleton_holder() = delete;
  singleton_holder(const singleton_holder<T>&) = delete;
  singleton_holder(singleton_holder<T>&&) = delete;
  singleton_holder<T>& operator=(const singleton_holder<T>&) = delete;
  singleton_holder<T>& operator=(singleton_holder<T>&&) = delete;
  ~singleton_holder() = delete;

 private:
  // static std::unique_ptr<T> p_instance;
  static T* p_instance;
};

template <typename T>
T* singleton_holder<T>::p_instance = nullptr;
// std::unique_ptr<T> singleton_holder<T>::p_instance = nullptr;

}  // namespace Aperture
