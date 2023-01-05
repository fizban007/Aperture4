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

#include "gsl/pointers"

namespace Aperture {

template <typename T>
struct nonown_ptr {
 private:
  T* p = nullptr;

 public:
  nonown_ptr() {}
  nonown_ptr(std::nullptr_t ptr) : p(nullptr) {}
  explicit nonown_ptr(gsl::not_null<T*>& ptr) : p(ptr.get()) {}

  explicit nonown_ptr(T* ptr) : p(ptr) {}
  nonown_ptr(const nonown_ptr<T>& other) : p(other.p) {}
  nonown_ptr(nonown_ptr<T>&& other) : p(other.p) {}

  nonown_ptr<T>& operator=(const nonown_ptr<T>& other) = default;
  nonown_ptr<T>& operator=(nonown_ptr<T>&& other) = default;

  bool operator==(std::nullptr_t ptr) const { return (p == nullptr); }
  bool operator!=(std::nullptr_t ptr) const { return (p != nullptr); }

  void reset(T* ptr) { p = ptr; }
  void release() { p = nullptr; }

  T& operator*() { return *p; }
  const T& operator*() const { return *p; }

  T* operator->() { return p; }
  const T* operator->() const { return p; }

  const T* get() const { return p; }
};

}  // namespace Aperture
