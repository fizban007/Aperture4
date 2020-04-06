#ifndef _VEC3_H_
#define _VEC3_H_

#include <cmath>
#include <iostream>
#include "core/cuda_control.h"

namespace Aperture {

///  Vectorized type as 3d vector
template <typename T>
struct Vec3 {
  T x, y, z;

  typedef Vec3<T> self_type;

  HOST_DEVICE Vec3()
      : x(static_cast<T>(0)),
        y(static_cast<T>(0)),
        z(static_cast<T>(0)) {}
  HOST_DEVICE Vec3(T xi, T yi, T zi) : x(xi), y(yi), z(zi) {}
  template <typename U>
  HOST_DEVICE Vec3(const Vec3<U>& other) {
    x = static_cast<T>(other.x);
    y = static_cast<T>(other.y);
    z = static_cast<T>(other.z);
  }

  Vec3(const self_type& other) = default;
  Vec3(self_type&& other) = default;

  HD_INLINE T& operator[](int idx) {
    if (idx == 0)
      return x;
    else if (idx == 1)
      return y;
    else
      return z;
  }

  HD_INLINE const T& operator[](int idx) const {
    if (idx == 0)
      return x;
    else if (idx == 1)
      return y;
    else
      return z;
  }

  HD_INLINE self_type& operator=(const self_type& other) {
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
  }

  HD_INLINE bool operator==(const self_type& other) const {
    return (x == other.x && y == other.y && z == other.z);
  }

  HD_INLINE bool operator!=(const self_type& other) const {
    return (x != other.x || y != other.y || z != other.z);
  }

  HD_INLINE self_type& operator+=(const self_type& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return (*this);
  }

  HD_INLINE self_type& operator-=(const self_type& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return (*this);
  }

  HD_INLINE self_type& operator+=(T n) {
    x += n;
    y += n;
    z += n;
    return (*this);
  }

  HD_INLINE self_type& operator-=(T n) {
    x -= n;
    y -= n;
    z -= n;
    return (*this);
  }

  HD_INLINE self_type& operator*=(T n) {
    x *= n;
    y *= n;
    z *= n;
    return (*this);
  }

  HD_INLINE self_type& operator/=(T n) {
    x /= n;
    y /= n;
    z /= n;
    return (*this);
  }

  HD_INLINE self_type operator+(const self_type& other) const {
    self_type tmp = *this;
    tmp += other;
    return tmp;
  }

  HD_INLINE self_type operator-(const self_type& other) const {
    self_type tmp = *this;
    tmp -= other;
    return tmp;
  }

  HD_INLINE self_type operator+(T n) const {
    self_type tmp = *this;
    tmp += n;
    return tmp;
  }

  HD_INLINE self_type operator-(T n) const {
    self_type tmp = *this;
    tmp -= n;
    return tmp;
  }

  HD_INLINE self_type operator*(T n) const {
    self_type tmp{x * n, y * n, z * n};
    return tmp;
  }

  HD_INLINE self_type operator/(T n) const {
    self_type tmp{x / n, y / n, z / n};
    return tmp;
  }

  HD_INLINE T dot(const self_type& other) const {
    return (x * other.x + y * other.y + z * other.z);
  }

  HD_INLINE self_type cross(const self_type& other) const {
    return Vec3<T>(y * other.z - z * other.y, z * other.x - x * other.z,
                   x * other.y - y * other.x);
  }

  HD_INLINE T length() const { return sqrt(this->dot(*this)); }

  HD_INLINE self_type& normalize() {
    double l = this->length();
    if (l > 1.0e-13) *this /= l;
    return *this;
  }

  HD_INLINE self_type normalize(T l) {
    this->normalize();
    (*this) *= l;
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const self_type& vec) {
    os << "( " << vec.x << ", " << vec.y << ", " << vec.z << " )";
    return os;
  }
};

////////////////////////////////////////////////////////////////////////////////
///  Class to store a multi-dimensional size
////////////////////////////////////////////////////////////////////////////////
struct Extent : public Vec3<uint32_t> {
  // Default initialize to 0 in the first size and 1 in the rest for
  // safety reasons
  HOST_DEVICE Extent() : Vec3(0, 1, 1) {}
  HOST_DEVICE explicit Extent(uint32_t w, uint32_t h = 1, uint32_t d = 1) : Vec3(w, h, d) {}
  HOST_DEVICE Extent(const Vec3<uint32_t>& vec) : Vec3(vec) {}

  HD_INLINE uint32_t& width() { return x; }
  HD_INLINE const uint32_t& width() const { return x; }
  HD_INLINE uint32_t& height() { return y; }
  HD_INLINE const uint32_t& height() const { return y; }
  HD_INLINE uint32_t& depth() { return z; }
  HD_INLINE const uint32_t& depth() const { return z; }

  HD_INLINE size_t size() const { return x * y * z; }

  // template <typename T>
  // cudaExtent cuda_ext(const T& t) const {
  //   return make_cudaExtent(x*sizeof(T), y, z);
  // }
  HD_INLINE int dim() const {
    if (y <= 1 && z <= 1) {
      return 1;
    } else if (z <= 1) {
      return 2;
    } else {
      return 3;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
///  Class to store a multi-dimensional index
////////////////////////////////////////////////////////////////////////////////
typedef Vec3<uint32_t> Index;

template <typename T>
HOST_DEVICE Vec3<T> operator*(const T& t, const Vec3<T>& v) {
  Vec3<T> result(v);
  result *= t;
  return result;
}

template <typename T>
HOST_DEVICE T
abs(const Vec3<T>& v) {
  // return v.length();
  return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

}  // namespace Aperture

#endif  // _VEC3_H_
