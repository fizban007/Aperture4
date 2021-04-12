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

/*
The MIT License(MIT)

Copyright(c) 2015 Alexandre Avenel

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files(the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef MORTON_H
#define MORTON_H

#include <cstdint>
#include <array>
#include <assert.h>
#include "core/cuda_control.h"
#include "core/constant_mem.h"

#if _MSC_VER
#include <immintrin.h>
#endif

#if (__GNUC__ && !defined(__CUDACC__))
#include <x86intrin.h>
#endif

namespace Aperture {

/*
BMI2 (Bit Manipulation Instruction Set 2) is a special set of instructions available for intel core i5, i7 (since Haswell architecture) and Xeon E3.
Some instructions are not available for Microsoft Visual Studio older than 2013.
*/
#if _MSC_VER
#define USE_BMI2
#endif

//mortonkey(x+1) = (mortonkey(x) - MAXMORTONKEY) & MAXMORTONKEY
// static const uint32_t morton3dLUT[256] =
constexpr uint32_t morton3dLUT[256] =
{
  0x00000000, 0x00000001, 0x00000008, 0x00000009, 0x00000040, 0x00000041, 0x00000048, 0x00000049,
  0x00000200, 0x00000201, 0x00000208, 0x00000209, 0x00000240, 0x00000241, 0x00000248, 0x00000249,
  0x00001000, 0x00001001, 0x00001008, 0x00001009, 0x00001040, 0x00001041, 0x00001048, 0x00001049,
  0x00001200, 0x00001201, 0x00001208, 0x00001209, 0x00001240, 0x00001241, 0x00001248, 0x00001249,
  0x00008000, 0x00008001, 0x00008008, 0x00008009, 0x00008040, 0x00008041, 0x00008048, 0x00008049,
  0x00008200, 0x00008201, 0x00008208, 0x00008209, 0x00008240, 0x00008241, 0x00008248, 0x00008249,
  0x00009000, 0x00009001, 0x00009008, 0x00009009, 0x00009040, 0x00009041, 0x00009048, 0x00009049,
  0x00009200, 0x00009201, 0x00009208, 0x00009209, 0x00009240, 0x00009241, 0x00009248, 0x00009249,
  0x00040000, 0x00040001, 0x00040008, 0x00040009, 0x00040040, 0x00040041, 0x00040048, 0x00040049,
  0x00040200, 0x00040201, 0x00040208, 0x00040209, 0x00040240, 0x00040241, 0x00040248, 0x00040249,
  0x00041000, 0x00041001, 0x00041008, 0x00041009, 0x00041040, 0x00041041, 0x00041048, 0x00041049,
  0x00041200, 0x00041201, 0x00041208, 0x00041209, 0x00041240, 0x00041241, 0x00041248, 0x00041249,
  0x00048000, 0x00048001, 0x00048008, 0x00048009, 0x00048040, 0x00048041, 0x00048048, 0x00048049,
  0x00048200, 0x00048201, 0x00048208, 0x00048209, 0x00048240, 0x00048241, 0x00048248, 0x00048249,
  0x00049000, 0x00049001, 0x00049008, 0x00049009, 0x00049040, 0x00049041, 0x00049048, 0x00049049,
  0x00049200, 0x00049201, 0x00049208, 0x00049209, 0x00049240, 0x00049241, 0x00049248, 0x00049249,
  0x00200000, 0x00200001, 0x00200008, 0x00200009, 0x00200040, 0x00200041, 0x00200048, 0x00200049,
  0x00200200, 0x00200201, 0x00200208, 0x00200209, 0x00200240, 0x00200241, 0x00200248, 0x00200249,
  0x00201000, 0x00201001, 0x00201008, 0x00201009, 0x00201040, 0x00201041, 0x00201048, 0x00201049,
  0x00201200, 0x00201201, 0x00201208, 0x00201209, 0x00201240, 0x00201241, 0x00201248, 0x00201249,
  0x00208000, 0x00208001, 0x00208008, 0x00208009, 0x00208040, 0x00208041, 0x00208048, 0x00208049,
  0x00208200, 0x00208201, 0x00208208, 0x00208209, 0x00208240, 0x00208241, 0x00208248, 0x00208249,
  0x00209000, 0x00209001, 0x00209008, 0x00209009, 0x00209040, 0x00209041, 0x00209048, 0x00209049,
  0x00209200, 0x00209201, 0x00209208, 0x00209209, 0x00209240, 0x00209241, 0x00209248, 0x00209249,
  0x00240000, 0x00240001, 0x00240008, 0x00240009, 0x00240040, 0x00240041, 0x00240048, 0x00240049,
  0x00240200, 0x00240201, 0x00240208, 0x00240209, 0x00240240, 0x00240241, 0x00240248, 0x00240249,
  0x00241000, 0x00241001, 0x00241008, 0x00241009, 0x00241040, 0x00241041, 0x00241048, 0x00241049,
  0x00241200, 0x00241201, 0x00241208, 0x00241209, 0x00241240, 0x00241241, 0x00241248, 0x00241249,
  0x00248000, 0x00248001, 0x00248008, 0x00248009, 0x00248040, 0x00248041, 0x00248048, 0x00248049,
  0x00248200, 0x00248201, 0x00248208, 0x00248209, 0x00248240, 0x00248241, 0x00248248, 0x00248249,
  0x00249000, 0x00249001, 0x00249008, 0x00249009, 0x00249040, 0x00249041, 0x00249048, 0x00249049,
  0x00249200, 0x00249201, 0x00249208, 0x00249209, 0x00249240, 0x00249241, 0x00249248, 0x00249249
};

constexpr uint32_t morton2dLUT[256] =
{ 0x0000, 0x0001, 0x0004, 0x0005, 0x0010, 0x0011, 0x0014, 0x0015,
  0x0040, 0x0041, 0x0044, 0x0045, 0x0050, 0x0051, 0x0054, 0x0055,
  0x0100, 0x0101, 0x0104, 0x0105, 0x0110, 0x0111, 0x0114, 0x0115,
  0x0140, 0x0141, 0x0144, 0x0145, 0x0150, 0x0151, 0x0154, 0x0155,
  0x0400, 0x0401, 0x0404, 0x0405, 0x0410, 0x0411, 0x0414, 0x0415,
  0x0440, 0x0441, 0x0444, 0x0445, 0x0450, 0x0451, 0x0454, 0x0455,
  0x0500, 0x0501, 0x0504, 0x0505, 0x0510, 0x0511, 0x0514, 0x0515,
  0x0540, 0x0541, 0x0544, 0x0545, 0x0550, 0x0551, 0x0554, 0x0555,
  0x1000, 0x1001, 0x1004, 0x1005, 0x1010, 0x1011, 0x1014, 0x1015,
  0x1040, 0x1041, 0x1044, 0x1045, 0x1050, 0x1051, 0x1054, 0x1055,
  0x1100, 0x1101, 0x1104, 0x1105, 0x1110, 0x1111, 0x1114, 0x1115,
  0x1140, 0x1141, 0x1144, 0x1145, 0x1150, 0x1151, 0x1154, 0x1155,
  0x1400, 0x1401, 0x1404, 0x1405, 0x1410, 0x1411, 0x1414, 0x1415,
  0x1440, 0x1441, 0x1444, 0x1445, 0x1450, 0x1451, 0x1454, 0x1455,
  0x1500, 0x1501, 0x1504, 0x1505, 0x1510, 0x1511, 0x1514, 0x1515,
  0x1540, 0x1541, 0x1544, 0x1545, 0x1550, 0x1551, 0x1554, 0x1555,
  0x4000, 0x4001, 0x4004, 0x4005, 0x4010, 0x4011, 0x4014, 0x4015,
  0x4040, 0x4041, 0x4044, 0x4045, 0x4050, 0x4051, 0x4054, 0x4055,
  0x4100, 0x4101, 0x4104, 0x4105, 0x4110, 0x4111, 0x4114, 0x4115,
  0x4140, 0x4141, 0x4144, 0x4145, 0x4150, 0x4151, 0x4154, 0x4155,
  0x4400, 0x4401, 0x4404, 0x4405, 0x4410, 0x4411, 0x4414, 0x4415,
  0x4440, 0x4441, 0x4444, 0x4445, 0x4450, 0x4451, 0x4454, 0x4455,
  0x4500, 0x4501, 0x4504, 0x4505, 0x4510, 0x4511, 0x4514, 0x4515,
  0x4540, 0x4541, 0x4544, 0x4545, 0x4550, 0x4551, 0x4554, 0x4555,
  0x5000, 0x5001, 0x5004, 0x5005, 0x5010, 0x5011, 0x5014, 0x5015,
  0x5040, 0x5041, 0x5044, 0x5045, 0x5050, 0x5051, 0x5054, 0x5055,
  0x5100, 0x5101, 0x5104, 0x5105, 0x5110, 0x5111, 0x5114, 0x5115,
  0x5140, 0x5141, 0x5144, 0x5145, 0x5150, 0x5151, 0x5154, 0x5155,
  0x5400, 0x5401, 0x5404, 0x5405, 0x5410, 0x5411, 0x5414, 0x5415,
  0x5440, 0x5441, 0x5444, 0x5445, 0x5450, 0x5451, 0x5454, 0x5455,
  0x5500, 0x5501, 0x5504, 0x5505, 0x5510, 0x5511, 0x5514, 0x5515,
  0x5540, 0x5541, 0x5544, 0x5545, 0x5550, 0x5551, 0x5554, 0x5555
};

constexpr uint64_t x3_mask = 0x4924924924924924; // 0b...00100100
constexpr uint64_t y3_mask = 0x2492492492492492; // 0b...10010010
constexpr uint64_t z3_mask = 0x9249249249249249; // 0b...01001001
constexpr uint64_t xy3_mask = x3_mask | y3_mask;
constexpr uint64_t xz3_mask = x3_mask | z3_mask;
constexpr uint64_t yz3_mask = y3_mask | z3_mask;

constexpr uint64_t x2_mask = 0xAAAAAAAAAAAAAAAA; //0b...10101010
constexpr uint64_t y2_mask = 0x5555555555555555; //0b...01010101

template<class T = uint64_t>
struct morton3d
{
public:
	T key;

public:

	HD_INLINE explicit morton3d() : key(0) {};
	HD_INLINE explicit morton3d(const T _key) : key(_key) {};

	/* If BMI2 intrinsics are not available, we rely on a look up table of precomputed morton codes.
	Ref : http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/ */
	HD_INLINE morton3d(const uint32_t x, const uint32_t y, const uint32_t z) : key(0) {
    // Honestly the morton3dLUT_dev array needs to be initialized, but the
    // compiler seems to be able to inline the correct array, so that it is not
    // really necessary.
#ifdef __CUDA_ARCH__
		key = morton3dLUT_dev[(x >> 16) & 0xFF] << 2 |
			morton3dLUT_dev[(y >> 16) & 0xFF] << 1 |
			morton3dLUT_dev[(z >> 16) & 0xFF];
		key = key << 24 |
			morton3dLUT_dev[(x >> 8) & 0xFF] << 2 |
			morton3dLUT_dev[(y >> 8) & 0xFF] << 1 |
			morton3dLUT_dev[(z >> 8) & 0xFF];
		key = key << 24 |
			morton3dLUT_dev[x & 0xFF] << 2 |
			morton3dLUT_dev[y & 0xFF] << 1 |
			morton3dLUT_dev[z & 0xFF];
#else
#ifdef USE_BMI2
		key = static_cast<T>(_pdep_u64(z, z3_mask) | _pdep_u64(y, y3_mask) | _pdep_u64(x, x3_mask));
#else
		key = morton3dLUT[(x >> 16) & 0xFF] << 2 |
			morton3dLUT[(y >> 16) & 0xFF] << 1 |
			morton3dLUT[(z >> 16) & 0xFF];
		key = key << 24 |
			morton3dLUT[(x >> 8) & 0xFF] << 2 |
			morton3dLUT[(y >> 8) & 0xFF] << 1 |
			morton3dLUT[(z >> 8) & 0xFF];
		key = key << 24 |
			morton3dLUT[x & 0xFF] << 2 |
			morton3dLUT[y & 0xFF] << 1 |
        morton3dLUT[z & 0xFF];
#endif
#endif
	}

	HD_INLINE void decode(uint64_t& x, uint64_t& y, uint64_t& z) const
	{
#if (defined(USE_BMI2) && !defined(__CUDA_ARCH__))
		x = _pext_u64(this->key, x3_mask);
		y = _pext_u64(this->key, y3_mask);
		z = _pext_u64(this->key, z3_mask);
#else
		x = compactBits(this->key >> 2);
		y = compactBits(this->key >> 1);
		z = compactBits(this->key);
#endif
	}

	//Binary operators
	HD_INLINE bool operator==(const morton3d m1) const
	{
		return this->key == m1.key;
	}

	HD_INLINE bool operator!=(const morton3d m1) const
	{
		return !operator==(m1);
	}

	HD_INLINE morton3d operator|(const morton3d m1) const
	{
		return morton3d<T>(this->key | m1.key);
	}

	HD_INLINE morton3d operator&(const morton3d m1) const
	{
		return morton3d<T>(this->key & m1.key);
	}

	HD_INLINE morton3d operator >> (const uint64_t d) const
	{
		assert(d < 22);
		return morton3d<T>(this->key >> (3 * d));
	}

	HD_INLINE morton3d operator<<(const uint64_t d) const
	{
		assert(d < 22);
		return morton3d<T>(this->key << (3 * d));
	}

	HD_INLINE void operator+=(const morton3d<T> m1)
	{
		T x_sum = (this->key | yz3_mask) + (m1.key & x3_mask);
		T y_sum = (this->key | xz3_mask) + (m1.key & y3_mask);
		T z_sum = (this->key | xy3_mask) + (m1.key & z3_mask);
		this->key = ((x_sum & x3_mask) | (y_sum & y3_mask) | (z_sum & z3_mask));
	}

	HD_INLINE void operator-=(const morton3d<T> m1)
	{
		T x_diff = (this->key & x3_mask) - (m1.key & x3_mask);
		T y_diff = (this->key & y3_mask) - (m1.key & y3_mask);
		T z_diff = (this->key & z3_mask) - (m1.key & z3_mask);
		this->key = ((x_diff & x3_mask) | (y_diff & y3_mask) | (z_diff & z3_mask));
	}

	/* Fast encode of morton3 code when BMI2 instructions aren't available.
	This does not work for values greater than 256.

	This function takes roughly the same time as a full encode (64 bits) using BMI2 intrinsic.*/
	static HD_INLINE morton3d morton3d_256(const uint32_t x, const uint32_t y, const uint32_t z)
	{
		assert(x < 256 && y < 256 && z < 256);
		uint64_t key = morton3dLUT[x] << 2 |
			morton3dLUT[y] << 1 |
			morton3dLUT[z];
		return morton3d(key);
	}

	/* Increment X part of a morton3 code (xyz interleaving)
	   morton3(4,5,6).incX() == morton3(5,5,6);

	   Ref : http://bitmath.blogspot.fr/2012/11/tesseral-arithmetic-useful-snippets.html */
	HD_INLINE morton3d incX(int n = 1) const
	{
#ifdef __CUDA_ARCH__
		const T x_sum = static_cast<T>((this->key | yz3_mask) + (morton3dLUT_dev[n] << 2));
#else
		const T x_sum = static_cast<T>((this->key | yz3_mask) + (morton3dLUT[n] << 2));
#endif
		return morton3d<T>((x_sum & x3_mask) | (this->key & yz3_mask));
	}

	HD_INLINE morton3d incY(int n = 1) const
	{
#ifdef __CUDA_ARCH__
		const T y_sum = static_cast<T>((this->key | xz3_mask) + (morton3dLUT_dev[n] << 1));
#else
		const T y_sum = static_cast<T>((this->key | xz3_mask) + (morton3dLUT[n] << 1));
#endif
		return morton3d<T>((y_sum & y3_mask) | (this->key & xz3_mask));
	}

	HD_INLINE morton3d incZ(int n = 1) const
	{
#ifdef __CUDA_ARCH__
		const T z_sum = static_cast<T>((this->key | xy3_mask) + morton3dLUT_dev[n]);
#else
		const T z_sum = static_cast<T>((this->key | xy3_mask) + morton3dLUT[n]);
#endif
		return morton3d<T>((z_sum & z3_mask) | (this->key & xy3_mask));
	}

	/* Decrement X part of a morton3 code (xyz interleaving)
	   morton3(4,5,6).decX() == morton3(3,5,6); */
	HD_INLINE morton3d decX(int n = 1) const
	{
#ifdef __CUDA_ARCH__
		const T x_diff = (this->key & x3_mask) - (morton3dLUT_dev[n] << 2);
#else
		const T x_diff = (this->key & x3_mask) - (morton3dLUT[n] << 2);
#endif
		return morton3d<T>((x_diff & x3_mask) | (this->key & yz3_mask));
	}

	HD_INLINE morton3d decY(int n = 1) const
	{
#ifdef __CUDA_ARCH__
		const T y_diff = (this->key & y3_mask) - (morton3dLUT_dev[n] << 1);
#else
		const T y_diff = (this->key & y3_mask) - (morton3dLUT[n] << 1);
#endif
		return morton3d<T>((y_diff & y3_mask) | (this->key & xz3_mask));
	}

	HD_INLINE morton3d decZ(int n = 1) const
	{
#ifdef __CUDA_ARCH__
		const T z_diff = (this->key & z3_mask) - morton3dLUT_dev[n];
#else
		const T z_diff = (this->key & z3_mask) - morton3dLUT[n];
#endif
		return morton3d<T>((z_diff & z3_mask) | (this->key & xy3_mask));
	}


	/*
	  min(morton3(4,5,6), morton3(8,3,7)) == morton3(4,3,6);
	  Ref : http://asgerhoedt.dk/?p=276
	*/
	static HD_INLINE morton3d min(const morton3d lhs, const morton3d rhs)
	{
		T lhsX = lhs.key & x3_mask;
		T rhsX = rhs.key & x3_mask;
		T lhsY = lhs.key & y3_mask;
		T rhsY = rhs.key & y3_mask;
		T lhsZ = lhs.key & z3_mask;
		T rhsZ = rhs.key & z3_mask;
		return morton3d<T>(std::min(lhsX, rhsX) + std::min(lhsY, rhsY) + std::min(lhsZ, rhsZ));
	}

	/*
	  max(morton3(4,5,6), morton3(8,3,7)) == morton3(8,5,7);
	*/
	static HD_INLINE morton3d max(const morton3d lhs, const morton3d rhs)
	{
		T lhsX = lhs.key & x3_mask;
		T rhsX = rhs.key & x3_mask;
		T lhsY = lhs.key & y3_mask;
		T rhsY = rhs.key & y3_mask;
		T lhsZ = lhs.key & z3_mask;
		T rhsZ = rhs.key & z3_mask;
		return morton3d<T>(std::max(lhsX, rhsX) + std::max(lhsY, rhsY) + std::max(lhsZ, rhsZ));
	}

private:
	HD_INLINE uint64_t compactBits(uint64_t n) const
	{
		n &= 0x1249249249249249;
		n = (n ^ (n >> 2)) & 0x30c30c30c30c30c3;
		n = (n ^ (n >> 4)) & 0xf00f00f00f00f00f;
		n = (n ^ (n >> 8)) & 0x00ff0000ff0000ff;
		n = (n ^ (n >> 16)) & 0x00ff00000000ffff;
		n = (n ^ (n >> 32)) & 0x1fffff;
		return n;
	}

};

/* Add two morton keys (xyz interleaving)
  morton3(4,5,6) + morton3(1,2,3) == morton3(5,7,9);*/
template<class T>
HD_INLINE morton3d<T> operator+(const morton3d<T> m1, const morton3d<T> m2)
{
	T x_sum = (m1.key | yz3_mask) + (m2.key & x3_mask);
	T y_sum = (m1.key | xz3_mask) + (m2.key & y3_mask);
	T z_sum = (m1.key | xy3_mask) + (m2.key & z3_mask);
	return morton3d<T>((x_sum & x3_mask) | (y_sum & y3_mask) | (z_sum & z3_mask));
}

/* Substract two morton keys (xyz interleaving)
   morton3(4,5,6) - morton3(1,2,3) == morton3(3,3,3);*/
template<class T>
HD_INLINE morton3d<T> operator-(const morton3d<T> m1, const morton3d<T> m2)
{
	T x_diff = (m1.key & x3_mask) - (m2.key & x3_mask);
	T y_diff = (m1.key & y3_mask) - (m2.key & y3_mask);
	T z_diff = (m1.key & z3_mask) - (m2.key & z3_mask);
	return morton3d<T>((x_diff & x3_mask) | (y_diff & y3_mask) | (z_diff & z3_mask));
}

template<class T>
HD_INLINE bool operator< (const morton3d<T>& lhs, const morton3d<T>& rhs)
{
	return (lhs.key) < (rhs.key);
}

template<class T>
HD_INLINE bool operator> (const morton3d<T>& lhs, const morton3d<T>& rhs)
{
	return (lhs.key) > (rhs.key);
}

template<class T>
HD_INLINE bool operator>= (const morton3d<T>& lhs, const morton3d<T>& rhs)
{
	return (lhs.key) >= (rhs.key);
}

template<class T>
HD_INLINE bool operator<= (const morton3d<T>& lhs, const morton3d<T>& rhs)
{
	return (lhs.key) <= (rhs.key);
}

template<class T>
std::ostream& operator<<(std::ostream& os, const morton3d<T>& m)
{
	uint64_t x, y, z;
	m.decode(x, y, z);
	os << m.key << ": " << x << ", " << y << ", " << z;
	return os;
}

template<class T = uint64_t>
struct morton2d
{
public:
	T key;

public:

	HD_INLINE morton2d() : key(0) {};
	HD_INLINE explicit morton2d(T _key) : key(_key) {};

	/* If BMI2 intrinsics are not available, we rely on a look up table of precomputed morton codes. */
	HD_INLINE morton2d(const uint32_t x, const uint32_t y) : key(0) {

#ifdef __CUDA_ARCH__
		key = morton2dLUT_dev[(x >> 24) & 0xFF] << 1 |
			morton2dLUT_dev[(y >> 24) & 0xFF];
		key = key << 16 |
			morton2dLUT_dev[(x >> 16) & 0xFF] << 1 |
			morton2dLUT_dev[(y >> 16) & 0xFF];
		key = key << 16 |
			morton2dLUT_dev[(x >> 8) & 0xFF] << 1 |
			morton2dLUT_dev[(y >> 8) & 0xFF];
		key = key << 16 |
			morton2dLUT_dev[x & 0xFF] << 1 |
        morton2dLUT_dev[y & 0xFF];
#else
#ifdef USE_BMI2
		key = static_cast<T>(_pdep_u64(y, y2_mask) | _pdep_u64(x, x2_mask));
#else
		key = morton2dLUT[(x >> 24) & 0xFF] << 1 |
			morton2dLUT[(y >> 24) & 0xFF];
		key = key << 16 |
			morton2dLUT[(x >> 16) & 0xFF] << 1 |
			morton2dLUT[(y >> 16) & 0xFF];
		key = key << 16 |
			morton2dLUT[(x >> 8) & 0xFF] << 1 |
			morton2dLUT[(y >> 8) & 0xFF];
		key = key << 16 |
			morton2dLUT[x & 0xFF] << 1 |
        morton2dLUT[y & 0xFF];
#endif
#endif
	}

	HD_INLINE constexpr void decode(uint64_t& x, uint64_t& y) const
	{
#if defined(USE_BMI2) && !defined(__CUDA_ARCH__)
		x = _pext_u64(this->key, x2_mask);
		y = _pext_u64(this->key, y2_mask);
#else
		x = compactBits(this->key >> 1);
		y = compactBits(this->key);
#endif
	}

	//Binary operators
	HD_INLINE bool operator==(const morton2d m1) const
	{
		return this->key == m1.key;
	}

	HD_INLINE bool operator!=(const morton2d m1) const
	{
		return !operator==(m1);
	}

	HD_INLINE morton2d operator|(const morton2d m1) const
	{
		return morton2d<T>(this->key | m1.key);
	}

	HD_INLINE morton2d operator&(const morton2d m1) const
	{
		return morton2d<T>(this->key & m1.key);
	}

	HD_INLINE morton2d operator >> (const uint64_t d) const
	{
		return morton2d<T>(this->key >> (2 * d));
	}

	HD_INLINE morton2d operator<<(const uint64_t d) const
	{
		return morton2d<T>(this->key << (2 * d));
	}

	HD_INLINE void operator+=(const morton2d<T> rhs)
	{
		T x_sum = (this->key | y2_mask) + (rhs.key & x2_mask);
		T y_sum = (this->key | x2_mask) + (rhs.key & y2_mask);
		this->key = (x_sum & x2_mask) | (y_sum & y2_mask);
	}

	HD_INLINE void operator-=(const morton2d<T> rhs)
	{
		T x_diff = (this->key & x2_mask) - (rhs.key & x2_mask);
		T y_diff = (this->key & y2_mask) - (rhs.key & y2_mask);
		this->key = (x_diff & x2_mask) | (y_diff & y2_mask);
	}

	/* Increment X part of a morton2 code (xy interleaving)
	morton2(4,5).incX() == morton2(5,5);

	Ref : http://bitmath.blogspot.fr/2012/11/tesseral-arithmetic-useful-snippets.html */
	HD_INLINE morton2d incX(int n = 1) const
	{
#ifdef __CUDA_ARCH__
		const T x_sum = static_cast<T>((this->key | y2_mask) + (morton2dLUT_dev[n] << 1));
#else
		const T x_sum = static_cast<T>((this->key | y2_mask) + (morton2dLUT[n] << 1));
#endif
		return morton2d<T>((x_sum & x2_mask) | (this->key & y2_mask));
	}

	HD_INLINE morton2d incY(int n = 1) const
	{
#ifdef __CUDA_ARCH__
		const T y_sum = static_cast<T>((this->key | x2_mask) + morton2dLUT_dev[n]);
#else
		const T y_sum = static_cast<T>((this->key | x2_mask) + morton2dLUT[n]);
#endif
		return morton2d<T>((y_sum & y2_mask) | (this->key & x2_mask));
	}

	HD_INLINE morton2d decX(int n = 1) const
	{
#ifdef __CUDA_ARCH__
		const T x_diff = static_cast<T>((this->key & x2_mask) - (morton2dLUT_dev[n] << 1));
#else
		const T x_diff = static_cast<T>((this->key & x2_mask) - (morton2dLUT[n] << 1));
#endif
		return morton2d<T>((x_diff & x2_mask) | (this->key & y2_mask));
	}

	HD_INLINE morton2d decY(int n = 1) const
	{
#ifdef __CUDA_ARCH__
		const T y_diff = static_cast<T>((this->key & y2_mask) - morton2dLUT_dev[n]);
#else
		const T y_diff = static_cast<T>((this->key & y2_mask) - morton2dLUT[n]);
#endif
		return morton2d<T>((y_diff & y2_mask) | (this->key & x2_mask));
	}

	/*
	  min(morton2(4,5), morton2(8,3)) == morton2(4,3);
	  Ref : http://asgerhoedt.dk/?p=276
	*/
	static HD_INLINE morton2d min(const morton2d lhs, const morton2d rhs)
	{
		T lhsX = lhs.key & x2_mask;
		T rhsX = rhs.key & x2_mask;
		T lhsY = lhs.key & y2_mask;
		T rhsY = rhs.key & y2_mask;
		return morton2d<T>(std::min(lhsX, rhsX) + std::min(lhsY, rhsY));
	}

	/*
	  max(morton2(4,5), morton2(8,3)) == morton2(8,5);
	*/
	static HD_INLINE morton2d max(const morton2d lhs, const morton2d rhs)
	{
		T lhsX = lhs.key & x2_mask;
		T rhsX = rhs.key & x2_mask;
		T lhsY = lhs.key & y2_mask;
		T rhsY = rhs.key & y2_mask;
		return morton2d<T>(std::max(lhsX, rhsX) + std::max(lhsY, rhsY));
	}


	/* Fast encode of morton2 code when BMI2 instructions aren't available.
	This does not work for values greater than 256.

	This function takes roughly the same time as a full encode (64 bits) using BMI2 intrinsic.*/
	static HD_INLINE morton2d morton2d_256(const uint32_t x, const uint32_t y)
	{
		assert(x < 256 && y < 256);
#ifdef __CUDA_ARCH__
		T key = morton2dLUT_dev[x] << 1 |
        morton2dLUT_dev[y];
#else
		T key = morton2dLUT[x] << 1 |
        morton2dLUT[y];
#endif
		return morton2d(key);
	}

private:
	HD_INLINE constexpr uint64_t compactBits(uint64_t n) const
	{
		n &= 0x5555555555555555;
		n = (n ^ (n >> 1)) & 0x3333333333333333;
		n = (n ^ (n >> 2)) & 0x0f0f0f0f0f0f0f0f;
		n = (n ^ (n >> 4)) & 0x00ff00ff00ff00ff;
		n = (n ^ (n >> 8)) & 0x0000ffff0000ffff;
		n = (n ^ (n >> 16)) & 0x00000000ffffffff;
		return n;
	}

};

/* Add two morton keys (xy interleaving)
morton2(4,5) + morton3(1,2) == morton2(5,7); */
template<class T>
HD_INLINE morton2d<T> operator+(const morton2d<T> lhs, const morton2d<T> rhs)
{
	T x_sum = (lhs.key | y2_mask) + (rhs.key & x2_mask);
	T y_sum = (lhs.key | x2_mask) + (rhs.key & y2_mask);
	return morton2d<T>((x_sum & x2_mask) | (y_sum & y2_mask));
}

/* Substract two mortons keys (xy interleaving)
  morton2(4,5) - morton2(1,2) == morton2(3,3); */
template<class T>
HD_INLINE morton2d<T> operator-(const morton2d<T> lhs, const morton2d<T> rhs)
{
	T x_diff = (lhs.key & x2_mask) - (rhs.key & x2_mask);
	T y_diff = (lhs.key & y2_mask) - (rhs.key & y2_mask);
	return morton2d<T>((x_diff & x2_mask) | (y_diff & y2_mask));
}

template<class T>
HD_INLINE bool operator< (const morton2d<T>& lhs, const morton2d<T>& rhs)
{
	return (lhs.key) < (rhs.key);
}

template<class T>
HD_INLINE bool operator> (const morton2d<T>& lhs, const morton2d<T>& rhs)
{
	return (lhs.key) > (rhs.key);
}

template<class T>
HD_INLINE bool operator>= (const morton2d<T>& lhs, const morton2d<T>& rhs)
{
	return (lhs.key) >= (rhs.key);
}

template<class T>
HD_INLINE bool operator<= (const morton2d<T>& lhs, const morton2d<T>& rhs)
{
	return (lhs.key) <= (rhs.key);
}

template<class T>
std::ostream& operator<<(std::ostream& os, const morton2d<T>& m)
{
	uint64_t x, y;
	m.decode(x, y);
	os << m.key << ": " << x << ", " << y;
	return os;
}

typedef morton2d<> morton2;
typedef morton3d<> morton3;

}

#endif
