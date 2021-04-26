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

#include "mpi_helper.h"
#include "core/particle_structs.h"
#include <cstdint>
#include <cstdio>

#define BUFSIZE 1024

namespace Aperture {

MPI_Datatype MPI_PARTICLES;
MPI_Datatype MPI_PHOTONS;

namespace MPI_Helper {

////////////////////////////////////////////////////////////////////////////////
///  Specialize the MPI built-in data types
////////////////////////////////////////////////////////////////////////////////
template <>
MPI_Datatype
get_mpi_datatype(const char& x) {
  return MPI_CHAR;
}

template <>
MPI_Datatype
get_mpi_datatype(const short& x) {
  return MPI_SHORT;
}

template <>
MPI_Datatype
get_mpi_datatype(const int& x) {
  return MPI_INT;
}

template <>
MPI_Datatype
get_mpi_datatype(const uint32_t& x) {
  return MPI_UINT32_T;
}

template <>
MPI_Datatype
get_mpi_datatype(const uint16_t& x) {
  return MPI_UINT16_T;
}

template <>
MPI_Datatype
get_mpi_datatype(const bool& x) {
  return MPI_C_BOOL;
}

template <>
MPI_Datatype
get_mpi_datatype(const long& x) {
  return MPI_LONG;
}

template <>
MPI_Datatype
get_mpi_datatype(const unsigned char& x) {
  return MPI_UNSIGNED_CHAR;
}

// template<>
// MPI_Datatype get_mpi_datatype(const unsigned short& x) { return
// MPI_UNSIGNED_SHORT; }

// template<>
// MPI_Datatype get_mpi_datatype(const unsigned int& x) { return
// MPI_UNSIGNED; }

template <>
MPI_Datatype
get_mpi_datatype(const unsigned long& x) {
  return MPI_UNSIGNED_LONG;
}

template <>
MPI_Datatype
get_mpi_datatype(const float& x) {
  return MPI_FLOAT;
}

template <>
MPI_Datatype
get_mpi_datatype(const double& x) {
  return MPI_DOUBLE;
}

template <>
MPI_Datatype
get_mpi_datatype(const long double& x) {
  return MPI_LONG_DOUBLE;
}

template <>
MPI_Datatype
get_mpi_datatype(const single_ptc_t& x) {
  return MPI_PARTICLES;
}

template <>
MPI_Datatype
get_mpi_datatype(const single_ph_t& x) {
  return MPI_PHOTONS;
}

void
handle_mpi_error(int error_code, int rank) {
  if (error_code != MPI_SUCCESS) {
    char error_string[BUFSIZE];
    int length_of_error_string;

    MPI_Error_string(error_code, error_string, &length_of_error_string);
    fprintf(stderr, "%3d: %s\n", rank, error_string);
  }
}

template <typename PtcType>
void register_particle_type(const PtcType& ptc, MPI_Datatype* type) {
  constexpr int n_entries = visit_struct::field_count<PtcType>();
  int blocklengths[n_entries];
  MPI_Datatype types[n_entries];
  MPI_Aint offsets[n_entries];

  int n = 0;
  int offset = 0;
  visit_struct::for_each(
      ptc, [&n, &offset, &blocklengths, &types, &offsets](const char* name, auto& x) {
        blocklengths[n] = 1;
        types[n] = get_mpi_datatype(
            typename std::remove_reference<decltype(x)>::type());
        offsets[n] = offset;
        n += 1;
        offset += sizeof(typename std::remove_reference<decltype(x)>::type);
      });

  MPI_Type_create_struct(n_entries, blocklengths, offsets, types, type);
  MPI_Type_commit(type);
}

template void register_particle_type(const single_ptc_t& ptc, MPI_Datatype* type);
template void register_particle_type(const single_ph_t& ptc, MPI_Datatype* type);

}  // namespace MPI_Helper

}  // namespace Aperture
