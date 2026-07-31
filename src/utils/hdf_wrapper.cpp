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

#include "hdf_wrapper.h"
#include "mpi.h"
#include "utils/logger.h"

namespace Aperture {

H5File::H5File() {}

// Only claim the handle is open if it is actually valid. Marking a negative
// hid_t as open makes the destructor call H5Fclose on it, which is where the
// "H5Fclose(): not a file ID" diagnostics come from after a failed create.
H5File::H5File(hid_t file_id) : m_file_id(file_id) { m_is_open = (file_id >= 0); }

H5File::H5File(const std::string& filename, H5OpenMode mode) {
  open(filename, mode);
}

H5File::H5File(H5File&& other) {
  m_file_id = other.m_file_id;
  // Carry over the source's open state rather than assuming it was open:
  // hdf_create() returns by value, so moving an invalid handle used to
  // resurrect it as "open" and hand the destructor a negative hid_t.
  m_is_open = other.m_is_open;
  m_is_parallel = other.m_is_parallel;
  other.m_is_open = false;
}

H5File::~H5File() { close(); }

H5File&
H5File::operator=(H5File&& other) {
  if (this == &other) return *this;
  close();
  m_file_id = other.m_file_id;
  m_is_open = other.m_is_open;
  m_is_parallel = other.m_is_parallel;
  other.m_is_open = false;
  return *this;
}

void
H5File::open(const std::string& filename, H5OpenMode mode) {
  unsigned int h5mode;
  if (mode == H5OpenMode::read_write || mode == H5OpenMode::rw_parallel)
    h5mode = H5F_ACC_RDWR;
  else
    h5mode = H5F_ACC_RDONLY;

  hid_t plist_id = H5P_DEFAULT;
  // Enable mpio when writing
  if (mode == H5OpenMode::rw_parallel ||
      mode == H5OpenMode::read_parallel) {
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    m_is_parallel = true;
  }

  m_file_id = H5Fopen(filename.c_str(), h5mode, plist_id);
  m_is_open = (m_file_id >= 0);
  if (!m_is_open) {
    Logger::print_err("Failed to open HDF5 file {}", filename);
  }

  if (mode == H5OpenMode::rw_parallel) {
    H5Pclose(plist_id);
  }
}

void
H5File::close() {
  if (m_is_open) {
    H5Fclose(m_file_id);
    m_is_open = false;
  }
}

H5File
hdf_create(const std::string& filename, H5CreateMode mode) {
  // auto h5mode =
  //     (mode == H5CreateMode::trunc ? H5F_ACC_TRUNC : H5F_ACC_EXCL);
  unsigned int h5mode;
  if (mode == H5CreateMode::trunc_parallel ||
      mode == H5CreateMode::trunc)
    h5mode = H5F_ACC_TRUNC;
  else
    h5mode = H5F_ACC_EXCL;

  hid_t plist_id = H5P_DEFAULT;
  bool parallel = false;
  if (mode == H5CreateMode::trunc_parallel ||
      mode == H5CreateMode::excl_parallel) {
    // Enable mpio when writing
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    parallel = true;
  }
  hid_t datafile =
      H5Fcreate(filename.c_str(), h5mode, H5P_DEFAULT, plist_id);

  if (mode == H5CreateMode::trunc_parallel ||
      mode == H5CreateMode::excl_parallel) {
    H5Pclose(plist_id);
  }

  // H5Fcreate returns a negative hid_t on failure. Report it here so the
  // failure is visible even for callers that do not check is_valid(); the
  // resulting H5File is marked not-open so writes are no-ops rather than
  // operations on a bogus id.
  if (datafile < 0) {
    Logger::print_err("Failed to create HDF5 file {}", filename);
  }

  H5File file(datafile);
  file.set_parallel(parallel);
  return file;
}

//// Explicitly specialize h5datatype functions
template <>
hid_t
h5datatype<char>() {
  return H5T_NATIVE_CHAR;
}

template <>
hid_t
h5datatype<float>() {
  return H5T_NATIVE_FLOAT;
}

template <>
hid_t
h5datatype<double>() {
  return H5T_NATIVE_DOUBLE;
}

template <>
hid_t
h5datatype<uint32_t>() {
  return H5T_NATIVE_UINT32;
}

template <>
hid_t
h5datatype<int>() {
  return H5T_NATIVE_INT;
}

template <>
hid_t
h5datatype<uint64_t>() {
  return H5T_NATIVE_UINT64;
}

}  // namespace Aperture
