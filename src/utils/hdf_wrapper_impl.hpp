#ifndef _HDF_WRAPPER_IMPL_H_
#define _HDF_WRAPPER_IMPL_H_

#include "hdf_wrapper.h"
#include "utils/logger.h"

namespace Aperture {

template <typename T>
void
H5File::write(T value, const std::string& name) {
  auto dataspace_id = H5Screate(H5S_SCALAR);
  auto dataset_id =
      H5Dcreate2(m_file_id, name.c_str(), h5datatype<T>(), dataspace_id,
                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  if (m_is_parallel) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) H5Sselect_none(dataspace_id);
  }
  auto status = H5Dwrite(dataset_id, h5datatype<T>(), H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, &value);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
}

template <typename T, int Dim>
void
H5File::write(const multi_array<T, Dim>& array, const std::string& name) {
  hsize_t dims[Dim];
  for (int i = 0; i < Dim; i++) {
    // TODO: implement a check for row major too
    dims[i] = array.extent()[Dim - 1 - i];
  }
  auto dataspace_id = H5Screate_simple(Dim, dims, NULL);
  auto dataset_id =
      H5Dcreate2(m_file_id, name.c_str(), h5datatype<T>(), dataspace_id,
                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  if (m_is_parallel) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) H5Sselect_none(dataspace_id);
  }
  auto status = H5Dwrite(dataset_id, h5datatype<T>(), H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, array.host_ptr());
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
}

template <typename T>
void
H5File::write(const T *array, size_t size, const std::string &name) {
  hsize_t dims[1];
  dims[0] = size;

  auto dataspace_id = H5Screate_simple(1, dims, NULL);
  auto dataset_id =
      H5Dcreate2(m_file_id, name.c_str(), h5datatype<T>(), dataspace_id,
                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  if (m_is_parallel) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) H5Sselect_none(dataspace_id);
  }
  auto status = H5Dwrite(dataset_id, h5datatype<T>(), H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, array);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
}

template <typename T, int Dim>
void
H5File::write_parallel(const multi_array<T, Dim>& array,
                       const extent_t<Dim>& ext_total, const index_t<Dim>& idx_dst,
                       const extent_t<Dim>& ext, const index_t<Dim>& idx_src,
                       const std::string& name) {
  hsize_t dims[Dim], array_dims[Dim];
  for (int i = 0; i < Dim; i++) {
    dims[i] = ext_total[Dim - 1 - i];
    array_dims[i] = array.extent()[Dim - 1 - i];
  }
  auto filespace_id = H5Screate_simple(Dim, dims, NULL);
  auto memspace_id = H5Screate_simple(Dim, array_dims, NULL);

  auto dataset_id =
      H5Dcreate2(m_file_id, name.c_str(), h5datatype<T>(), filespace_id,
                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t offsets[Dim], offsets_l[Dim], out_dim[Dim];
  hsize_t count[Dim], stride[Dim];
  for (int i = 0; i < Dim; i++) {
    count[i] = 1;
    stride[i] = 1;
    offsets[i] = idx_dst[Dim - i - 1];
    offsets_l[i] = idx_src[Dim - i - 1];
    out_dim[i] = ext[Dim - i - 1];
  }
  H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offsets, stride,
                      count, out_dim);
  H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET, offsets_l, stride,
                      count, out_dim);

  auto plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  auto status = H5Dwrite(dataset_id, h5datatype<T>(), memspace_id,
                         filespace_id, plist_id, array.host_ptr());

  H5Dclose(dataset_id);
  H5Sclose(filespace_id);
  H5Sclose(memspace_id);
  H5Pclose(plist_id);

  if (status < 0) {
    Logger::print_err("H5Dwrite error! Status is {}", status);
  }
}

template <typename T>
void
H5File::write_parallel(const T* array, size_t array_size,
                       size_t len_total, size_t idx_dst, size_t len,
                       size_t idx_src, const std::string& name) {
  if (len == 0 && len_total == 0) {
    write(array, array_size, name);
    return;
  }

  hsize_t dims[1], array_dims[1];
  dims[0] = len_total;
  array_dims[0] = array_size;
  auto filespace_id = H5Screate_simple(1, dims, NULL);
  auto memspace_id = H5Screate_simple(1, array_dims, NULL);

  auto dataset_id =
      H5Dcreate2(m_file_id, name.c_str(), h5datatype<T>(), filespace_id,
                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t offsets[1], offsets_l[1], out_dim[1];
  hsize_t count[1], stride[1];
  // count[0] = ((len == 0 && len_total == 0) ? 0 : 1);
  count[0] = 1;
  stride[0] = 1;
  offsets[0] = idx_dst;
  offsets_l[0] = idx_src;
  out_dim[0] = len;

  H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offsets, stride,
                      count, out_dim);
  H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET, offsets_l, stride,
                      count, out_dim);

  auto plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  hid_t status;
  status = H5Dwrite(dataset_id, h5datatype<T>(), memspace_id,
                    filespace_id, plist_id, array);

  H5Dclose(dataset_id);
  H5Sclose(filespace_id);
  H5Sclose(memspace_id);
  H5Pclose(plist_id);

  if (status < 0) {
    Logger::print_err("H5Dwrite error! Status is {}", status);
  }
}

template <typename T, int Dim>
multi_array<T, Dim>
H5File::read_multi_array(const std::string& name) {
  extent_t<Dim> ext;

  auto dataset = H5Dopen(m_file_id, name.c_str(), H5P_DEFAULT);

  hsize_t dims[Dim];
  auto dataspace = H5Dget_space(dataset); /* dataspace handle */
  int dim = H5Sget_simple_extent_ndims(dataspace);
  H5Sget_simple_extent_dims(dataspace, dims, NULL);

  for (int i = 0; i < dim; i++) {
    ext[i] = dims[dim - i - 1];
  }

  multi_array<T, Dim> result(ext);
  H5Dread(dataset, h5datatype<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
          result.host_ptr());

  H5Dclose(dataset);
  H5Sclose(dataspace);
  return result;
}

template <typename T>
T
H5File::read_scalar(const std::string& name) {
  T result;
  auto dataset = H5Dopen(m_file_id, name.c_str(), H5P_DEFAULT);
  auto dataspace = H5Dget_space(dataset); /* dataspace handle */

  H5Dread(dataset, h5datatype<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
          &result);

  H5Dclose(dataset);
  H5Sclose(dataspace);
  return result;
}

template <typename T>
std::vector<T>
H5File::read_vector(const std::string& name) {
  auto dataset = H5Dopen(m_file_id, name.c_str(), H5P_DEFAULT);
  auto dataspace = H5Dget_space(dataset); /* dataspace handle */
  int dim = H5Sget_simple_extent_ndims(dataspace);
  std::vector<hsize_t> dims(dim);
  H5Sget_simple_extent_dims(dataspace, dims.data(), NULL);
  size_t total_dim = 1;
  for (int i = 0; i < dim; i++) {
    total_dim *= dims[i];
  }

  std::vector<T> result(total_dim);
  H5Dread(dataset, h5datatype<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
          result.data());

  H5Dclose(dataset);
  H5Sclose(dataspace);
  return result;
}

template <typename T, int Dim>
void
H5File::read_subset(multi_array<T, Dim>& array, const std::string& name,
                    const index_t<Dim>& idx_src, const extent_t<Dim>& ext,
                    const index_t<Dim>& idx_dst) {
  hsize_t dims[Dim], array_dims[Dim];
  for (int i = 0; i < ext.dim(); i++) {
    // dims[i] = ext_total[ext_total.dim() - 1 - i];
    array_dims[i] = array.extent()[array.dim() - 1 - i];
  }
  auto dataset = H5Dopen(m_file_id, name.c_str(), H5P_DEFAULT);
  auto dataspace = H5Dget_space(dataset); /* dataspace handle */
  int dim = H5Sget_simple_extent_ndims(dataspace);
  H5Sget_simple_extent_dims(dataspace, dims, NULL);

  auto memspace = H5Screate_simple(ext.dim(), array_dims, NULL);

  hsize_t offsets[Dim];
  hsize_t offsets_l[Dim];
  hsize_t out_dim[Dim];
  hsize_t count[Dim];
  hsize_t stride[Dim];
  for (int i = 0; i < dim; i++) {
    offsets[i] = idx_src[dim - i - 1];
    offsets_l[i] = idx_dst[dim - i - 1];
    out_dim[i] = ext[dim - i - 1];
    count[i] = 1;
    stride[i] = 1;
  }
  H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offsets, stride, count,
                      out_dim);
  H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets_l, stride,
                      count, out_dim);

  auto plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  auto status = H5Dread(dataset, h5datatype<T>(), memspace, dataspace,
                        plist_id, array.host_ptr());

  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Sclose(memspace);
  H5Pclose(plist_id);

  if (status < 0) {
    Logger::print_err("H5Dread error in subset read! Status is {}", status);
  }
}

template <typename T>
void
H5File::read_subset(T* array, size_t array_size,
                    const std::string& name, size_t idx_src, size_t len,
                    size_t idx_dst) {
  hsize_t dims[1] = {0};
  hsize_t array_dims[1] = {0};
  // dims[i] = ext_total[ext_total.dim() - 1 - i];
  array_dims[0] = array_size;

  auto dataset = H5Dopen(m_file_id, name.c_str(), H5P_DEFAULT);
  auto dataspace = H5Dget_space(dataset); /* dataspace handle */
  // int dim = H5Sget_simple_extent_ndims(dataspace);
  H5Sget_simple_extent_dims(dataspace, dims, NULL);

  // Logger::print_debug("reading subset, dims[0] is {}", dims[0]);
  if (dims[0] == 0) {
    H5Sclose(dataspace);
    H5Dclose(dataset);
    return;
  }

  auto memspace = H5Screate_simple(1, array_dims, NULL);

  hsize_t offsets[1] = {1};
  hsize_t offsets_l[1] = {1};
  hsize_t out_dim[1] = {1};
  hsize_t count[1] = {1};
  hsize_t stride[1] = {1};

  offsets[0] = idx_src;
  offsets_l[0] = idx_dst;
  out_dim[0] = len;

  H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offsets, stride, count,
                      out_dim);
  H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offsets_l, stride,
                      count, out_dim);

  auto plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  auto status = H5Dread(dataset, h5datatype<T>(), memspace, dataspace,
                        plist_id, array);

  H5Dclose(dataset);
  H5Sclose(dataspace);
  H5Sclose(memspace);
  H5Pclose(plist_id);

  if (status < 0) {
    Logger::print_err("H5Dread error in subset read! Status is {}", status);
  }
}

}  // namespace Aperture

#endif  // _HDF_WRAPPER_IMPL_H_
