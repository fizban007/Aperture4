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

#include "data_exporter.h"
#include "core/detail/multi_array_helpers.h"
#include "data/multi_array_data.hpp"
#include "data/particle_data.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "framework/params_store.h"
#include <filesystem>
// #include <fmt/ostream.h>
#include <fmt/core.h>

namespace fs = std::filesystem;

namespace Aperture {

template <typename Conf>
data_exporter<Conf>::data_exporter(sim_environment& env,
                                   const grid_t<Conf>& grid,
                                   const domain_comm<Conf>* comm)
    : system_t(env), m_grid(grid), m_comm(comm), m_output_grid(grid) {
  m_env.params().get_value("ptc_output_interval", m_ptc_output_interval);
  m_env.params().get_value("fld_output_interval", m_fld_output_interval);
  m_env.params().get_value("snapshot_interval", m_snapshot_interval);
  m_env.params().get_value("output_dir", m_output_dir);
  m_env.params().get_value("downsample", m_downsample);

  // Resize the tmp data array
  size_t max_ptc_num = 100, max_ph_num = 100;
  m_env.params().get_value("max_ptc_num", max_ptc_num);
  m_env.params().get_value("max_ph_num", max_ph_num);

  tmp_ptc_data.set_memtype(MemType::host_device);
  tmp_ptc_data.resize(std::max(max_ptc_num, max_ph_num));

  // Obtain the output grid
  for (int i = 0; i < Conf::dim; i++) {
    m_output_grid.dims[i] =
        m_grid.reduced_dim(i) / m_downsample + 2 * m_grid.guard[i];
    m_output_grid.offset[i] = m_grid.offset[i] / m_downsample;
  }
  tmp_grid_data.resize(m_output_grid.extent_less());

  // Obtain the global extent of the grid
  m_env.params().get_vec_t("N", m_global_ext);
  m_global_ext /= m_downsample;
}

template <typename Conf>
data_exporter<Conf>::~data_exporter() {
  if (m_xmf.is_open()) {
    m_xmf.close();
  }
}

template <typename Conf>
void
data_exporter<Conf>::init() {
  // make sure output directory is a directory
  if (m_output_dir.back() != '/') m_output_dir.push_back('/');
  fs::path outPath(m_output_dir);

  std::error_code returnedError;
  fs::create_directories(outPath, returnedError);

  // Copy config file to the output directory
  copy_config_file();

  // Write the grid in the simulation to the output directory
  write_grid();
}

template <typename Conf>
void
data_exporter<Conf>::register_data_components() {
  // tmp_E = m_env.register_data<vector_field<Conf>>("E_output", m_output_grid,
  //                                                 field_type::vert_centered);
  // tmp_B = m_env.register_data<vector_field<Conf>>("B_output", m_output_grid,
  //                                                 field_type::vert_centered);
}

template <typename Conf>
void
data_exporter<Conf>::update(double dt, uint32_t step) {
  double time = m_env.get_time();
  if (step % m_fld_output_interval == 0) {
    // Output downsampled fields!
    std::string filename =
        fmt::format("{}fld.{:05d}.h5", m_output_dir, m_fld_num);
    H5File datafile = hdf_create(filename, H5CreateMode::trunc_parallel);

    if (!m_xmf.is_open() && is_root()) {
      m_xmf.open(m_output_dir + "data.xmf");
    }
    write_xmf_step_header(m_xmf_buffer, time);

    // Loop over all data components and output according to their type
    for (auto& it : m_env.data_map()) {
      // Do not output the skipped components
      if (it.second->skip_output()) continue;

      // Logger::print_info("Working on {}", it.first);
      auto data = it.second.get();
      if (auto ptr = dynamic_cast<vector_field<Conf>*>(data)) {
        Logger::print_info("Writing vector field {}", it.first);
        write(*ptr, it.first, datafile, false);
      } else if (auto ptr = dynamic_cast<scalar_field<Conf>*>(data)) {
        Logger::print_info("Writing scalar field {}", it.first);
        write(*ptr, it.first, datafile, false);
      } else if (auto ptr = dynamic_cast<multi_array_data<float, 1>*>(data)) {
        Logger::print_info("Writing 1D array {}", it.first);
        write(*ptr, it.first, datafile, false);
      } else if (auto ptr = dynamic_cast<multi_array_data<float, 2>*>(data)) {
        Logger::print_info("Writing 2D array {}", it.first);
        write(*ptr, it.first, datafile, false);
      } else if (auto ptr = dynamic_cast<multi_array_data<float, 3>*>(data)) {
        Logger::print_info("Writing 3D array {}", it.first);
        write(*ptr, it.first, datafile, false);
      }
      if (data->reset_after_output()) {
        data->init();
      }
    }

    datafile.close();

    // Only write the xmf file on the root rank
    if (is_root()) {
      write_xmf_step_close(m_xmf_buffer);
      write_xmf_tail(m_xmf_buffer);

      if (step == 0) {
        write_xmf_head(m_xmf);
      } else {
        m_xmf.seekp(-26, std::ios_base::end);
      }
      m_xmf << m_xmf_buffer;
      m_xmf_buffer = "";
    }

    // Increment the output number
    m_fld_num += 1;
  }

  if (step % m_ptc_output_interval == 0) {
    // Output tracked particles!

    for (auto& it : m_env.data_map()) {
      auto data = it.second.get();
      if (auto ptr = dynamic_cast<particle_data_t*>(data)) {
        Logger::print_info("Writing tracked particles");
      } else if (auto ptr = dynamic_cast<photon_data_t*>(data)) {
        Logger::print_info("Writing tracked photons");
      }
    }
    m_ptc_num += 1;
  }

  if (m_snapshot_interval > 0 && step % m_snapshot_interval == 0) {
    write_snapshot((fs::path(m_output_dir) / "snapshot.h5").string(), step,
                   time);
  }
}

template <typename Conf>
void
data_exporter<Conf>::write_snapshot(const std::string& filename, uint32_t step,
                                    double time) {
  H5File snapfile = hdf_create(filename, H5CreateMode::trunc_parallel);

  // Walk over all data components and write them to the snapshot file according
  // to their `include_in_snapshot`
  for (auto& it : m_env.data_map()) {
    auto data = it.second.get();
    if (!data->include_in_snapshot()) {
      continue;
    }
    Logger::print_info("Writing {} to snapshot", it.first);
    if (auto ptr = dynamic_cast<vector_field<Conf>*>(data)) {
      write(*ptr, it.first, snapfile, true);
    } else if (auto ptr = dynamic_cast<scalar_field<Conf>*>(data)) {
      write(*ptr, it.first, snapfile, true);
    }
  }

  // Write simulation stats
  snapfile.write(step, "step");
  snapfile.write(time, "time");
  snapfile.write(m_fld_num, "output_fld_num");
  snapfile.write(m_ptc_num, "output_ptc_num");
  snapfile.write(m_fld_output_interval, "output_fld_interval");
  snapfile.write(m_ptc_output_interval, "output_ptc_interval");
  int num_ranks = 1;
  if (m_comm != nullptr) num_ranks = m_comm->size();

  snapfile.close();
}

template <typename Conf>
void
data_exporter<Conf>::load_snapshot(const std::string& filename, uint32_t& step,
                                   double& time) {
  H5File snapfile(filename, H5OpenMode::read_parallel);

  // Read simulation stats
  step = snapfile.read_scalar<uint32_t>("step");
  time = snapfile.read_scalar<double>("time");
  m_fld_num = snapfile.read_scalar<int>("output_fld_num");
  m_ptc_num = snapfile.read_scalar<int>("output_ptc_num");

  // Walk over all data components and read them from the snapshot file
  // according to their `include_in_snapshot`
  for (auto& it : m_env.data_map()) {
    auto data = it.second.get();
    if (!data->include_in_snapshot()) {
      continue;
    }
    Logger::print_info("Writing {} to snapshot", it.first);
    if (auto ptr = dynamic_cast<vector_field<Conf>*>(data)) {
      read(*ptr, it.first, snapfile, true);
      if (m_comm != nullptr) {
        m_comm->send_guard_cells(*ptr);
      }
    } else if (auto ptr = dynamic_cast<scalar_field<Conf>*>(data)) {
      read(*ptr, it.first, snapfile, true);
      if (m_comm != nullptr) {
        m_comm->send_guard_cells(*ptr);
      }
    }
  }

  // read field output interval to sort out the xmf file
  auto fld_interval = snapfile.read_scalar<int>("output_fld_interval");
  // TODO: touch up the xmf file

  snapfile.close();
}

template <typename Conf>
void
data_exporter<Conf>::copy_config_file() {
  std::string path = m_output_dir + "config.toml";
  std::string conf_file =
      m_env.params().template get_as<std::string>("config_file");
  Logger::print_info("Copying config file from {} to {}", conf_file, path);
  fs::path conf_path(conf_file);
  if (fs::exists(conf_path)) {
    fs::copy_file(conf_file, path, fs::copy_options::overwrite_existing);
  }
}

template <typename Conf>
void
data_exporter<Conf>::write_grid() {
  std::string meshfilename = m_output_dir + "grid.h5";
  H5File meshfile = hdf_create(meshfilename, H5CreateMode::trunc_parallel);

  // std::vector<float> x1_array(out_ext.x);
  multi_array<float, Conf::dim> x1_array(m_output_grid.extent_less(),
                                         MemType::host_only);
  multi_array<float, Conf::dim> x2_array(m_output_grid.extent_less(),
                                         MemType::host_only);
  multi_array<float, Conf::dim> x3_array(m_output_grid.extent_less(),
                                         MemType::host_only);

  // All data output points are cell centers
  for (auto idx : x1_array.indices()) {
    auto p = idx.get_pos() * m_downsample + m_grid.guards();
    auto x = m_grid.cart_coord(p);

    x1_array[idx] = x[0];
    if constexpr (Conf::dim > 1) x2_array[idx] = x[1];
    if constexpr (Conf::dim > 2) x3_array[idx] = x[2];
  }

  write_multi_array_helper("x1", x1_array, m_global_ext,
                           m_output_grid.offsets(), meshfile);
  if constexpr (Conf::dim > 1)
    write_multi_array_helper("x2", x2_array, m_global_ext,
                             m_output_grid.offsets(), meshfile);
  if constexpr (Conf::dim > 2)
    write_multi_array_helper("x3", x3_array, m_global_ext,
                             m_output_grid.offsets(), meshfile);

  meshfile.close();
}

template <typename Conf>
void
data_exporter<Conf>::write_xmf_head(std::ofstream& fs) {
  if (!is_root()) return;
  fs << "<?xml version=\"1.0\" ?>" << std::endl;
  fs << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>" << std::endl;
  fs << "<Xdmf>" << std::endl;
  fs << "<Domain>" << std::endl;
  fs << "<Grid Name=\"Aperture\" GridType=\"Collection\" "
        "CollectionType=\"Temporal\" >"
     << std::endl;
}

template <typename Conf>
void
data_exporter<Conf>::write_xmf_step_header(std::string& buffer, double time) {
  if (!is_root()) return;

  if (Conf::dim == 3) {
    m_dim_str = fmt::format("{} {} {}", m_global_ext[2], m_global_ext[1],
                            m_global_ext[0]);
  } else if (Conf::dim == 2) {
    m_dim_str = fmt::format("{} {}", m_global_ext[1], m_global_ext[0]);
  } else if (Conf::dim == 1) {
    m_dim_str = fmt::format("{} 1", m_global_ext[0]);
  }

  buffer += "<Grid Name=\"quadmesh\" Type=\"Uniform\">\n";
  buffer += fmt::format("  <Time Type=\"Single\" Value=\"{}\"/>\n", time);
  if (Conf::dim == 3) {
    buffer += fmt::format(
        "  <Topology Type=\"3DSMesh\" NumberOfElements=\"{}\"/>\n", m_dim_str);
    buffer += "  <Geometry GeometryType=\"X_Y_Z\">\n";
  } else if (Conf::dim == 2) {
    buffer += fmt::format(
        "  <Topology Type=\"2DSMesh\" NumberOfElements=\"{}\"/>\n", m_dim_str);
    buffer += "  <Geometry GeometryType=\"X_Y\">\n";
  } else if (Conf::dim == 1) {
    buffer += fmt::format(
        "  <Topology Type=\"2DSMesh\" NumberOfElements=\"{}\"/>\n", m_dim_str);
    buffer += "  <Geometry GeometryType=\"X_Y\">\n";
  }
  buffer += fmt::format(
      "    <DataItem Dimensions=\"{}\" NumberType=\"Float\" "
      "Precision=\"4\" Format=\"HDF\">\n",
      m_dim_str);
  buffer += "      grid.h5:x1\n";
  buffer += "    </DataItem>\n";
  if (Conf::dim >= 2) {
    buffer += fmt::format(
        "    <DataItem Dimensions=\"{}\" NumberType=\"Float\" "
        "Precision=\"4\" Format=\"HDF\">\n",
        m_dim_str);
    buffer += "      grid.h5:x2\n";
    buffer += "    </DataItem>\n";
  }
  if (Conf::dim >= 3) {
    buffer += fmt::format(
        "    <DataItem Dimensions=\"{}\" NumberType=\"Float\" "
        "Precision=\"4\" Format=\"HDF\">\n",
        m_dim_str);
    buffer += "      grid.h5:x3\n";
    buffer += "    </DataItem>\n";
  }

  buffer += "  </Geometry>\n";
}

template <typename Conf>
void
data_exporter<Conf>::write_xmf_step_close(std::string& buffer) {
  if (!is_root()) return;
  buffer += "</Grid>\n";
}

template <typename Conf>
void
data_exporter<Conf>::write_xmf_tail(std::string& buffer) {
  if (!is_root()) return;
  buffer += "</Grid>\n";
  buffer += "</Domain>\n";
  buffer += "</Xdmf>\n";
}

template <typename Conf>
void
data_exporter<Conf>::write_grid_multiarray(
    const std::string& name, const typename Conf::multi_array_t& array,
    stagger_t stagger, H5File& file) {
  if (array.dev_allocated() && tmp_grid_data.dev_allocated()) {
    resample_dev(array, tmp_grid_data, m_grid.guards(), index_t<Conf::dim>{},
                 stagger, m_output_stagger, m_downsample);
    tmp_grid_data.copy_to_host();
  } else {
    resample(array, tmp_grid_data, m_grid.guards(), index_t<Conf::dim>{},
             stagger, m_output_stagger, m_downsample);
  }

  // Logger::print_debug("writing global_ext {}x{}", m_global_ext[0],
  // m_global_ext[1]);

  write_multi_array_helper(name, tmp_grid_data, m_global_ext,
                           m_output_grid.offsets(), file);
}

template <typename Conf>
void
data_exporter<Conf>::write_multi_array_helper(
    const std::string& name,
    const multi_array<float, Conf::dim, typename Conf::idx_t>& array,
    const extent_t<Conf::dim>& global_ext, const index_t<Conf::dim>& offsets,
    H5File& file) {
  if (m_comm != nullptr && m_comm->size() > 1) {
    file.write_parallel(array, global_ext, offsets, array.extent(),
                        index_t<Conf::dim>{}, name);
  } else {
    file.write(array, name);
  }
}

template <typename Conf>
void
data_exporter<Conf>::write_xmf_field_entry(std::string& buffer, int num,
                                           const std::string& name) {
  if (is_root()) {
    m_xmf_buffer += fmt::format(
        "  <Attribute Name=\"{}\" Center=\"Node\" "
        "AttributeType=\"Scalar\">\n",
        name);
    m_xmf_buffer += fmt::format(
        "    <DataItem Dimensions=\"{}\" NumberType=\"Float\" "
        "Precision=\"4\" Format=\"HDF\">\n",
        m_dim_str);
    m_xmf_buffer += fmt::format("      fld.{:05d}.h5:{}\n", num, name);
    m_xmf_buffer += "    </DataItem>\n";
    m_xmf_buffer += "  </Attribute>\n";
  }
}

template <typename Conf>
template <int N>
void
data_exporter<Conf>::write(field_t<N, Conf>& data, const std::string& name,
                           H5File& datafile, bool snapshot) {
  // Loop over all components, downsample them, then write them to the file
  for (int i = 0; i < N; i++) {
    std::string namestr;
    if (N == 1) {
      namestr = name;
    } else {
      namestr = name + std::to_string(i + 1);
    }

    if (!snapshot) {
      write_grid_multiarray(namestr, data[i], data.stagger(i), datafile);
      write_xmf_field_entry(m_xmf_buffer, m_fld_num, namestr);
    } else {
      extent_t<Conf::dim> ext_total, ext;
      index_t<Conf::dim> pos_src, pos_dst;
      compute_snapshot_ext_offset(ext_total, ext, pos_src, pos_dst);

      data[i].copy_to_host();
      if (m_comm != nullptr && m_comm->size() > 1) {
        datafile.write_parallel(data[i], ext_total, pos_dst, ext, pos_src,
                                namestr);
      } else {
        datafile.write(data[i], namestr);
      }
    }
  }
}

template <typename Conf>
template <typename T, int Rank>
void
data_exporter<Conf>::write(multi_array_data<T, Rank>& data,
                           const std::string& name, H5File& datafile,
                           bool snapshot) {
  if (!snapshot) {
    if (m_comm != nullptr && m_comm->size() > 1) {
      m_comm->gather_to_root(static_cast<buffer<T>&>(data));
    } else {
      data.copy_to_host();
    }
    // gather_to_root only touches host memory, so we can directly use it to
    // write output
    if (is_root()) {
      datafile.write(static_cast<multi_array<T, Rank>&>(data), name);
    }
  }
}

template <typename Conf>
template <int N>
void
data_exporter<Conf>::read(field_t<N, Conf>& data, const std::string& name,
                          H5File& datafile, bool snapshot) {
  // Loop over all components, downsample them, then write them to the file
  for (int i = 0; i < N; i++) {
    std::string namestr;
    if (N == 1) {
      namestr = name;
    } else {
      namestr = name + std::to_string(i + 1);
    }

    if (snapshot) {
      extent_t<Conf::dim> ext_total, ext;
      index_t<Conf::dim> pos_src, pos_dst;
      compute_snapshot_ext_offset(ext_total, ext, pos_dst, pos_src);

      if (m_comm != nullptr && m_comm->size() > 1) {
        datafile.read_subset(data[i], namestr, pos_src, ext, pos_dst);
      } else {
        auto array =
            datafile.read_multi_array<typename Conf::value_t, Conf::dim>(
                namestr);
        data[i].host_copy_from(array);
      }
      data[i].copy_to_device();
    }
  }
}

template <typename Conf>
void
data_exporter<Conf>::compute_snapshot_ext_offset(extent_t<Conf::dim>& ext_total,
                                                 extent_t<Conf::dim>& ext,
                                                 index_t<Conf::dim>& pos_array,
                                                 index_t<Conf::dim>& pos_file) {
  ext_total = m_global_ext * m_downsample;
  ext = m_grid.extent_less();
  pos_file = m_grid.offsets();
  pos_array = index_t<Conf::dim>{};
  for (int n = 0; n < Conf::dim; n++) {
    if (pos_file[n] > 0) {
      pos_array[n] += m_grid.guard[n];
      pos_file[n] += m_grid.guard[n];
    }
    if (m_comm != nullptr &&
        m_comm->domain_info().neighbor_left[n] == MPI_PROC_NULL) {
      ext[n] += m_grid.guard[n];
    }
    if (m_comm != nullptr &&
        m_comm->domain_info().neighbor_right[n] == MPI_PROC_NULL) {
      ext[n] += m_grid.guard[n];
    }
  }
}

INSTANTIATE_WITH_CONFIG(data_exporter);

}  // namespace Aperture
