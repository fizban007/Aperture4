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

#include "core/detail/multi_array_helpers.h"
#include "data/multi_array_data.hpp"
#include "data/particle_data.h"
#include "data_exporter.h"
#include "framework/config.h"
#include "framework/environment.h"
#include "framework/params_store.h"
#include "utils/for_each_dual.hpp"
#include "utils/timer.h"
#if (__GNUC__ >= 8 || __clang_major__ >= 7) && \
    !defined(__USE_BOOST_FILESYSTEM__)
#include <filesystem>
#else
#define USE_BOOST_FILESYSTEM
#include <boost/filesystem.hpp>
#endif
// #include <fmt/ostream.h>
#include <fmt/core.h>

#ifndef USE_BOOST_FILESYSTEM
namespace fs = std::filesystem;
#else
namespace fs = boost::filesystem;
#endif

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
// data_exporter<Conf>::data_exporter(sim_environment& env,
data_exporter<Conf, ExecPolicy>::data_exporter(
    const grid_t<Conf>& grid, const domain_comm<Conf, ExecPolicy>* comm)
    : m_grid(grid), m_comm(comm), m_output_grid(grid) {
  sim_env().params().get_value("ptc_output_interval", m_ptc_output_interval);
  sim_env().params().get_value("fld_output_interval", m_fld_output_interval);
  sim_env().params().get_value("snapshot_interval", m_snapshot_interval);
  sim_env().params().get_value("output_dir", m_output_dir);
  sim_env().params().get_value("downsample", m_downsample);

  // Resize the tmp data array
  size_t max_ptc_num = 100, max_ph_num = 100;
  sim_env().params().get_value("max_ptc_num", max_ptc_num);
  sim_env().params().get_value("max_ph_num", max_ph_num);

  // tmp_ptc_data.set_memtype(MemType::host_device);
  tmp_ptc_data32.set_memtype(ExecPolicy<Conf>::data_mem_type());
  tmp_ptc_data32.resize(std::max(max_ptc_num, max_ph_num));
  tmp_ptc_data64.set_memtype(ExecPolicy<Conf>::data_mem_type());
  tmp_ptc_data64.resize(std::max(max_ptc_num, max_ph_num));

  // Obtain the output grid
  for (int i = 0; i < Conf::dim; i++) {
    m_output_grid.N[i] = m_grid.reduced_dim(i) / m_downsample;
    m_output_grid.dims[i] =
        m_grid.reduced_dim(i) / m_downsample + 2 * m_grid.guard[i];
    m_output_grid.offset[i] = m_grid.offset[i] / m_downsample;
  }
  tmp_grid_data.resize(m_output_grid.extent_less());

  // Obtain the global extent of the grid
  sim_env().params().get_vec_t("N", m_global_ext);
  m_global_ext /= m_downsample;
  m_global_ext.get_strides();
}

template <typename Conf, template <class> class ExecPolicy>
data_exporter<Conf, ExecPolicy>::~data_exporter() {
  if (m_xmf.is_open()) {
    m_xmf.close();
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::init() {
  // make sure output directory is a directory
  if (m_output_dir.back() != '/') m_output_dir.push_back('/');
  fs::path outPath(m_output_dir);

#ifndef USE_BOOST_FILESYSTEM
  std::error_code returnedError;
  fs::create_directories(outPath, returnedError);
#else
  fs::create_directories(outPath);
#endif

  // Copy config file to the output directory
  copy_config_file();

  // Write the grid in the simulation to the output directory
  write_grid();
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::register_data_components() {
  // tmp_E = m_env.register_data<vector_field<Conf>>("E_output", m_output_grid,
  //                                                 field_type::vert_centered);
  // tmp_B = m_env.register_data<vector_field<Conf>>("B_output", m_output_grid,
  //                                                 field_type::vert_centered);
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write_field_data(data_t* data,
                                                  const std::string& name,
                                                  H5File& datafile,
                                                  bool snapshot) {
  using value_t = typename Conf::value_t;
  if (auto* ptr = dynamic_cast<vector_field<Conf>*>(data)) {
    Logger::print_detail("Writing vector field {}", name);
    write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<scalar_field<Conf>*>(data)) {
    Logger::print_detail("Writing scalar field {}", name);
    write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<field_t<4, Conf>*>(data)) {
    Logger::print_detail("Writing 4-vector field {}", name);
    write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<field_t<6, Conf>*>(data)) {
    Logger::print_detail("Writing antisymmetric 4-tensor field {}", name);
    write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<field_t<10, Conf>*>(data)) {
    Logger::print_detail("Writing symmetric 4-tensor field {}", name);
    write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<scalar_data<value_t>*>(data)) {
    Logger::print_detail("Writing scalar data {}", name);
    write(*ptr, name, datafile, false);
    // } else if (auto* ptr = dynamic_cast<momentum_space<Conf>*>(data)) {
    //   Logger::print_detail("Writing momentum space data");
    //   write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<phase_space<Conf, 1>*>(data)) {
    Logger::print_detail("Writing 1D phase space data {}", name);
    write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<phase_space<Conf, 2>*>(data)) {
    Logger::print_detail("Writing 2D phase space data {}", name);
    write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<phase_space<Conf, 3>*>(data)) {
    Logger::print_detail("Writing 3D phase space data {}", name);
    write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<phase_space_vlasov<Conf, 1>*>(data)) {
    Logger::print_detail("Writing 1D phase space data {}", name);
    write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<multi_array_data<float, 1>*>(data)) {
    Logger::print_detail("Writing 1D array {}", name);
    write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<multi_array_data<float, 2>*>(data)) {
    Logger::print_detail("Writing 2D array {}", name);
    write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<multi_array_data<float, 3>*>(data)) {
    Logger::print_detail("Writing 3D array {}", name);
    write(*ptr, name, datafile, false);
  } else if (auto* ptr = dynamic_cast<multi_array_data<uint32_t, 3>*>(data)) {
    Logger::print_detail("Writing 3D array {}", name);
    write(*ptr, name, datafile, false);
  } else {
    Logger::print_detail("Data exporter doesn't know how to write {}", name);
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::update(double dt, uint32_t step) {
  double time = sim_env().get_time();
  using value_t = typename Conf::value_t;

  // Load snapshot if it is a restart
  if (sim_env().is_restart()) {
    double time = 0.0;
    load_snapshot(sim_env().restart_file(), step, time);
    sim_env().set_step(step);
    sim_env().set_time(time);
    sim_env().finish_restart();
    return;
  }

  if (m_comm != nullptr) {
    m_comm->barrier();
  }
  if (step % m_fld_output_interval == 0) {
    // timer::stamp("write_field");
    // Output downsampled fields!
    std::string filename =
        fmt::format("{}fld.{:05d}.h5", m_output_dir, m_fld_num);
    auto create_mode = H5CreateMode::trunc_parallel;
    // if (!is_multi_rank()) create_mode = H5CreateMode::trunc;
    if (sim_env().use_mpi() == false) create_mode = H5CreateMode::trunc;
    H5File datafile = hdf_create(filename, create_mode);

    datafile.write(step, "step");
    datafile.write(time, "time");

    if (!m_xmf.is_open() && is_root()) {
      m_xmf.open(m_output_dir + "data.xmf");
    }
    write_xmf_step_header(m_xmf_buffer, time);

    // Loop over all data components and output according to their type
    for (auto& it : sim_env().data_map()) {
      // Do not output the skipped components
      if (it.second->skip_output()) continue;
      if (it.second->m_special_output_interval != 0) continue;

      // Logger::print_info("Working on {}", it.first);
      auto data = it.second.get();
      write_field_data(data, it.first, datafile, false);

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
    // timer::show_duration_since_stamp("write_field", "ms",
    //                                  "write_field");
  }

  if (m_ptc_output_interval > 0 && step % m_ptc_output_interval == 0) {
    // Output tracked particles!
    std::string filename =
        fmt::format("{}ptc.{:05d}.h5", m_output_dir, m_ptc_num);
    auto create_mode = H5CreateMode::trunc_parallel;
    if (sim_env().use_mpi() == false) create_mode = H5CreateMode::trunc;
    H5File datafile = hdf_create(filename, create_mode);

    datafile.write(step, "step");
    datafile.write(time, "time");

    for (auto& it : sim_env().data_map()) {
      auto data = it.second.get();
      if (auto* ptr = dynamic_cast<tracked_particles_t*>(data)) {
        Logger::print_detail("Writing tracked particles");
        write(*ptr, it.first, datafile, false);
      } else if (auto* ptr = dynamic_cast<tracked_photons_t*>(data)) {
        Logger::print_detail("Writing tracked photons");
        write(*ptr, it.first, datafile, false);
      }
    }
    m_ptc_num += 1;
    datafile.close();
  }

  // Loop over the data map again to find special output cases
  for (auto& it : sim_env().data_map()) {
    // m_special_output_interval == 0 means it follows the standard output
    // interval rule
    if (it.second->m_special_output_interval == 0) continue;

    if (step % it.second->m_special_output_interval == 0) {
      // Specifically write output file for this data component

      auto data = it.second.get();
      std::string filename = fmt::format("{}data_{}.{:05d}.h5", m_output_dir,
                                         it.first, data->m_special_output_num);
      auto create_mode = H5CreateMode::trunc_parallel;
      if (sim_env().use_mpi() == false) create_mode = H5CreateMode::trunc;
      H5File datafile = hdf_create(filename, create_mode);

      datafile.write(step, "step");
      datafile.write(time, "time");

      write_field_data(data, it.first, datafile, false);

      if (data->reset_after_output()) {
        data->init();
      }

      datafile.close();
      data->m_special_output_num += 1;
    }
  }

  // Save snapshot
  if (m_snapshot_interval > 0 && step % m_snapshot_interval == 0 && step > 0) {
    write_snapshot((fs::path(m_output_dir) / "snapshot.h5").string(), step,
                   time);
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write_snapshot(const std::string& filename,
                                                uint32_t step, double time) {
  auto create_mode = H5CreateMode::trunc_parallel;
  if (sim_env().use_mpi() == false) create_mode = H5CreateMode::trunc;
  // if (!is_multi_rank()) create_mode = H5CreateMode::trunc;
  H5File snapfile = hdf_create(filename, create_mode);

  // Walk over all data components and write them to the snapshot file according
  // to their `include_in_snapshot`
  for (auto& it : sim_env().data_map()) {
    auto data = it.second.get();
    if (!data->include_in_snapshot()) {
      continue;
    }
    Logger::print_detail("Writing {} to snapshot", it.first);
    if (auto* ptr = dynamic_cast<vector_field<Conf>*>(data)) {
      write(*ptr, it.first, snapfile, true);
    } else if (auto* ptr = dynamic_cast<scalar_field<Conf>*>(data)) {
      write(*ptr, it.first, snapfile, true);
    } else if (auto* ptr = dynamic_cast<particle_data_t*>(data)) {
      // write(*ptr, it.first, snapfile, true);
      write_ptc_snapshot(*ptr, it.first, snapfile);
      // Also write particle numbers of each rank
      size_t ptc_num = ptr->number();
      int num_ranks = 1;
      int rank = 0;
      if (is_multi_rank()) {
        num_ranks = m_comm->size();
        rank = m_comm->rank();
        snapfile.write_parallel(&ptc_num, 1, num_ranks, rank, 1, 0, "ptc_num");
      } else {
        snapfile.write(&ptc_num, 1, "ptc_num");
      }
    } else if (auto* ptr = dynamic_cast<photon_data_t*>(data)) {
      // write(*ptr, it.first, snapfile, true);
      write_ptc_snapshot(*ptr, it.first, snapfile);
      // Also write photon numbers of each rank
      size_t ph_num = ptr->number();
      int num_ranks = 1;
      int rank = 0;
      if (is_multi_rank()) {
        num_ranks = m_comm->size();
        rank = m_comm->rank();
        snapfile.write_parallel(&ph_num, 1, num_ranks, rank, 1, 0, "ph_num");
      } else {
        snapfile.write(&ph_num, 1, "ph_num");
      }
    } else if (auto* ptr = dynamic_cast<rng_states_t<exec_tag>*>(data)) {
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

  // Copy the current data.xmf file to a snapshot one
  std::string xmf_file = m_output_dir + "data.xmf";
  std::string xmf_snapshot_file = m_output_dir + "snapshot.xmf";
  Logger::print_detail("Copying xmf file from {} to {}", xmf_file,
                       xmf_snapshot_file);
#ifndef USE_BOOST_FILESYSTEM
  fs::copy_file(xmf_file, xmf_snapshot_file,
                fs::copy_options::overwrite_existing);
#else
  fs::copy_file(xmf_file, xmf_snapshot_file,
                fs::copy_option::overwrite_if_exists);
#endif

  Logger::print_info("Finishd writing snapshot at time {}, step {}",
                     time, step);
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::load_snapshot(const std::string& filename,
                                               uint32_t& step, double& time) {
  H5File snapfile(filename, H5OpenMode::read_parallel);

  // Read simulation stats
  step = snapfile.read_scalar<uint32_t>("step");
  Logger::print_info("Snapshot step is {}", step);
  time = snapfile.read_scalar<double>("time");
  Logger::print_info("Snapshot time is {}", time);
  m_fld_num = snapfile.read_scalar<int>("output_fld_num");
  m_ptc_num = snapfile.read_scalar<int>("output_ptc_num");

  // Walk over all data components and read them from the snapshot file
  // according to their `include_in_snapshot`
  for (auto& it : sim_env().data_map()) {
    auto data = it.second.get();
    if (!data->include_in_snapshot()) {
      continue;
    }
    Logger::print_detail("Loading {} from snapshot", it.first);
    if (auto* ptr = dynamic_cast<vector_field<Conf>*>(data)) {
      read(*ptr, it.first, snapfile, true);
      if (m_comm != nullptr) {
        m_comm->send_guard_cells(*ptr);
      }
    } else if (auto* ptr = dynamic_cast<scalar_field<Conf>*>(data)) {
      read(*ptr, it.first, snapfile, true);
      if (m_comm != nullptr) {
        m_comm->send_guard_cells(*ptr);
      }
    } else if (auto* ptr = dynamic_cast<particle_data_t*>(data)) {
      size_t ptc_num;
      int rank = 0;
      if (is_multi_rank()) {
        rank = m_comm->rank();
        snapfile.read_subset(&ptc_num, 1, "ptc_num", rank, 1, 0);
      } else {
        snapfile.read_array(&ptc_num, 1, "ptc_num");
      }
      Logger::print_info_all("rank {} has ptc_num {}", rank, ptc_num);
      uint64_t total = ptc_num, offset = 0;
      if (is_multi_rank()) {
        m_comm->get_total_num_offset(ptc_num, total, offset);
      }
      read_ptc_snapshot(*ptr, it.first, snapfile, ptc_num, total, offset);
    } else if (auto* ptr = dynamic_cast<photon_data_t*>(data)) {
      size_t ph_num;
      int rank = 0;
      if (is_multi_rank()) {
        rank = m_comm->rank();
        snapfile.read_subset(&ph_num, 1, "ph_num", rank, 1, 0);
      } else {
        snapfile.read_array(&ph_num, 1, "ph_num");
      }
      Logger::print_info_all("rank {} has ph_num {}", rank, ph_num);
      uint64_t total = ph_num, offset = 0;
      if (is_multi_rank()) {
        m_comm->get_total_num_offset(ph_num, total, offset);
      }
      read_ptc_snapshot(*ptr, it.first, snapfile, ph_num, total, offset);
    } else if (auto* ptr = dynamic_cast<rng_states_t<exec_tag>*>(data)) {
      read(*ptr, it.first, snapfile, true);
    }
  }

  // read field output interval to sort out the xmf file
  auto fld_interval = snapfile.read_scalar<int>("output_fld_interval");

  // Copy snapshot.xmf back to data.xmf
  std::string xmf_file = m_output_dir + "data.xmf";
  std::string xmf_snapshot_file = m_output_dir + "snapshot.xmf";
  Logger::print_detail("Copying xmf file from {} to {}", xmf_snapshot_file,
                       xmf_file);
#ifndef USE_BOOST_FILESYSTEM
  fs::copy_file(xmf_snapshot_file, xmf_file,
                fs::copy_options::overwrite_existing);
#else
  fs::copy_file(xmf_snapshot_file, xmf_file,
                fs::copy_option::overwrite_if_exists);
#endif

  snapfile.close();
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::copy_config_file() {
  std::string path = m_output_dir + "config.toml";
  std::string conf_file =
      sim_env().params().template get_as<std::string>("config_file");
  Logger::print_detail("Copying config file from {} to {}", conf_file, path);
  fs::path conf_path(conf_file);
  if (fs::exists(conf_path)) {
#ifndef USE_BOOST_FILESYSTEM
    fs::copy_file(conf_file, path, fs::copy_options::overwrite_existing);
#else
    fs::copy_file(conf_file, path, fs::copy_option::overwrite_if_exists);
#endif
  } else {
    sim_env().params().write(path);
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write_grid() {
  std::string meshfilename = m_output_dir + "grid.h5";
  Logger::print_info("Writing to grid file {}", meshfilename);
  auto create_mode = H5CreateMode::trunc_parallel;
  if (sim_env().use_mpi() == false) create_mode = H5CreateMode::trunc;
  H5File meshfile = hdf_create(meshfilename, create_mode);

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
    if CONST_EXPR (Conf::dim > 1) x2_array[idx] = x[1];
    if CONST_EXPR (Conf::dim > 2) x3_array[idx] = x[2];
  }

  write_multi_array_helper("x1", x1_array, m_global_ext,
                           m_output_grid.offsets(), meshfile);
  if CONST_EXPR (Conf::dim > 1)
    write_multi_array_helper("x2", x2_array, m_global_ext,
                             m_output_grid.offsets(), meshfile);
  if CONST_EXPR (Conf::dim > 2)
    write_multi_array_helper("x3", x3_array, m_global_ext,
                             m_output_grid.offsets(), meshfile);

  meshfile.close();
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write_xmf_head(std::ofstream& fs) {
  if (!is_root()) return;
  fs << "<?xml version=\"1.0\" ?>" << std::endl;
  fs << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>" << std::endl;
  fs << "<Xdmf>" << std::endl;
  fs << "<Domain>" << std::endl;
  fs << "<Grid Name=\"Aperture\" GridType=\"Collection\" "
        "CollectionType=\"Temporal\" >"
     << std::endl;
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write_xmf_step_header(std::string& buffer,
                                                       double time) {
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

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write_xmf_step_close(std::string& buffer) {
  if (!is_root()) return;
  buffer += "</Grid>\n";
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write_xmf_tail(std::string& buffer) {
  if (!is_root()) return;
  buffer += "</Grid>\n";
  buffer += "</Domain>\n";
  buffer += "</Xdmf>\n";
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write_grid_multiarray(
    const std::string& name, const typename Conf::multi_array_t& array,
    stagger_t stagger, H5File& file) {
  // if (m_downsample != 1) {
  // if (array.dev_allocated() && tmp_grid_data.dev_allocated()) {
  //   resample(exec_tags::device{}, array, tmp_grid_data, m_grid.guards(),
  //   index_t<Conf::dim>{},
  //                stagger, m_output_stagger, m_downsample);
  //   tmp_grid_data.copy_to_host();
  // } else {
  //   resample(exec_tags::host{}, array, tmp_grid_data, m_grid.guards(),
  //   index_t<Conf::dim>{},
  //            stagger, m_output_stagger, m_downsample);
  // }
  resample(exec_tag{}, array, tmp_grid_data, m_grid.guards(),
           index_t<Conf::dim>{}, stagger, m_output_stagger, m_downsample);
  tmp_grid_data.copy_to_host();
  // } else {
  //   tmp_grid_data.copy_from(array);
  //   tmp_grid_data.copy_to_host();
  // }

  // Logger::print_debug("writing global_ext {}x{}", m_global_ext[0],
  // m_global_ext[1]);

  write_multi_array_helper(name, tmp_grid_data, m_global_ext,
                           m_output_grid.offsets(), file);
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write_multi_array_helper(
    const std::string& name,
    const multi_array<float, Conf::dim, typename Conf::idx_t>& array,
    const extent_t<Conf::dim>& global_ext, const index_t<Conf::dim>& offsets,
    H5File& file) {
  if (is_multi_rank()) {
    file.write_parallel(array, global_ext, offsets, array.extent(),
                        index_t<Conf::dim>{}, name);
  } else {
    file.write(array, name);
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write_xmf_field_entry(
    std::string& buffer, int num, const std::string& name) {
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

template <typename Conf, template <class> class ExecPolicy>
template <typename PtcData>
void
data_exporter<Conf, ExecPolicy>::write_ptc_snapshot(PtcData& data,
                                                    const std::string& name,
                                                    H5File& datafile) {
  auto& ptc_buffer64 = tmp_ptc_data64;
  auto& ptc_buffer32 = tmp_ptc_data32;
  size_t number = data.number();
  Logger::print_debug("Writing snapshot of {} particles", number);
  size_t total = number;
  size_t offset = 0;
  bool multi_rank = is_multi_rank();
  if (multi_rank) {
    m_comm->get_total_num_offset(number, total, offset);
  }
  visit_struct::for_each(
      adapt(exec_tag{}, data),
      [&ptc_buffer64, &ptc_buffer32, &name, &datafile, number, total, offset,
       multi_rank](const char* entry, auto u) {
        // Copy to the temporary ptc buffer
        // ptr_copy(exec_tags::device{}, reinterpret_cast<uint64_t*>(u),
        // ptc_buffer.dev_ptr(), number,
        typedef typename std::remove_reference_t<decltype(*u)> data_type;
        auto write_component = [&](auto& ptc_buffer) {
          auto ptr = adapt(exec_tag{}, ptc_buffer);
          ptr_copy(exec_tag{}, reinterpret_cast<decltype(ptr)>(u), ptr, number,
                   0, 0);
          ptc_buffer.copy_to_host();
          if (multi_rank) {
            datafile.write_parallel(
                reinterpret_cast<const data_type*>(ptc_buffer.host_ptr()),
                number, total, offset, number, 0, name + "_" + entry);
          } else {
            datafile.write(reinterpret_cast<data_type*>(ptc_buffer.host_ptr()),
                           number, name + "_" + entry);
          }
        };
        if (sizeof(data_type) == 4) {
          write_component(ptc_buffer32);
        } else if (sizeof(data_type) == 8) {
          write_component(ptc_buffer64);
        }
      });
}

template <typename Conf, template <class> class ExecPolicy>
template <typename PtcData>
void
data_exporter<Conf, ExecPolicy>::read_ptc_snapshot(PtcData& data,
                                                   const std::string& name,
                                                   H5File& datafile,
                                                   size_t number, size_t total,
                                                   size_t offset) {
  auto& ptc_buffer64 = tmp_ptc_data64;
  auto& ptc_buffer32 = tmp_ptc_data32;
  bool multi_rank = is_multi_rank();
  // visit_struct::for_each(data.dev_ptrs(), [&ptc_buffer, &name, &datafile,
  visit_struct::for_each(
      adapt(exec_tag{}, data),
      [&ptc_buffer64, &ptc_buffer32, &name, &datafile, number, total, offset,
       multi_rank](const char* entry, auto u) {
        typedef typename std::remove_reference_t<decltype(*u)> data_type;
        Logger::print_info("reading {}", entry);
        auto read_component = [&](auto& ptc_buffer) {
          if (multi_rank) {
            datafile.read_subset(
                reinterpret_cast<data_type*>(ptc_buffer.host_ptr()), number,
                name + "_" + entry, offset, number, 0);
          } else {
            datafile.read_array(
                reinterpret_cast<data_type*>(ptc_buffer.host_ptr()), number,
                name + "_" + entry);
          }
          ptc_buffer.copy_to_device();
          // ptr_copy(exec_tags::device{}, ptc_buffer.dev_ptr(),
          // reinterpret_cast<uint64_t*>(u), number, 0,
          auto ptr = adapt(exec_tag{}, ptc_buffer);
          ptr_copy(typename ExecPolicy<Conf>::exec_tag{}, ptr,
                   reinterpret_cast<decltype(ptr)>(u), number, 0, 0);
          // for (int i = 0; i < number; i++) {
          //   Logger::print_debug("{} is {}", entry, u[i]);
          // }
        };
        if (sizeof(data_type) == 4) {
          read_component(ptc_buffer32);
        } else if (sizeof(data_type) == 8) {
          read_component(ptc_buffer64);
        }
      });
  data.set_num(number);
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write(particle_data_t& data,
                                       const std::string& name,
                                       H5File& datafile, bool snapshot) {
  if (snapshot) {
    write_ptc_snapshot(data, name, datafile);
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write(photon_data_t& data,
                                       const std::string& name,
                                       H5File& datafile, bool snapshot) {
  if (snapshot) {
    write_ptc_snapshot(data, name, datafile);
  }
}

template <typename Conf, template <class> class ExecPolicy>
template <int N>
void
data_exporter<Conf, ExecPolicy>::write(field_t<N, Conf>& data,
                                       const std::string& name,
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
      // Logger::print_info_all(
      //     "ext_total[0] is {}, ext[0] is {}, pos_src[0] is {}, pos_dst[0] is "
      //     "{}",
      //     ext_total[0], ext[0], pos_src[0], pos_dst[0]);

      data[i].copy_to_host();
      if (is_multi_rank()) {
        datafile.write_parallel(data[i], ext_total, pos_dst, ext, pos_src,
                                namestr);
      } else {
        datafile.write(data[i], namestr);
      }
    }
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write(rng_states_t<exec_tag>& data,
                                       const std::string& name,
                                       H5File& datafile, bool snapshot) {
  // No point writing rng states for regular data output
  if (!snapshot) {
    return;
  }
  int n_ranks = 1;
  int rank = 0;
  if (is_multi_rank()) {
    n_ranks = m_comm->size();
    rank = m_comm->rank();
  }
  // The number 4 is due to our rand_state having 4 uint64_t numbers as state
  // variables
  size_t len_total = data.size() * 4 * n_ranks;
  size_t len = data.size() * 4;
  size_t pos_src = 0;
  size_t pos_dst = data.size() * 4 * rank;

  data.copy_to_host();
  // Writing everything as a 1D plain array
  if (is_multi_rank()) {
    datafile.write_parallel(
        reinterpret_cast<uint64_t*>(data.states().host_ptr()), data.size() * 4,
        len_total, pos_dst, len, pos_src, name);
  } else {
    datafile.write(reinterpret_cast<uint64_t*>(data.states().host_ptr()),
                   data.size() * 4, name);
  }
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::write(momentum_space<Conf>& data,
                                       const std::string& name,
                                       H5File& datafile, bool snapshot) {
  // No point including this in the snapshot
  if (snapshot) {
    return;
  }

  data.copy_to_host();

  // First figure out the extent and offset of each node
  extent_t<Conf::dim + 1> ext_total(
      data.m_num_bins[0], m_global_ext * m_downsample / data.m_downsample);
  extent_t<Conf::dim + 1> ext(data.m_num_bins[0], data.m_grid_ext);
  // ext.get_strides();
  // ext_total.get_strides();
  index_t<Conf::dim + 1> idx_src(0);
  index_t<Conf::dim + 1> idx_dst(0);
  // FIXME: This again assumes a uniform grid, which is no good in the long term
  if (is_multi_rank()) {
    for (int i = 0; i < Conf::dim + 1; i++) {
      if (i > 0)
        idx_dst[i] = m_comm->domain_info().mpi_coord[i - 1] * ext[i];
      else
        idx_dst[i] = 0;
    }
    // Logger::print_info_all("idx_dst is {}, {}, {}, {}; ext_total is {}, {},
    // {}, {}",
    //                    idx_dst[0], idx_dst[1], idx_dst[2], idx_dst[3],
    //                    ext_total[0], ext_total[1], ext_total[2],
    //                    ext_total[3]);

    datafile.write_parallel(data.e_p1, ext_total, idx_dst, ext, idx_src,
                            name + "_p1_e");
    datafile.write_parallel(data.p_p1, ext_total, idx_dst, ext, idx_src,
                            name + "_p1_p");
    ext[0] = ext_total[0] = data.m_num_bins[1];
    ext.get_strides();
    ext_total.get_strides();
    datafile.write_parallel(data.e_p2, ext_total, idx_dst, ext, idx_src,
                            name + "_p2_e");
    datafile.write_parallel(data.p_p2, ext_total, idx_dst, ext, idx_src,
                            name + "_p2_p");
    ext[0] = ext_total[0] = data.m_num_bins[2];
    ext.get_strides();
    ext_total.get_strides();
    datafile.write_parallel(data.e_p3, ext_total, idx_dst, ext, idx_src,
                            name + "_p3_e");
    datafile.write_parallel(data.p_p3, ext_total, idx_dst, ext, idx_src,
                            name + "_p3_p");
    ext[0] = ext_total[0] = data.m_num_bins[3];
    ext.get_strides();
    ext_total.get_strides();
    datafile.write_parallel(data.e_E, ext_total, idx_dst, ext, idx_src,
                            name + "_E_e");
    datafile.write_parallel(data.p_E, ext_total, idx_dst, ext, idx_src,
                            name + "_E_p");
  } else {
    datafile.write(data.e_p1, name + "_p1_e");
    datafile.write(data.p_p1, name + "_p1_p");
    ext[0] = ext_total[0] = data.m_num_bins[1];
    datafile.write(data.e_p2, name + "_p2_e");
    datafile.write(data.p_p2, name + "_p2_p");
    ext[0] = ext_total[0] = data.m_num_bins[2];
    datafile.write(data.e_p3, name + "_p3_e");
    datafile.write(data.p_p3, name + "_p3_p");
    ext[0] = ext_total[0] = data.m_num_bins[3];
    datafile.write(data.e_E, name + "_E_e");
    datafile.write(data.p_E, name + "_E_p");
  }
}

template <typename Conf, template <class> class ExecPolicy>
template <int N>
void
data_exporter<Conf, ExecPolicy>::write(phase_space<Conf, N>& data,
                                       const std::string& name,
                                       H5File& datafile, bool snapshot) {
  // Should not include this in the snapshot either
  if (snapshot) {
    return;
  }

  data.copy_to_host();

  // First figure out the extent and offset of each node
  // extent_t<Conf::dim + N> ext_total(
  //     data.m_num_bins[0], m_global_ext * m_downsample / data.m_downsample);
  // extent_t<Conf::dim + N> ext(data.m_num_bins[0], data.m_grid_ext);
  extent_t<Conf::dim + N> ext_total;
  extent_t<Conf::dim + N> ext;
  for (int i = 0; i < Conf::dim + N; i++) {
    if (i < N) {
      ext_total[i] = data.m_num_bins[i];
      ext[i] = data.m_num_bins[i];
    } else {
      ext_total[i] = m_global_ext[i - N] * m_downsample / data.m_downsample;
      ext[i] = data.m_grid_ext[i - N];
    }
  }
  ext.get_strides();
  ext_total.get_strides();

  index_t<Conf::dim + N> idx_src(0);
  index_t<Conf::dim + N> idx_dst(0);
  // FIXME: This again assumes a uniform grid, which is no good in the long term
  if (is_multi_rank()) {
    for (int i = 0; i < Conf::dim + N; i++) {
      if (i < N) {
        idx_dst[i] = 0;
      } else {
        idx_dst[i] = m_comm->domain_info().mpi_coord[i - N] * ext[i];
      }
    }
    // Logger::print_info_all("idx_dst is {}, {}, {}, {}; ext_total is {}, {},
    // {}, {}",
    //                    idx_dst[0], idx_dst[1], idx_dst[2], idx_dst[3],
    //                    ext_total[0], ext_total[1], ext_total[2],
    //                    ext_total[3]);

    datafile.write_parallel(data.data, ext_total, idx_dst, ext, idx_src, name);
  } else {
    datafile.write(data.data, name);
  }
}

template <typename Conf, template <class> class ExecPolicy>
template <int N>
void
data_exporter<Conf, ExecPolicy>::write(phase_space_vlasov<Conf, N>& data,
                                       const std::string& name,
                                       H5File& datafile, bool snapshot) {
  // Should not include this in the snapshot either
  if (snapshot) {
    return;
  }

  data.copy_to_host();

  // First figure out the extent and offset of each node
  extent_t<Conf::dim + N> ext_total;
  extent_t<Conf::dim + N> ext;
  for (int i = 0; i < Conf::dim + N; i++) {
    if (i < N) {
      ext_total[i] = data.m_num_bins[i];
      ext[i] = data.m_num_bins[i];
    } else {
      ext_total[i] = m_global_ext[i - N];
      ext[i] = data.m_grid_ext[i - N] - 2 * m_grid.guard[i - N];
    }
  }
  ext.get_strides();
  ext_total.get_strides();

  index_t<Conf::dim + N> idx_src{};
  for (int i = 0; i < Conf::dim; i++) {
    idx_src[i + N] = m_grid.guard[i];
  }
  index_t<Conf::dim + N> idx_dst{};
  // FIXME: This again assumes a uniform grid, which is no good in the long term
  if (is_multi_rank()) {
    for (int i = 0; i < Conf::dim + N; i++) {
      if (i < N) {
        idx_dst[i] = 0;
      } else {
        idx_dst[i] = m_comm->domain_info().mpi_coord[i - N] * ext[i];
      }
    }
    // Logger::print_info_all("idx_dst is {}, {}; ext_total is {}, {}", idx_dst[0],
    //                        idx_dst[1], idx_dst[2], idx_dst[3], ext_total[0],
    //                        ext_total[1], ext_total[2], ext_total[3]);

    datafile.write_parallel(data.data, ext_total, idx_dst, ext, idx_src, name);
  } else {
    datafile.write(data.data, name);
  }
}

template <typename Conf, template <class> class ExecPolicy>
template <typename T>
void
data_exporter<Conf, ExecPolicy>::write(scalar_data<T>& data,
                                       const std::string& name,
                                       H5File& datafile, bool snapshot) {
  // The same behavior for snapshot or not
  if (is_multi_rank() && data.do_gather()) {
    m_comm->gather_to_root(data.data());
  } else {
    data.copy_to_host();
  }
  if (is_root()) {
    datafile.write(data.data(), name);
  }
}

template <typename Conf, template <class> class ExecPolicy>
template <typename T, int Rank>
void
data_exporter<Conf, ExecPolicy>::write(multi_array_data<T, Rank>& data,
                                       const std::string& name,
                                       H5File& datafile, bool snapshot) {
  // This makes most sense when not doing snapshot
  if (!snapshot) {
    if (is_multi_rank()) {
      if (data.gather_to_root) {
        // gather_to_root automatically takes care of copying to host
        m_comm->gather_to_root(static_cast<buffer<T>&>(data));
      }
    } else {
      data.copy_to_host();
    }
    // if (is_root()) {
    //   extent_t<Rank> domain_ext = data.extent();
    //   Logger::print_info("{}", data[2]);
    //   for (auto idx : range(default_idx_t(0, domain_ext),
    //                         default_idx_t(domain_ext.size(), domain_ext))) {
    //     index_t<Rank> pos = get_pos(idx, domain_ext);
    //     Logger::print_info("Coord ({}, {}, {}) has number {}",
    //                        pos[0], pos[1], pos[2], data[idx]);
    //   }
    // }

    // gather_to_root only touches host memory, so we can directly use it to
    // write output
    datafile.write(static_cast<multi_array<T, Rank>&>(data), name);
  } else {
    int ranks = 1;
    if (is_multi_rank()) {
      ranks = m_comm->size();
    }
    extent_t<Rank + 1> ext_total, ext;
    index_t<Rank + 1> pos_src, pos_dst;
  }
}

template <typename Conf, template <class> class ExecPolicy>
template <typename BufferType>
void
data_exporter<Conf, ExecPolicy>::write(tracked_ptc<BufferType>& data,
                                       const std::string& name,
                                       H5File& datafile, bool snapshot) {
  // No need to specifically write tracked_ptc into snapshot
  if (!snapshot) {
    timer::stamp("write_tracked_ptc");
    size_t number = data.number();
    size_t total = number;
    size_t offset = 0;
    bool multi_rank = is_multi_rank();
    if (multi_rank) {
      m_comm->get_total_num_offset(number, total, offset);
      if (total == 0) return;
      datafile.write_parallel(data.x1.host_ptr(), data.size(), total, offset,
                              number, 0, name + "_x1");
      datafile.write_parallel(data.x2.host_ptr(), data.size(), total, offset,
                              number, 0, name + "_x2");
      datafile.write_parallel(data.x3.host_ptr(), data.size(), total, offset,
                              number, 0, name + "_x3");
      datafile.write_parallel(data.p1.host_ptr(), data.size(), total, offset,
                              number, 0, name + "_p1");
      datafile.write_parallel(data.p2.host_ptr(), data.size(), total, offset,
                              number, 0, name + "_p2");
      datafile.write_parallel(data.p3.host_ptr(), data.size(), total, offset,
                              number, 0, name + "_p3");
      datafile.write_parallel(data.E.host_ptr(), data.size(), total, offset,
                              number, 0, name + "_E");
      datafile.write_parallel(data.flag.host_ptr(), data.size(), total, offset,
                              number, 0, name + "_flag");
      datafile.write_parallel(data.weight.host_ptr(), data.size(), total,
                              offset, number, 0, name + "_weight");
      datafile.write_parallel(data.id.host_ptr(), data.size(), total, offset,
                              number, 0, name + "_id");
    } else {
      if (total == 0) return;
      datafile.write(data.x1.host_ptr(), number, name + "_x1");
      datafile.write(data.x2.host_ptr(), number, name + "_x2");
      datafile.write(data.x3.host_ptr(), number, name + "_x3");
      datafile.write(data.p1.host_ptr(), number, name + "_p1");
      datafile.write(data.p2.host_ptr(), number, name + "_p2");
      datafile.write(data.p3.host_ptr(), number, name + "_p3");
      datafile.write(data.E.host_ptr(), number, name + "_E");
      datafile.write(data.flag.host_ptr(), number, name + "_flag");
      datafile.write(data.weight.host_ptr(), number, name + "_weight");
      datafile.write(data.id.host_ptr(), number, name + "_id");
    }
    timer::show_duration_since_stamp("write_tracked_ptc", "ms",
                                     "write_tracked_ptc");
  }
}

template <typename Conf, template <class> class ExecPolicy>
template <int N>
void
data_exporter<Conf, ExecPolicy>::read(field_t<N, Conf>& data,
                                      const std::string& name, H5File& datafile,
                                      bool snapshot) {
  // Loop over all components, reading them from file
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

      if (is_multi_rank()) {
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

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::read(rng_states_t<exec_tag>& data,
                                      const std::string& name, H5File& datafile,
                                      bool snapshot) {
  // No point reading rng states for regular data output
  if (!snapshot) {
    return;
  }
  int n_ranks = 1;
  int rank = 0;
  if (is_multi_rank()) {
    n_ranks = m_comm->size();
    rank = m_comm->rank();
  }
  // The number 4 is due to our rand_state having 4 uint64_t numbers as state
  // variables
  size_t len_total = data.size() * 4 * n_ranks;
  size_t len = data.size() * 4;
  size_t pos_src = data.size() * 4 * rank;
  size_t pos_dst = 0;

  if (is_multi_rank()) {
    datafile.read_subset(reinterpret_cast<uint64_t*>(data.states().host_ptr()),
                         data.size() * 4, name, pos_src, len, pos_dst);
  } else {
    datafile.read_array(reinterpret_cast<uint64_t*>(data.states().host_ptr()),
                        data.size() * 4, name);
  }
  data.copy_to_host();
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::compute_snapshot_ext_offset(
    extent_t<Conf::dim>& ext_total, extent_t<Conf::dim>& ext,
    index_t<Conf::dim>& pos_array, index_t<Conf::dim>& pos_file) {
  // Figure out the total ext and individual ext, and individual offsets for
  // snaphsot output
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
        // m_comm->domain_info().neighbor_left[n] == MPI_PROC_NULL) {
        m_comm->domain_info().mpi_coord[n] == 0) {
      ext[n] += m_grid.guard[n];
    }
    if (m_comm != nullptr &&
        // m_comm->domain_info().neighbor_right[n] == MPI_PROC_NULL) {
        m_comm->domain_info().mpi_coord[n] == m_comm->domain_info().mpi_dims[n] - 1) {
      ext[n] += m_grid.guard[n];
    }
    ext_total[n] += 2 * m_grid.guard[n];
  }
  // Logger::print_info_all("ext_total: {}x{}x{}", ext_total[0], ext_total[1],
  // ext_total[2]); Logger::print_info_all("ext: {}x{}x{}", ext[0], ext[1],
  // ext[2]); Logger::print_info_all("pos_file: {}x{}x{}", pos_file[0],
  // pos_file[1], pos_file[2]); Logger::print_info_all("pos_array: {}x{}x{}",
  // pos_array[0], pos_array[1], pos_array[2]);
  ext.get_strides();
  ext_total.get_strides();
}

template <typename Conf, template <class> class ExecPolicy>
void
data_exporter<Conf, ExecPolicy>::compute_ext_offset(
    const extent_t<Conf::dim>& ext_total, const extent_t<Conf::dim>& ext,
    int downsample, index_t<Conf::dim>& offsets) const {}

// INSTANTIATE_WITH_CONFIG(data_exporter);

}  // namespace Aperture
