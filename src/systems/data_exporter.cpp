#include "data_exporter.h"
#include "core/detail/multi_array_helpers.h"
#include "data/particle_data.h"
#include "framework/config.h"
#include "framework/environment.hpp"
#include "framework/params_store.hpp"
#include <boost/filesystem.hpp>
#include <fmt/ostream.h>

namespace fs = boost::filesystem;

namespace Aperture {

template <typename Conf>
data_exporter<Conf>::data_exporter(sim_environment& env,
                                   const grid_t<Conf>& grid,
                                   const domain_comm<Conf>* comm)
    : system_t(env), m_grid(grid), m_comm(comm) {
  m_env.params().get_value("ptc_output_interval", m_ptc_output_interval);
  m_env.params().get_value("fld_output_interval", m_fld_output_interval);
  m_env.params().get_value("snapshot_interval", m_snapshot_interval);
  m_env.params().get_value("output_dir", m_output_dir);
  m_env.params().get_value("downsample", m_downsample);

  // Resize the tmp data array
  size_t max_ptc_num = 1, max_ph_num = 1;
  m_env.params().get_value("max_ptc_num", max_ptc_num);
  m_env.params().get_value("max_ph_num", max_ph_num);

  tmp_ptc_data.set_memtype(MemType::host_device);
  tmp_ptc_data.resize(std::max(max_ptc_num, max_ph_num));

  // Obtain the local extent and offset of the output grid
  m_local_ext = m_grid.extent_less() / m_downsample;
  for (int i = 0; i < Conf::dim; i++)
    m_local_offset[i] = m_grid.offset[i] / m_downsample;
  tmp_grid_data.resize(m_local_ext);

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

  boost::system::error_code returnedError;
  fs::create_directories(outPath, returnedError);

  // Copy config file to the output directory
  copy_config_file();

  // Write the grid in the simulation to the output directory
  write_grid();
}

template <typename Conf>
void
data_exporter<Conf>::update(double dt, uint32_t step) {
  if (step % m_fld_output_interval == 0) {
    double time = m_env.get_time();

    // Output downsampled fields!
    std::string filename =
        fmt::format("{}fld.{:05d}.h5", m_output_dir,
                    m_fld_num);
    H5File datafile = hdf_create(filename, H5CreateMode::trunc_parallel);

    if (!m_xmf.is_open() && m_comm->is_root()) {
      m_xmf.open(m_output_dir + "data.xmf");
    }
    write_xmf_step_header(m_xmf_buffer, time);

    for (auto& it : m_env.data_map()) {
      // Do not output the skipped components
      if (it.second->skip_output()) continue;

      // Logger::print_info("Working on {}", it.first);
      auto data = it.second.get();
      if (auto ptr = dynamic_cast<vector_field<Conf>*>(data)) {
        Logger::print_info("Writing vector field {}", it.first);
        // Vector field has 3 components
        for (int i = 0; i < 3; i++) {
          std::string name = it.first + std::to_string(i + 1);
          write_grid_multiarray(name, ptr->at(i),
                                ptr->stagger(i), datafile);
          write_xmf_field_entry(m_xmf_buffer, m_fld_num, name);
        }
      } else if (auto ptr = dynamic_cast<scalar_field<Conf>*>(data)) {
        Logger::print_info("Writing scalar field {}", it.first);
        // Scalar field only has one component
        std::string name = it.first;
        write_grid_multiarray(name, ptr->at(0),
                              ptr->stagger(0), datafile);
        write_xmf_field_entry(m_xmf_buffer, m_fld_num, name);
      }
    }

    datafile.close();

    if (m_comm->is_root()) {
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
}

template <typename Conf>
void
data_exporter<Conf>::copy_config_file() {
  std::string path = m_output_dir + "config.toml";
  std::string conf_file = m_env.params().get_as<std::string>("config_file");
  Logger::print_info("Copying config file from {} to {}", conf_file, path);
  fs::path conf_path(conf_file);
  if (fs::exists(conf_path)) {
    fs::copy_file(
        conf_file, path, fs::copy_option::overwrite_if_exists);
  }
}

template <typename Conf>
void
data_exporter<Conf>::write_grid() {
  std::string meshfilename = m_output_dir + "grid.h5";
  H5File meshfile =
      hdf_create(meshfilename, H5CreateMode::trunc_parallel);

  // std::vector<float> x1_array(out_ext.x);
  multi_array<float, Conf::dim> x1_array(m_local_ext, MemType::host_only);
  multi_array<float, Conf::dim> x2_array(m_local_ext, MemType::host_only);
  multi_array<float, Conf::dim> x3_array(m_local_ext, MemType::host_only);

  // All data output points are cell centers
  for (auto idx : x1_array.indices()) {
    auto p = idx.get_pos() * m_downsample + m_grid.guards();
    auto x = m_grid.cart_coord(p);

    x1_array[idx] = x[0];
    if constexpr (Conf::dim > 1)
      x2_array[idx] = x[1];
    if constexpr (Conf::dim > 2)
      x3_array[idx] = x[2];
  }

  meshfile.write_parallel(x1_array, m_global_ext, m_local_offset,
                          m_local_ext, index_t<Conf::dim>{}, "x1");
  if constexpr (Conf::dim > 1)
    meshfile.write_parallel(x2_array, m_global_ext, m_local_offset,
                            m_local_ext, index_t<Conf::dim>{}, "x2");
  if constexpr (Conf::dim > 2)
    meshfile.write_parallel(x3_array, m_global_ext, m_local_offset,
                            m_local_ext, index_t<Conf::dim>{}, "x3");

  meshfile.close();
}

template <typename Conf>
void
data_exporter<Conf>::write_xmf_head(std::ofstream& fs) {
  if (!m_comm->is_root()) return;
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
  if (!m_comm->is_root()) return;

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
  if (!m_comm->is_root()) return;
  buffer += "</Grid>\n";
}

template <typename Conf>
void
data_exporter<Conf>::write_xmf_tail(std::string& buffer) {
  if (!m_comm->is_root()) return;
  buffer += "</Grid>\n";
  buffer += "</Domain>\n";
  buffer += "</Xdmf>\n";
}

template <typename Conf>
void
data_exporter<Conf>::write_grid_multiarray(const std::string &name,
                                           const typename Conf::multi_array_t &array,
                                           stagger_t stagger, H5File &file) {
  if (array.dev_allocated() && tmp_grid_data.dev_allocated()) {
    resample_dev(array, tmp_grid_data, m_grid.guards(),
                 stagger, m_output_stagger, m_downsample);
    tmp_grid_data.copy_to_host();
  } else {
    resample(array, tmp_grid_data, m_grid.guards(),
                 stagger, m_output_stagger, m_downsample);
  }

  // Logger::print_debug("writing global_ext {}x{}", m_global_ext[0], m_global_ext[1]);
  file.write_parallel(tmp_grid_data, m_global_ext, m_local_offset,
                      m_local_ext, index_t<Conf::dim>{}, name);
}

template <typename Conf>
void
data_exporter<Conf>::write_xmf_field_entry(std::string &buffer, int num, const std::string &name) {
  if (m_comm->is_root()) {
    m_xmf_buffer += fmt::format(
        "  <Attribute Name=\"{}\" Center=\"Node\" "
        "AttributeType=\"Scalar\">\n", name);
    m_xmf_buffer += fmt::format(
        "    <DataItem Dimensions=\"{}\" NumberType=\"Float\" "
        "Precision=\"4\" Format=\"HDF\">\n", m_dim_str);
    m_xmf_buffer +=
        fmt::format("      fld.{:05d}.h5:{}\n", num, name);
    m_xmf_buffer += "    </DataItem>\n";
    m_xmf_buffer += "  </Attribute>\n";
  }
}

template class data_exporter<Config<1>>;
template class data_exporter<Config<2>>;
template class data_exporter<Config<3>>;

}  // namespace Aperture
