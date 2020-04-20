#include "data_exporter.h"
#include "framework/environment.hpp"
#include "framework/config.h"
#include "data/particle_data.h"
#include <boost/filesystem.hpp>
#include <fmt/ostream.h>

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
data_exporter<Conf>::~data_exporter() {}

template <typename Conf>
void
data_exporter<Conf>::init() {

  // make sure output directory is a directory
  if (m_output_dir.back() != '/') m_output_dir.push_back('/');
  boost::filesystem::path outPath(m_output_dir);

  boost::system::error_code returnedError;
  boost::filesystem::create_directories(outPath, returnedError);

  // Copy config file to the output directory
  copy_config_file();

  // Write the grid in the simulation to the output directory
}

template <typename Conf>
void
data_exporter<Conf>::update(double time, uint32_t step) {
  if (step % m_fld_output_interval == 0) {
    // Output downsampled fields!

    for (auto& it : m_env.data_map()) {
      // Logger::print_info("Working on {}", it.first);
      auto data = it.second.get();
      if (auto ptr = dynamic_cast<vector_field<Conf>*>(data)) {
        Logger::print_info("Writing vector field {}", it.first);
      } else if (auto ptr = dynamic_cast<scalar_field<Conf>*>(data)) {
        Logger::print_info("Writing scalar field {}", it.first);
      }
    }
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
  }

  m_output_num += 1;
}

template <typename Conf>
void
data_exporter<Conf>::copy_config_file() {
  std::string path = m_output_dir + "config.toml";
  std::string conf_file = m_env.params().get_as<std::string>("config_file");
  Logger::print_info("Copying config file from {} to {}", conf_file, path);
  boost::filesystem::copy_file(
      conf_file, path, boost::filesystem::copy_option::overwrite_if_exists);
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
    m_dim_str =
        fmt::format("{} {} {}", m_global_ext[2],
                    m_global_ext[1], m_global_ext[0]);
  } else if (Conf::dim == 2) {
    m_dim_str = fmt::format("{} {}", m_global_ext[1],
                            m_global_ext[0]);
  } else if (Conf::dim == 1) {
    m_dim_str = fmt::format("{} 1", m_global_ext[0]);
  }

  buffer += "<Grid Name=\"quadmesh\" Type=\"Uniform\">\n";
  buffer +=
      fmt::format("  <Time Type=\"Single\" Value=\"{}\"/>\n", time);
  if (Conf::dim == 3) {
    buffer += fmt::format(
        "  <Topology Type=\"3DSMesh\" NumberOfElements=\"{}\"/>\n",
        m_dim_str);
    buffer += "  <Geometry GeometryType=\"X_Y_Z\">\n";
  } else if (Conf::dim == 2) {
    buffer += fmt::format(
        "  <Topology Type=\"2DSMesh\" NumberOfElements=\"{}\"/>\n",
        m_dim_str);
    buffer += "  <Geometry GeometryType=\"X_Y\">\n";
  } else if (Conf::dim == 1) {
    buffer += fmt::format(
        "  <Topology Type=\"2DSMesh\" NumberOfElements=\"{}\"/>\n",
        m_dim_str);
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

template class data_exporter<Config<1>>;
template class data_exporter<Config<2>>;
template class data_exporter<Config<3>>;

}  // namespace Aperture
