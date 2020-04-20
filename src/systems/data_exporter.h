#ifndef _DATA_EXPORTER_H_
#define _DATA_EXPORTER_H_

#include "core/multi_array.hpp"
#include "framework/system.h"
#include "systems/domain_comm.hpp"
#include "systems/grid.h"
#include "utils/hdf_wrapper.h"
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

namespace Aperture {

class sim_environment;

template <typename Conf>
class data_exporter : public system_t {
 private:
  const domain_comm<Conf>* m_comm;
  const grid_t<Conf>& m_grid;

  std::ofstream m_xmf;  //!< This is the accompanying xmf file
                        //!< describing the hdf structure
  std::string m_dim_str;
  std::string m_xmf_buffer;

  /// tmp_ptc_data stores temporary tracked particles
  buffer_t<double> tmp_ptc_data;
  /// tmp_grid_data stores the temporary downsampled data for output
  multi_array<float, Conf::dim> tmp_grid_data;

  /// Sets the directory of all the data files
  std::string m_output_dir = "Data/";

  int m_output_num = 0;
  int m_ptc_output_interval = 1;
  int m_fld_output_interval = 1;
  int m_snapshot_interval = 1;
  int m_downsample = 1;
  extent_t<Conf::dim> m_local_ext;
  index_t<Conf::dim> m_local_offset;
  extent_t<Conf::dim> m_global_ext;
  stagger_t m_output_stagger = stagger_t(0b000);

  void copy_config_file();

 public:
  data_exporter(sim_environment& env, const grid_t<Conf>& grid,
                const domain_comm<Conf>* comm = nullptr);
  virtual ~data_exporter();

  static std::string name() { return "data_exporter"; }

  void init();
  void update(double time, uint32_t step);

  void write_grid();
  void write_xmf_head(std::ofstream& fs);
  // void write_xmf_step_header(std::ofstream& fs, double time);
  void write_xmf_step_header(std::string& buffer, double time);
  // void write_xmf_step_close(std::ofstream& fs);
  void write_xmf_step_close(std::string& buffer);
  // void write_xmf_tail(std::ofstream& fs);
  void write_xmf_tail(std::string& buffer);
  void write_xmf_field_entry(std::string& buffer, int num, const std::string& name);
  void write_xmf(uint32_t step, double time);
  void prepare_xmf_restart(uint32_t restart_step, int data_interval,
                           float time);
  // void write_output(sim_data& data, uint32_t timestep, double time);

  // void write_field_output(sim_data& data, uint32_t timestep,
  //                         double time);
  // void write_ptc_output(sim_data& data, uint32_t timestep, double time);
  void write_grid_multiarray(const std::string& name,
                             const typename Conf::multi_array_t& array,
                             stagger_t stagger, H5File& file);

  // template <typename T, int Dim>
  // void write_multi_array(const multi_array<T, Dim>& array,
  //                        const std::string& name,
  //                        const extent_t<Dim>& total_ext,
  //                        const index_t<Dim>& offset, H5File& file);

  // template <typename Func>
  // void add_grid_output(sim_data& data, const std::string& name, Func f,
  //                      H5File& file, uint32_t timestep);

  // template <typename T>
  // void add_grid_output(multi_array<T>& array, Stagger stagger,
  //                      const std::string& name, H5File& file,
  //                      uint32_t timestep);

  // template <typename Ptc>
  // void add_ptc_output(Ptc& data, size_t num, H5File& file,
  //                     const std::string& prefix);

  // template <typename Ptc>
  // void read_ptc_output(Ptc& data, size_t num, H5File& file,
  //                      const std::string& prefix);

  // template <typename T, typename Func>
  // void add_tracked_ptc_output(sim_data& data, int sp,
  //                             const std::string& name,
  //                             uint64_t total, uint64_t offset,
  //                             Func f, H5File& file);

  // void save_snapshot(const std::string& filename, sim_data& data,
  //                    uint32_t step, Scalar time);
  // void load_snapshot(const std::string& filename, sim_data& data,
  //                    uint32_t& step, Scalar& time);

  // buffer_t<float>& grid_buffer() { return tmp_grid_data; }
  // buffer_t<double>& ptc_buffer() { return tmp_ptc_data; }
};

}  // namespace Aperture

#endif  // _DATA_EXPORTER_H_
