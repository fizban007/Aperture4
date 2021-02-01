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

#ifndef _DATA_EXPORTER_H_
#define _DATA_EXPORTER_H_

#include "core/multi_array.hpp"
#include "data/fields.h"
#include "data/momentum_space.hpp"
#include "framework/system.h"
#include "systems/domain_comm.h"
#include "systems/grid.h"
#include "utils/hdf_wrapper.h"
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

namespace Aperture {

// class sim_environment;
class particle_data_t;
class curand_states_t;
template <typename T, int Rank>
class multi_array_data;

template <typename Conf>
class data_exporter : public system_t {
 public:
  // data_exporter(sim_environment& env, const grid_t<Conf>& grid,
  data_exporter(const grid_t<Conf>& grid,
                const domain_comm<Conf>* comm = nullptr);
  virtual ~data_exporter();

  static std::string name() { return "data_exporter"; }

  void init() override;
  void update(double time, uint32_t step) override;
  void register_data_components() override;

  void write_grid();
  void write_xmf_head(std::ofstream& fs);
  void write_xmf_step_header(std::string& buffer, double time);
  void write_xmf_step_close(std::string& buffer);
  void write_xmf_tail(std::string& buffer);
  void write_xmf_field_entry(std::string& buffer, int num,
                             const std::string& name);
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

  void write_snapshot(const std::string& filename, uint32_t step, double time);
  void load_snapshot(const std::string& filename, uint32_t& step, double& time);

  bool is_root() const {
    if (m_comm != nullptr)
      return m_comm->is_root();
    else
      return true;
  }

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

  // buffer<float>& grid_buffer() { return tmp_grid_data; }
  // buffer<double>& ptc_buffer() { return tmp_ptc_data; }
  void write(particle_data_t& data, const std::string& name, H5File& datafile,
             bool snapshot = false);
  template <int N>
  void write(field_t<N, Conf>& data, const std::string& name, H5File& datafile,
             bool snapshot = false);
  void write(momentum_space<Conf>& data, const std::string& name, H5File& datafile,
             bool snapshot = false);
  void write(curand_states_t& data, const std::string& name, H5File& datafile,
             bool snapshot = false);
  template <typename T, int Rank>
  void write(multi_array_data<T, Rank>& data, const std::string& name,
             H5File& datafile, bool snapshot = false);
  void read(particle_data_t& data, const std::string& name, H5File& datafile,
            bool snapshot = false);
  template <int N>
  void read(field_t<N, Conf>& data, const std::string& name, H5File& datafile,
            bool snapshot = false);
  void read(curand_states_t& data, const std::string& name, H5File& datafile,
            bool snapshot = false);
  template <typename T, int Rank>
  void read(multi_array_data<T, Rank>& data, const std::string& name,
            H5File& datafile, bool snapshot = false);

 private:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf>* m_comm;

  std::ofstream m_xmf;  //!< This is the accompanying xmf file
                        //!< describing the hdf structure
  std::string m_dim_str;
  std::string m_xmf_buffer;

  /// tmp_ptc_data stores temporary tracked particles
  buffer<double> tmp_ptc_data;
  /// tmp_grid_data stores the temporary downsampled data for output
  multi_array<float, Conf::dim> tmp_grid_data;
  grid_t<Conf> m_output_grid;

  /// Sets the directory of all the data files
  std::string m_output_dir = "Data/";

  int m_fld_num = 0;
  int m_ptc_num = 0;
  int m_ptc_output_interval = 1;
  int m_fld_output_interval = 1;
  int m_snapshot_interval = 1;
  int m_downsample = 1;
  // extent_t<Conf::dim> m_local_ext;
  // index_t<Conf::dim> m_local_offset;
  extent_t<Conf::dim> m_global_ext;
  stagger_t m_output_stagger = stagger_t(0b000);

  void copy_config_file();
  void compute_snapshot_ext_offset(extent_t<Conf::dim>& ext_total,
                                   extent_t<Conf::dim>& ext,
                                   index_t<Conf::dim>& pos_array,
                                   index_t<Conf::dim>& pos_file);
  void compute_ext_offset(const extent_t<Conf::dim>& ext_total,
                          const extent_t<Conf::dim>& ext,
                          int downsample,
                          index_t<Conf::dim>& offsets) const;
  void write_multi_array_helper(
      const std::string& name,
      const multi_array<float, Conf::dim, typename Conf::idx_t>& array,
      const extent_t<Conf::dim>& global_ext, const index_t<Conf::dim>& offsets,
      H5File& file);
};

}  // namespace Aperture

#endif  // _DATA_EXPORTER_H_
