#ifndef _RADIATIVE_TRANSFER_H_
#define _RADIATIVE_TRANSFER_H_

#include "data/particle_data.h"
#include "data/curand_states.h"
#include "framework/system.h"
#include "systems/grid.h"
#include "systems/domain_comm.h"
#include <memory>

namespace Aperture {

template <typename Conf>
class radiative_transfer_common : public system_t {
 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf>* m_comm;

  particle_data_t* ptc;
  photon_data_t* ph;

  scalar_field<Conf>* rho_ph;
  scalar_field<Conf>* photon_produced;
  scalar_field<Conf>* pair_produced;

  // parameters for this module
  uint32_t m_data_interval = 1;
  uint32_t m_sort_interval = 20;
  int m_ph_per_scatter = 1;
  float m_tracked_fraction = 0.01;

 public:
  radiative_transfer_common(sim_environment& env, const grid_t<Conf>& grid,
                     const domain_comm<Conf>* comm = nullptr);
  virtual ~radiative_transfer_common() {}

  virtual void init() override;
  virtual void update(double dt, uint32_t step) override;
  // virtual void register_dependencies() override;

  // virtual void move_photons(double dt, uint32_t step);
  virtual void emit_photons(double dt) = 0;
  virtual void produce_pairs(double dt) = 0;
  // virtual void sort_photons();
  // virtual void clear_guard_cells();
};

template <typename Conf, typename RadImpl>
class radiative_transfer : public radiative_transfer_common<Conf> {
 protected:
  RadImpl m_rad;

 public:
  static std::string name() { return "radiative_transfer"; }

  radiative_transfer(sim_environment& env, const grid_t<Conf>& grid,
                     const domain_comm<Conf>* comm = nullptr);

  virtual void init() override;
  virtual void register_dependencies() override;

  virtual void emit_photons(double dt) override;
  virtual void produce_pairs(double dt) override;
};

template <typename Conf, typename RadImpl>
class radiative_transfer_cu : public radiative_transfer_common<Conf> {
 protected:
  RadImpl m_rad;

  curand_states_t* m_rand_states;
  buffer<int> m_num_per_block;
  buffer<int> m_cum_num_per_block;
  buffer<int> m_pos_in_block;

  vector_field<Conf>* B;
  std::vector<scalar_field<Conf>*> Rho;

  int m_threads_per_block = 512;
  int m_blocks_per_grid = 256;

 public:
  static std::string name() { return "radiative_transfer"; }

  radiative_transfer_cu(sim_environment& env, const grid_t<Conf>& grid,
                        const domain_comm<Conf>* comm = nullptr);
  virtual ~radiative_transfer_cu() {}

  void init() override;
  // void update(double dt, uint32_t step) override;
  void register_dependencies() override;

  // virtual void move_photons(double dt, uint32_t step) override;
  virtual void emit_photons(double dt) override;
  virtual void produce_pairs(double dt) override;
  // virtual void sort_photons() override;
  // virtual void clear_guard_cells() override;
};

}

#endif  // _RADIATIVE_TRANSFER_H_
