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
class radiative_transfer : public system_t {
 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf>& m_comm;

  particle_data_t* ptc;
  photon_data_t* ph;

  scalar_field<Conf>* rho_ph;
 
 public:
  static std::string name() { return "radiative_transfer"; }

  radiative_transfer(sim_environment& env, const grid_t<Conf>& grid,
                     const domain_comm<Conf>* comm = nullptr);
  virtual ~radiative_transfer() {}

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_dependencies() override;

  virtual void move_photons(double dt);
  virtual void emit_photons(double dt);
  virtual void produce_pairs(double dt);
};

template <typename Conf>
class radiative_transfer_cu : public radiative_transfer<Conf> {
 protected:
  curand_states_t* m_rand_states;
  buffer<int> m_num_per_block;
  buffer<int> m_cum_num_per_block;
  buffer<int> m_pos_in_block;

 public:
  static std::string name() { return "radiative_transfer"; }

  radiative_transfer_cu(sim_environment& env, const grid_t<Conf>& grid,
                        const domain_comm<Conf>* comm = nullptr);
  virtual ~radiative_transfer_cu() {}

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_dependencies() override;

  virtual void move_photons(double dt) override;
  virtual void emit_photons(double dt) override;
  virtual void produce_pairs(double dt) override;
};

}

#endif  // _RADIATIVE_TRANSFER_H_
