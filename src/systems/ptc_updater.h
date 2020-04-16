#ifndef _PTC_UPDATER_H_
#define _PTC_UPDATER_H_

#include "algorithms/pushers.hpp"
#include "core/enum_types.h"
#include "data/particle_data.h"
#include "framework/environment.hpp"
#include "framework/parse_params.hpp"
#include "framework/system.h"
#include "systems/domain_comm.hpp"
#include "systems/grid.h"

namespace Aperture {

template <typename Conf>
class ptc_updater : public system_t {
 protected:
  const grid_t<Conf>& m_grid;
  std::shared_ptr<const domain_comm<Conf>> m_comm = nullptr;

  Pusher m_pusher = Pusher::higuera;

  std::shared_ptr<particle_data_t> ptc;
  std::shared_ptr<vector_field<Conf>> E, B, J;
  std::vector<std::shared_ptr<scalar_field<Conf>>> Rho;

  vector_field<Conf> Etmp, Btmp;

  uint32_t m_num_species = 2;
  // By default the maximum number of species is 8
  float m_charges[max_ptc_types];
  float m_masses[max_ptc_types];
  float m_q_over_m[max_ptc_types];

  void init_charge_mass() {
    // Default values are 1.0
    double q_e = 1.0, ion_mass = 1.0;
    get_from_store("q_e", q_e, m_env.params());
    get_from_store("ion_mass", ion_mass, m_env.params());

    for (int i = 0; i < (max_ptc_types); i++) {
      m_charges[i] = q_e;
      m_masses[i] = q_e;
    }
    m_charges[(int)PtcType::electron] *= -1.0;
    m_masses[(int)PtcType::ion] *= ion_mass;
    for (int i = 0; i < (max_ptc_types); i++) {
      m_q_over_m[i] = m_charges[i] / m_masses[i];
    }
  }

 public:
  static std::string name() { return "ptc_updater"; }

  ptc_updater(sim_environment& env, const grid_t<Conf>& grid) :
      system_t(env), m_grid(grid) {}

  ptc_updater(sim_environment& env, const grid_t<Conf>& grid,
              std::shared_ptr<const domain_comm<Conf>> comm)
      : system_t(env), m_grid(grid), m_comm(comm) {}

  void init();
  void update(double dt, uint32_t step);
  void register_dependencies();

  template <typename P>
  void push(double dt);
  void move(double dt);
  void move_and_deposit(double dt, uint32_t step);

  void use_pusher(Pusher p) {
    m_pusher = p;
  }
};

template <typename Conf>
class ptc_updater_cu : public ptc_updater<Conf> {
 public:
  ptc_updater_cu(sim_environment& env, const grid_t<Conf>& grid) :
      ptc_updater<Conf>(env, grid) {}

  ptc_updater_cu(sim_environment& env, const grid_t<Conf>& grid,
                 std::shared_ptr<const domain_comm<Conf>> comm) :
      ptc_updater<Conf>(env, grid, comm) {}

  void init();
  void update(double dt, uint32_t step);
  void register_dependencies();

  template <typename P>
  void push(double dt);
  void move_and_deposit(double dt, uint32_t step);
};

}  // namespace Aperture

#endif  // _PTC_UPDATER_H_
