#ifndef _PTC_UPDATER_H_
#define _PTC_UPDATER_H_

#include "algorithms/pushers.hpp"
#include "core/enum_types.h"
#include "data/particle_data.h"
#include "data/curand_states.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/domain_comm.h"
#include "systems/grid.h"
#include "utils/interpolation.hpp"

namespace Aperture {

template <typename Conf>
class ptc_updater : public system_t {
 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf>* m_comm = nullptr;

  Pusher m_pusher = Pusher::higuera;
  typedef bspline<1> spline_t;

  particle_data_t* ptc;
  vector_field<Conf> *E, *B, *J;
  std::vector<scalar_field<Conf>*> Rho;

  std::unique_ptr<typename Conf::multi_array_t> jtmp;

  // Parameters for this module
  uint32_t m_num_species = 2;
  uint32_t m_data_interval = 1;
  uint32_t m_sort_interval = 20;
  uint32_t m_filter_times = 1;

  // By default the maximum number of species is 8
  float m_charges[max_ptc_types];
  float m_masses[max_ptc_types];
  float m_q_over_m[max_ptc_types];

  void init_charge_mass() {
    // Default values are 1.0
    double q_e = 1.0, ion_mass = 1.0;
    m_env.params().get_value("q_e", q_e);
    m_env.params().get_value("ion_mass", ion_mass);

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
  typedef typename Conf::value_t value_t;
  static std::string name() { return "ptc_updater"; }

  ptc_updater(sim_environment& env, const grid_t<Conf>& grid,
              const domain_comm<Conf>* comm = nullptr)
      : system_t(env), m_grid(grid), m_comm(comm) {}

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_dependencies() override;

  void move_and_deposit(double dt, uint32_t step);
  void filter_current(int n_times, uint32_t step);

  template <typename P>
  void push(double dt);
  // void move(double dt);

  virtual void push_default(double dt);
  virtual void move_deposit_1d(double dt, uint32_t step);
  virtual void move_deposit_2d(double dt, uint32_t step);
  virtual void move_deposit_3d(double dt, uint32_t step);
  virtual void clear_guard_cells();
  virtual void sort_particles();
  virtual void filter_field(vector_field<Conf>& f, int comp);
  virtual void filter_field(scalar_field<Conf>& f);
  virtual void fill_multiplicity(int n, value_t weight = 1.0);

  void use_pusher(Pusher p) {
    m_pusher = p;
  }
};

template <typename Conf>
class ptc_updater_cu : public ptc_updater<Conf> {
 protected:
  buffer<ndptr<Scalar, Conf::dim>> m_rho_ptrs;
  curand_states_t* m_rand_states;

 public:
  typedef buffer<ndptr<Scalar, Conf::dim>> rho_ptrs_t;

  static std::string name() { return "ptc_updater"; }

  typedef ptc_updater<Conf> base_class;

  ptc_updater_cu(sim_environment& env, const grid_t<Conf>& grid,
                 const domain_comm<Conf>* comm = nullptr) :
      ptc_updater<Conf>(env, grid, comm) {}

  void init() override;
  // void update(double dt, uint32_t step);
  void register_dependencies() override;

  template <typename P>
  void push(double dt);
  // void move_and_deposit(double dt, uint32_t step);

  virtual void push_default(double dt) override;
  virtual void move_deposit_1d(double dt, uint32_t step) override;
  virtual void move_deposit_2d(double dt, uint32_t step) override;
  virtual void move_deposit_3d(double dt, uint32_t step) override;
  virtual void clear_guard_cells() override;
  virtual void sort_particles() override;
  virtual void filter_field(vector_field<Conf>& f, int comp) override;
  virtual void filter_field(scalar_field<Conf>& f) override;
  virtual void fill_multiplicity(int n, typename Conf::value_t weight = 1.0) override;

  rho_ptrs_t& get_rho_ptrs() { return m_rho_ptrs; }
};

}  // namespace Aperture

#endif  // _PTC_UPDATER_H_
