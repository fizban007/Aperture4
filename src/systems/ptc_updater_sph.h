#ifndef _PTC_UPDATER_SPH_H_
#define _PTC_UPDATER_SPH_H_

#include "ptc_updater.h"
#include "grid_sph.h"

namespace Aperture {

template <typename Conf>
class ptc_updater_sph_cu : public ptc_updater_cu<Conf> {
 protected:
  typename Conf::value_t m_compactness = 0.0;
  typename Conf::value_t m_omega = 0.0;
  int m_damping_length = 32;
  float m_r_cutoff = 10.0;

 public:
  typedef typename Conf::value_t value_t;
  static std::string name() { return "ptc_updater"; }

  ptc_updater_sph_cu(sim_environment& env, const grid_sph_t<Conf>& grid,
                        const domain_comm<Conf>* comm = nullptr) :
      ptc_updater_cu<Conf>(env, grid, comm) {}

  void init() override;
  void register_data_components() override;

  virtual void move_deposit_2d(value_t dt, uint32_t step) override;
  virtual void move_photons_2d(value_t dt, uint32_t step) override;
  virtual void filter_field(vector_field<Conf>& f, int comp) override;
  virtual void filter_field(scalar_field<Conf>& f) override;
};


}

#endif  // _PTC_UPDATER_SPH_H_
