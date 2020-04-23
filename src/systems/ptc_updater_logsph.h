#ifndef _PTC_UPDATER_LOGSPH_H_
#define _PTC_UPDATER_LOGSPH_H_

#include "ptc_updater.h"
#include "grid_logsph.h"

namespace Aperture {

template <typename Conf>
class ptc_updater_logsph_cu : public ptc_updater_cu<Conf> {
 protected:
  typename Conf::value_t m_compactness = 0.0;
  typename Conf::value_t m_omega = 0.0;
 public:
  static std::string name() { return "ptc_updater"; }

  ptc_updater_logsph_cu(sim_environment& env, const grid_logsph_t<Conf>& grid,
                        const domain_comm<Conf>* comm = nullptr) :
      ptc_updater_cu<Conf>(env, grid, comm) {}

  void init() override;
  void register_dependencies() override;

  virtual void move_deposit_2d(double dt, uint32_t step) override;
};


}

#endif  // _PTC_UPDATER_LOGSPH_H_
