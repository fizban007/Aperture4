#ifndef _PTC_UPDATER_MAGNETAR_H_
#define _PTC_UPDATER_MAGNETAR_H_

#include "systems/ptc_updater_sph.h"
#include <memory>

namespace Aperture {

template <typename Pusher>
struct pusher_impl_magnetar;

template <typename Conf>
class ptc_updater_magnetar : public ptc_updater_sph_cu<Conf> {
 public:
  typedef typename Conf::value_t value_t;
  static std::string name() { return "ptc_updater"; }

  ptc_updater_magnetar(sim_environment& env, const grid_sph_t<Conf>& grid,
                       const domain_comm<Conf>* comm = nullptr);

  virtual void push_default(double dt) override;

 protected:
  std::unique_ptr<pusher_impl_magnetar<boris_pusher>> m_impl_boris;
  std::unique_ptr<pusher_impl_magnetar<vay_pusher>> m_impl_vay;
  std::unique_ptr<pusher_impl_magnetar<higuera_pusher>> m_impl_higuera;
};

}

#endif  // _PTC_UPDATER_MAGNETAR_H_
