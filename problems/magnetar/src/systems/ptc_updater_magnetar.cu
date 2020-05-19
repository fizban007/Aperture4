#include "ptc_updater_magnetar.h"

namespace Aperture {

template <typename Pusher>
struct update_momentum_magnetar {
  template <typename Scalar>
  HD_INLINE void operator()(Scalar& p1, Scalar& p2, Scalar& p3, Scalar& gamma,
                            Scalar E1, Scalar E2, Scalar E3, Scalar B1,
                            Scalar B2, Scalar B3, Scalar qdt_over_2m,
                            Scalar dt) {
    Pusher::operator()(p1, p2, p3, gamma, E1, E2, E3, B1, B2, B3,
                       qdt_over_2m, dt);
  }
};

template <typename Conf>
ptc_updater_magnetar<Conf>::ptc_updater_magnetar(sim_environment& env,
                                                 const grid_sph_t<Conf>& grid,
                                                 const domain_comm<Conf>* comm)
    : ptc_updater_sph_cu<Conf>(env, grid, comm) {}

template <typename Conf>
void
ptc_updater_magnetar<Conf>::push_default(double dt) {
  // dispatch according to enum. This will also instantiate all the versions of
  // push
  if (this->m_pusher == Pusher::boris) {
    this->template push<update_momentum_magnetar<boris_pusher>>(dt);
  } else if (this->m_pusher == Pusher::vay) {
    this->template push<update_momentum_magnetar<vay_pusher>>(dt);
  } else if (this->m_pusher == Pusher::higuera) {
    this->template push<update_momentum_magnetar<higuera_pusher>>(dt);
  }
}

}  // namespace Aperture
