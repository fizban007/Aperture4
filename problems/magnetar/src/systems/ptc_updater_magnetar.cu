#include "ptc_updater_magnetar.h"
#include "framework/config.h"
#include "systems/grid_sph.h"
#include "systems/forces/sync_cooling.hpp"
#include "systems/forces/gravity.hpp"
#include "systems/helpers/ptc_update_helper.hpp"
#include "utils/kernel_helper.hpp"

namespace Aperture {

template <typename Pusher>
struct pusher_impl_magnetar {
  Pusher pusher;
  double cooling_coef = 0.0, B0 = 1.0, g0 = 0.1;
  bool gravity_on = false;

  pusher_impl_magnetar(sim_environment& env) {
    env.params().get_value("sync_cooling_coef", cooling_coef);
    env.params().get_value("Bp", B0);
    env.params().get_value("gravity", g0);
    env.params().get_value("gravity_on", gravity_on);
  }

  HOST_DEVICE pusher_impl_magnetar(const pusher_impl_magnetar<Pusher>& other) = default;

  template <typename Scalar>
  __device__ void operator()(ptc_ptrs& ptc, uint32_t n, EB_t<Scalar>& EB,
                            Scalar qdt_over_2m, Scalar dt) {
    using Conf = Config<2>;
    Scalar p1 = ptc.p1[n], p2 = ptc.p2[n], p3 = ptc.p3[n];
    Scalar gamma = ptc.E[n];

    pusher(p1, p2, p3, gamma, EB.E1, EB.E2,
           EB.E3, EB.B1, EB.B2, EB.B3, qdt_over_2m, dt);

    if (gravity_on) {
      auto& grid = dev_grid<2>();
      auto ext = grid.extent();
      auto idx = Conf::idx(ptc.cell[n], ext);
      auto pos = idx.get_pos();
      Scalar r = grid_sph_t<Conf>::radius(grid.template pos<0>(pos[0], ptc.x1[n]));
      gravity(p1, p2, p3, gamma, r, dt, (Scalar)g0, qdt_over_2m * 2.0f / dt);
    }

    sync_kill_perp(p1, p2, p3, gamma, EB.E1, EB.E2, EB.E3,
                   EB.B1, EB.B2, EB.B3, qdt_over_2m * 2.0f / dt,
                   (Scalar)cooling_coef, (Scalar)B0);

    ptc.p1[n] = p1;
    ptc.p2[n] = p2;
    ptc.p3[n] = p3;
    ptc.E[n] = gamma;
  }
};

template <typename Conf>
ptc_updater_magnetar<Conf>::ptc_updater_magnetar(sim_environment& env,
                                                 const grid_sph_t<Conf>& grid,
                                                 const domain_comm<Conf>* comm)
    : ptc_updater_sph_cu<Conf>(env, grid, comm) {}

template <typename Conf>
void
ptc_updater_magnetar<Conf>::init() {
  ptc_updater_sph_cu<Conf>::init();

  m_impl_boris = std::make_unique<pusher_impl_magnetar<boris_pusher>>(this->m_env);
  m_impl_vay = std::make_unique<pusher_impl_magnetar<vay_pusher>>(this->m_env);
  m_impl_higuera = std::make_unique<pusher_impl_magnetar<higuera_pusher>>(this->m_env);
}

template <typename Conf>
void
ptc_updater_magnetar<Conf>::push_default(double dt) {
  // dispatch according to enum. This will also instantiate all the versions of
  // push
  if (this->m_pusher == Pusher::boris) {
    this->push(dt, *m_impl_boris);
  } else if (this->m_pusher == Pusher::vay) {
    this->push(dt, *m_impl_vay);
  } else if (this->m_pusher == Pusher::higuera) {
    this->push(dt, *m_impl_higuera);
  }
}

#include "systems/ptc_updater_cu_impl.hpp"

template class ptc_updater_magnetar<Config<2>>;

}  // namespace Aperture
