#include "algorithms/pushers.hpp"
#include "core/constant_mem_func.h"
#include "framework/config.h"
#include "ptc_updater.h"
#include "utils/interpolation.hpp"
#include "utils/range.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
void
ptc_updater<Conf>::init() {
  init_charge_mass();

  Etmp = vector_field<Conf>(m_grid);
  Btmp = vector_field<Conf>(m_grid);
}

template <typename Conf>
void
ptc_updater<Conf>::register_dependencies() {
  size_t max_ptc_num = 1000000;
  get_from_store("max_ptc_num", max_ptc_num, m_env.params());

  ptc = m_env.register_data<particle_data_t>("particles", max_ptc_num,
                                             MemType::host_only);

  E = m_env.register_data<vector_field<Conf>>("E", m_grid,
                                              field_type::edge_centered,
                                              MemType::host_only);
  B = m_env.register_data<vector_field<Conf>>("B", m_grid,
                                              field_type::face_centered,
                                              MemType::host_only);
  J = m_env.register_data<vector_field<Conf>>("J", m_grid,
                                              field_type::edge_centered,
                                              MemType::host_only);

  get_from_store("num_species", m_num_species, m_env.params());
  Rho.resize(m_num_species);
  for (int i = 0; i < m_num_species; i++) {
    Rho[i] = m_env.register_data<scalar_field<Conf>>(
        std::string("Rho_") + ptc_type_name(i), m_grid,
        field_type::vert_centered,
        MemType::host_only);
  }
}

template <typename Conf>
void
ptc_updater<Conf>::update(double dt, uint32_t step) {
  // First update particle momentum
  if (m_pusher == Pusher::boris) {
    push<boris_pusher>(dt);
  } else if (m_pusher == Pusher::vay) {
    push<vay_pusher>(dt);
  } else if (m_pusher == Pusher::higuera) {
    push<higuera_pusher>(dt);
  }
}

template <typename Conf>
template <typename T>
void
ptc_updater<Conf>::push(double dt) {
  // TODO: First interpolate E and B fields to vertices and store them in Etmp
  // and Btmp

  auto num = ptc->number();
  auto ext = m_grid.extent();
  T pusher;
  if (num > 0) {
    for (auto n : range(0ul, num)) {
      uint32_t cell = ptc->cell[n];
      if (cell == empty_cell) continue;
      auto idx = (*E)[0].idx_at(cell);
      auto pos = idx.get_pos();

      auto interp = interpolator<bspline<1>, Conf::dim>{};
      auto flag = ptc->flag[n];
      int sp = get_ptc_type(flag);

      Scalar qdt_over_2m = dt * 0.5f * m_q_over_m[sp];

      auto x = vec_t<Pos_t, 3>(ptc->x1[n], ptc->x2[n], ptc->x3[n]);
      //  Grab E & M fields at the particle position
      Scalar E1 = interp((*E)[0], x, idx, pos);
      Scalar E2 = interp((*E)[1], x, idx, pos);
      Scalar E3 = interp((*E)[2], x, idx, pos);
      Scalar B1 = interp((*B)[0], x, idx, pos);
      Scalar B2 = interp((*B)[1], x, idx, pos);
      Scalar B3 = interp((*B)[2], x, idx, pos);

      Logger::print_debug("E1 {}, E2 {}, E3 {}, B1 {}, B2 {}, B3 {}",
                          E1, E2, E3, B1, B2, B3);

      //  Push particles
      Scalar p1 = ptc->p1[n], p2 = ptc->p2[n], p3 = ptc->p3[n],
          gamma = ptc->E[n];
      if (!check_flag(flag, PtcFlag::ignore_EM)) {
        pusher(p1, p2, p3, gamma, E1, E2, E3, B1, B2, B3,
               qdt_over_2m, (Scalar)dt);
      }

      // if (dev_params.rad_cooling_on && sp != (int)ParticleType::ion) {
      //   sync_kill_perp(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3,
      //                  q_over_m);
      // }
      ptc->p1[n] = p1;
      ptc->p2[n] = p2;
      ptc->p3[n] = p3;
      ptc->E[n] = gamma;

      if (p1 != p1 || p2 != p2 || p3 != p3) {
        printf(
            "NaN detected after push! p1 is %f, p2 is %f, p3 is %f, gamma "
            "is %f\n",
            p1, p2, p3, gamma);
        exit(1);
      }
    }
  }
}

template <typename Conf>
void
ptc_updater<Conf>::move(double dt) {
  // TODO: First interpolate E and B fields to vertices and store them in Etmp
  // and Btmp

  auto num = ptc->number();
  if (num > 0) {
    auto ext = m_grid.extent();

    for (auto n : range(0ul, num)) {
      uint32_t cell = ptc->cell[n];
      if (cell == empty_cell) continue;
      auto idx = (*E)[0].idx_at(cell);
      auto pos = idx.get_pos();

      //  Move particles
      auto x1 = ptc->x1[n], x2 = ptc->x2[n], x3 = ptc->x3[n];
      Scalar v1 = ptc->p1[n], v2 = ptc->p2[n], v3 = ptc->p3[n],
          gamma = ptc->E[n];

      v1 /= gamma;
      v2 /= gamma;
      v3 /= gamma;

      auto new_x1 = x1 + (v1 * dt) * m_grid.inv_delta[0];
      int dc1 = std::floor(new_x1);
      pos[0] += dc1;
      ptc->x1[n] = new_x1 - (Pos_t)dc1;

      if constexpr (Conf::dim > 1) {
          auto new_x2 = x2 + (v2 * dt) * m_grid.inv_delta[1];
          int dc2 = std::floor(new_x2);
          pos[1] += dc2;
          ptc->x2[n] = new_x2 - (Pos_t)dc2;
        } else {
        ptc->x2[n] = x2 + v2 * dt;
      }

      if constexpr (Conf::dim > 2) {
          auto new_x3 = x3 + (v3 * dt) * m_grid.inv_delta[2];
          int dc3 = std::floor(new_x3);
          pos[2] += dc3;
          ptc->x3[n] = new_x3 - (Pos_t)dc3;
        } else {
        ptc->x3[n] = x3 + v3 * dt;
      }

      ptc->cell[n] = (*E)[0].idx_at(pos).linear;
    }
  }
}

template class ptc_updater<Config<1>>;
template class ptc_updater<Config<2>>;
template class ptc_updater<Config<3>>;

}  // namespace Aperture
