#include "ptc_updater.h"
#include "algorithms/pushers.hpp"
#include "core/constant_mem_func.h"
#include "core/detail/multi_array_helpers.h"
#include "framework/config.h"
#include "helpers/ptc_update_helper.hpp"
#include "utils/double_buffer.h"
#include "utils/range.hpp"
#include "utils/util_functions.h"
#include "utils/timer.h"
#include <random>

namespace Aperture {

template <typename Conf>
ptc_updater<Conf>::ptc_updater(sim_environment& env, const grid_t<Conf>& grid,
                               const domain_comm<Conf>* comm) :
    system_t(env), m_grid(grid), m_comm(comm) {
  m_env.params().get_value("data_interval", m_data_interval);
  m_env.params().get_value("sort_interval", m_sort_interval);
  m_env.params().get_value("current_smoothing", m_filter_times);
  init_charge_mass();

  auto pusher = m_env.params().get_as<std::string>("pusher");

  if (pusher == "boris") {
    m_pusher = Pusher::boris;
  } else if (pusher == "vay") {
    m_pusher = Pusher::vay;
  } else if (pusher == "higuera") {
    m_pusher = Pusher::higuera;
  }
}

template <typename Conf>
void
ptc_updater<Conf>::init() {
  // Allocate the tmp array for current filtering
  jtmp = std::make_unique<typename Conf::multi_array_t>(m_grid.extent(), MemType::host_only);

  m_env.get_data_optional("photons", &ph);
}

template <typename Conf>
void
ptc_updater<Conf>::register_dependencies() {
  size_t max_ptc_num = 10000;
  m_env.params().get_value("max_ptc_num", max_ptc_num);

  ptc = m_env.register_data<particle_data_t>("particles", max_ptc_num,
                                             MemType::host_only);

  E = m_env.register_data<vector_field<Conf>>(
      "E", m_grid, field_type::edge_centered, MemType::host_only);
  B = m_env.register_data<vector_field<Conf>>(
      "B", m_grid, field_type::face_centered, MemType::host_only);
  J = m_env.register_data<vector_field<Conf>>(
      "J", m_grid, field_type::edge_centered, MemType::host_only);

  m_env.params().get_value("num_species", m_num_species);
  Rho.resize(m_num_species);
  for (int i = 0; i < m_num_species; i++) {
    Rho[i] = m_env.register_data<scalar_field<Conf>>(
        std::string("Rho_") + ptc_type_name(i), m_grid,
        field_type::vert_centered, MemType::host_only);
  }
}

template <typename Conf>
void
ptc_updater<Conf>::push_default(double dt) {
  // dispatch according to enum
  if (m_pusher == Pusher::boris) {
    push<boris_pusher>(dt);
  } else if (m_pusher == Pusher::vay) {
    push<vay_pusher>(dt);
  } else if (m_pusher == Pusher::higuera) {
    push<higuera_pusher>(dt);
  }
}

template <typename Conf>
void
ptc_updater<Conf>::update(double dt, uint32_t step) {
  Logger::print_info("Pushing {} particles", ptc->number());
  timer::stamp("pusher");
  // First update particle momentum
  push_default(dt);
  timer::show_duration_since_stamp("push", "ms", "pusher");

  timer::stamp("depositer");
  // Then move particles and deposit current
  move_and_deposit(dt, step);
  timer::show_duration_since_stamp("deposit", "ms", "depositer");

  // Communicate deposited current and charge densities
  if (m_comm != nullptr) {
    m_comm->send_add_guard_cells(*J);
    m_comm->send_guard_cells(*J);
    if ((step + 1) % m_data_interval == 0) {
      for (uint32_t i = 0; i < Rho.size(); i++) {
        m_comm->send_add_guard_cells(*(Rho[i]));
        m_comm->send_guard_cells(*(Rho[i]));
      }
    }
  }

  timer::stamp("filter");
  // Filter current
  filter_current(m_filter_times, step);
  timer::show_duration_since_stamp("filter", "ms", "filter");

  // Send particles
  if (m_comm != nullptr) {
    m_comm->send_particles(*ptc);
  }

  // Clear guard cells
  clear_guard_cells();

  // sort at the given interval
  if ((step % m_sort_interval) == 0) {
    sort_particles();
  }
}

template <typename Conf>
template <typename T>
void
ptc_updater<Conf>::push(double dt) {
  auto num = ptc->number();
  auto ext = m_grid.extent();
  T pusher;
  if (num > 0) {
    for (auto n : range(0, num)) {
      uint32_t cell = ptc->cell[n];
      if (cell == empty_cell) continue;
      auto idx = E->at(0).idx_at(cell);
      // auto pos = idx.get_pos();

      auto interp = interpolator<spline_t, Conf::dim>{};
      auto flag = ptc->flag[n];
      int sp = get_ptc_type(flag);

      Scalar qdt_over_2m = dt * 0.5f * m_q_over_m[sp];

      auto x = vec_t<Pos_t, 3>(ptc->x1[n], ptc->x2[n], ptc->x3[n]);
      //  Grab E & M fields at the particle position
      Scalar E1 = interp(E->at(0), x, idx, stagger_t(0b110));
      Scalar E2 = interp(E->at(1), x, idx, stagger_t(0b101));
      Scalar E3 = interp(E->at(2), x, idx, stagger_t(0b011));
      Scalar B1 = interp(B->at(0), x, idx, stagger_t(0b001));
      Scalar B2 = interp(B->at(1), x, idx, stagger_t(0b010));
      Scalar B3 = interp(B->at(2), x, idx, stagger_t(0b100));

      // Logger::print_debug("E1 {}, E2 {}, E3 {}, B1 {}, B2 {}, B3 {}",
      //                     E1, E2, E3, B1, B2, B3);

      //  Push particles
      Scalar p1 = ptc->p1[n], p2 = ptc->p2[n], p3 = ptc->p3[n],
             gamma = ptc->E[n];
      if (!check_flag(flag, PtcFlag::ignore_EM)) {
        pusher(p1, p2, p3, gamma, E1, E2, E3, B1, B2, B3, qdt_over_2m,
               (Scalar)dt);
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
ptc_updater<Conf>::move_and_deposit(double dt, uint32_t step) {
  if (Conf::dim == 1)
    move_deposit_1d(dt, step);
  else if (Conf::dim == 2)
    move_deposit_2d(dt, step);
  else if (Conf::dim == 3)
    move_deposit_3d(dt, step);
}

template <typename Conf>
void
ptc_updater<Conf>::move_deposit_1d(double dt, uint32_t step) {
  auto num = ptc->number();
  if (num > 0) {
    auto ext = m_grid.extent();

    for (auto n : range(0, num)) {
      uint32_t cell = ptc->cell[n];
      if (cell == empty_cell) continue;
      auto idx = (*E)[0].idx_at(cell);
      auto pos = idx.get_pos();

      // step 1: Move particles
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
      ptc->x2[n] = x2 + v2 * dt;
      ptc->x3[n] = x3 + v3 * dt;

      ptc->cell[n] = m_grid.get_idx(pos).linear;

      // step 2: Deposit current
      auto flag = ptc->flag[n];
      auto sp = get_ptc_type(flag);
      auto interp = spline_t{};
      if (check_flag(flag, PtcFlag::ignore_current)) continue;
      auto weight = m_charges[sp] * ptc->weight[n];

      int i_0 = (dc1 == -1 ? -spline_t::radius : 1 - spline_t::radius);
      int i_1 = (dc1 == 1 ? spline_t::radius + 1 : spline_t::radius);
      Scalar djx = 0.0f;
      for (int i = i_0; i <= i_1; i++) {
        Scalar sx0 = interp(-x1 + i);
        Scalar sx1 = interp(-new_x1 + i);

        // j1 is movement in x1
        int offset = i + pos[0] - dc1;
        djx += sx1 - sx0;
        (*J)[0][offset] += -weight * djx;
        // Logger::print_debug("J0 is {}", (*J)[0][offset]);

        // j2 is simply v2 times rho at center
        Scalar val1 = 0.5f * (sx0 + sx1);
        (*J)[1][offset] += weight * v2 * val1;

        // j3 is simply v3 times rho at center
        (*J)[2][offset] += weight * v3 * val1;

        // rho is deposited at the final position
        if ((step + 1) % m_data_interval == 0) {
          (*Rho[sp])[0][offset] += weight * sx1;
        }
      }
    }
  }
}

template <typename Conf>
void
ptc_updater<Conf>::move_deposit_2d(double dt, uint32_t step) {
  auto num = ptc->number();
  if (num > 0) {
    auto ext = m_grid.extent();

    for (auto n : range(0, num)) {
      uint32_t cell = ptc->cell[n];
      if (cell == empty_cell) continue;
      auto idx = (*E)[0].idx_at(cell);
      auto pos = idx.get_pos();

      // step 1: Move particles
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

      auto new_x2 = x2 + (v2 * dt) * m_grid.inv_delta[1];
      int dc2 = std::floor(new_x2);
      pos[1] += dc2;
      ptc->x2[n] = new_x2 - (Pos_t)dc2;
      ptc->x3[n] = x3 + v3 * dt;

      ptc->cell[n] = m_grid.get_idx(pos).linear;

      // step 2: Deposit current
      auto flag = ptc->flag[n];
      auto sp = get_ptc_type(flag);
      auto interp = spline_t{};
      if (check_flag(flag, PtcFlag::ignore_current)) continue;
      auto weight = m_charges[sp] * ptc->weight[n];

      int i_0 = (dc1 == -1 ? -spline_t::radius : 1 - spline_t::radius);
      int i_1 = (dc1 == 1 ? spline_t::radius + 1 : spline_t::radius);
      int j_0 = (dc2 == -1 ? -spline_t::radius : 1 - spline_t::radius);
      int j_1 = (dc2 == 1 ? spline_t::radius + 1 : spline_t::radius);
      Scalar djy[2 * spline_t::radius + 1] = {};
      for (int j = j_0; j <= j_1; j++) {
        Scalar sy0 = interp(-x2 + j);
        Scalar sy1 = interp(-new_x2 + j);

        Scalar djx = 0.0f;
        for (int i = i_0; i <= i_1; i++) {
          Scalar sx0 = interp(-x1 + i);
          Scalar sx1 = interp(-new_x1 + i);

          // j1 is movement in x1
          auto offset = idx.inc_x(i).inc_y(j);
          djx += movement2d(sy0, sy1, sx0, sx1);
          (*J)[0][offset] += -weight * djx;

          // j2 is movement in x2
          djy[i - i_0] += movement2d(sx0, sx1, sy0, sy1);
          (*J)[1][offset] += -weight * djy[i - i_0];
          // Logger::print_debug("J1 is {}", (*J)[1][offset]);

          // j3 is simply v3 times rho at center
          (*J)[2][offset] += weight * v3 * center2d(sx0, sx1, sy0, sy1);

          // rho is deposited at the final position
          if ((step + 1) % m_data_interval == 0) {
            (*Rho[sp])[0][offset] += weight * sx1 * sy1;
          }
        }
      }
    }
  }
}

template <typename Conf>
void
ptc_updater<Conf>::move_deposit_3d(double dt, uint32_t step) {
  auto num = ptc->number();
  if (num > 0) {
    auto ext = m_grid.extent();

    for (auto n : range(0, num)) {
      uint32_t cell = ptc->cell[n];
      if (cell == empty_cell) continue;
      auto idx = (*E)[0].idx_at(cell);
      auto pos = idx.get_pos();

      // step 1: Move particles
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

      auto new_x2 = x2 + (v2 * dt) * m_grid.inv_delta[1];
      int dc2 = std::floor(new_x2);
      pos[1] += dc2;
      ptc->x2[n] = new_x2 - (Pos_t)dc2;

      auto new_x3 = x3 + (v3 * dt) * m_grid.inv_delta[2];
      int dc3 = std::floor(new_x3);
      pos[2] += dc3;
      ptc->x3[n] = new_x3 - (Pos_t)dc3;

      ptc->cell[n] = m_grid.get_idx(pos).linear;

      // step 2: Deposit current
      auto flag = ptc->flag[n];
      auto sp = get_ptc_type(flag);
      auto interp = spline_t{};
      if (check_flag(flag, PtcFlag::ignore_current)) continue;
      auto weight = m_charges[sp] * ptc->weight[n];

      int i_0 = (dc1 == -1 ? -spline_t::radius : 1 - spline_t::radius);
      int i_1 = (dc1 == 1 ? spline_t::radius + 1 : spline_t::radius);
      int j_0 = (dc2 == -1 ? -spline_t::radius : 1 - spline_t::radius);
      int j_1 = (dc2 == 1 ? spline_t::radius + 1 : spline_t::radius);
      int k_0 = (dc3 == -1 ? -spline_t::radius : 1 - spline_t::radius);
      int k_1 = (dc3 == 1 ? spline_t::radius + 1 : spline_t::radius);

      Scalar djz[2 * spline_t::radius + 1][2 * spline_t::radius + 1] = {};
      for (int k = k_0; k <= k_1; k++) {
        Scalar sz0 = interp(-x3 + k);
        Scalar sz1 = interp(-new_x3 + k);

        Scalar djy[2 * spline_t::radius + 1] = {};
        for (int j = j_0; j <= j_1; j++) {
          Scalar sy0 = interp(-x2 + j);
          Scalar sy1 = interp(-new_x2 + j);

          Scalar djx = 0.0f;
          for (int i = i_0; i <= i_1; i++) {
            Scalar sx0 = interp(-x1 + i);
            Scalar sx1 = interp(-new_x1 + i);

            // j1 is movement in x1
            auto offset = idx.inc_x(i).inc_y(j).inc_z(k);
            djx += movement3d(sy0, sy1, sz0, sz1, sx0, sx1);
            (*J)[0][offset] += -weight * djx;

            // j2 is movement in x2
            djy[i - i_0] += movement3d(sz0, sz1, sx0, sx1, sy0, sy1);
            (*J)[1][offset] += -weight * djy[i - i_0];
            // Logger::print_debug("J1 is {}", (*J)[1][offset]);

            // j3 is movement in x3
            djz[j - j_0][i - i_0] += movement3d(sx0, sx1, sy0, sy1, sz0, sz1);
            (*J)[2][offset] += -weight * djz[j - j_0][i - i_0];

            // rho is deposited at the final position
            if ((step + 1) % m_data_interval == 0) {
              (*Rho[sp])[0][offset] += weight * sx1 * sy1 * sz1;
            }
          }
        }
      }
    }
  }
}

template <typename Conf>
void
ptc_updater<Conf>::clear_guard_cells() {
  for (auto n : range(0, ptc->number())) {
    uint32_t cell = ptc->cell[n];
    if (cell == empty_cell) continue;
    auto idx = typename Conf::idx_t(cell, m_grid.extent());
    auto pos = idx.get_pos();

    if (!m_grid.is_in_bound(pos)) ptc->cell[n] = empty_cell;
  }
}

template <typename Conf>
void
ptc_updater<Conf>::sort_particles() {
  ptc->sort_by_cell_host(m_grid.extent().size());
}

template <typename Conf>
void
ptc_updater<Conf>::fill_multiplicity(int mult, value_t weight) {
  auto num = ptc->number();
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (auto idx : range(m_grid.begin(), m_grid.end())) {
    auto pos = idx.get_pos();
    if (m_grid.is_in_bound(pos)) {
      for (int n = 0; n < mult; n++) {
        uint32_t offset = num + idx.linear * mult * 2 + n * 2;

        ptc->x1[offset] = ptc->x1[offset + 1] = dist(engine);
        ptc->x2[offset] = ptc->x2[offset + 1] = dist(engine);
        ptc->x3[offset] = ptc->x3[offset + 1] = dist(engine);
        ptc->p1[offset] = ptc->p1[offset + 1] = 0.0;
        ptc->p2[offset] = ptc->p2[offset + 1] = 0.0;
        ptc->p3[offset] = ptc->p3[offset + 1] = 0.0;
        ptc->E[offset] = ptc->E[offset + 1] = 1.0;
        ptc->cell[offset] = ptc->cell[offset + 1] = idx.linear;
        ptc->weight[offset] = ptc->weight[offset + 1] = weight;
        ptc->flag[offset] =
            set_ptc_type_flag(bit_or(PtcFlag::primary), PtcType::electron);
        ptc->flag[offset + 1] =
            set_ptc_type_flag(bit_or(PtcFlag::primary), PtcType::positron);
      }
    }
  }
  ptc->set_num(num + mult * 2 * m_grid.extent().size());
}

template <typename Conf>
void
ptc_updater<Conf>::filter_current(int n_times, uint32_t step) {
  Logger::print_info("Filtering current {} times", n_times);
  for (int n = 0; n < n_times; n++) {
    this->filter_field(*J, 0);
    this->filter_field(*J, 1);
    this->filter_field(*J, 2);

    if (m_comm != nullptr)
      m_comm->send_guard_cells(*J);

    if ((step + 1) % m_data_interval == 0) {
      for (int sp = 0; sp < m_num_species; sp++) {
        this->filter_field(*Rho[sp]);
        if (m_comm != nullptr)
          m_comm->send_guard_cells(*Rho[sp]);
      }
    }
  }
}

template <typename Conf>
void
ptc_updater<Conf>::filter_field(vector_field<Conf>& f, int comp) {}

template <typename Conf>
void
ptc_updater<Conf>::filter_field(scalar_field<Conf>& f) {}

template class ptc_updater<Config<1>>;
template class ptc_updater<Config<2>>;
template class ptc_updater<Config<3>>;

}  // namespace Aperture
