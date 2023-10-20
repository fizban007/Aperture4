#pragma once

#include "systems/vlasov_solver.h"
#include "core/detail/multi_array_helpers.h"


namespace Aperture {

template <typename Conf,
          int Dim_P,
          template <class> class ExecPolicy,
          template <class> class CoordPolicy>
vlasov_solver<Conf, Dim_P, ExecPolicy, CoordPolicy>::vlasov_solver(const grid_t<Conf>& grid,
                                                                   const domain_comm<Conf, ExecPolicy>* comm) :
                                                                   m_grid(grid), m_comm(comm) {
  // Read parameters
  sim_env().params().get_vec_t("momentum_ext", m_momentum_ext);
  sim_env().params().get_vec_t("momentum_lower", m_momentum_lower);
  sim_env().params().get_vec_t("momentum_upper", m_momentum_upper);

  for (int i = 0; i < Dim_P; i++) {
    m_momentum_delta[i] = (m_momentum_upper[i] - m_momentum_lower[i]) / m_momentum_ext[i];
  }

  sim_env().params().get_value("num_species", m_num_species);
  if (m_num_species > max_ptc_types) {
    Logger::print_err("Too many species of particles requested! Aborting");
    throw std::runtime_error("too many species");
  }

  for (int i = 0; i < Conf::dim + Dim_P; i++) {
    if (i < Dim_P) {
      m_ext_total[i] = m_momentum_ext[i];
    } else {
      m_ext_total[i] = m_grid.dims[i - Dim_P];
    }
  }

  df_tmp = std::make_unique<phase_space<Conf, Dim_P>>(m_grid, 1, m_momentum_ext.data(),
                m_momentum_lower.data(), m_momentum_upper.data());
}

template <typename Conf,
          int Dim_P,
          template <class> class ExecPolicy,
          template <class> class CoordPolicy>
vlasov_solver<Conf, Dim_P, ExecPolicy, CoordPolicy>::~vlasov_solver() {}

template <typename Conf,
          int Dim_P,
          template <class> class ExecPolicy,
          template <class> class CoordPolicy>
void
vlasov_solver<Conf, Dim_P, ExecPolicy, CoordPolicy>::init() {
}

template <typename Conf,
          int Dim_P,
          template <class> class ExecPolicy,
          template <class> class CoordPolicy>
void
vlasov_solver<Conf, Dim_P, ExecPolicy, CoordPolicy>::update(double dt, uint32_t step) {
  auto& E = *(this->E);
  auto& B = *(this->B);
  auto& J = *(this->J);
  auto& ext_total = m_ext_total;
  auto& p_lower = m_momentum_lower;
  auto& p_upper = m_momentum_upper;
  auto& p_delta = m_momentum_delta;
  auto num_species = m_num_species;
  using idx_t = default_idx_t<Conf::dim + Dim_P>;

  J.init();

  for (int sp = 0; sp < num_species; sp++) {
    // Launch kernel to update f
    ExecPolicy<Conf>::launch(
        [&] LAMBDA(auto f, auto df, auto E, auto J) {
          auto& grid = ExecPolicy<Conf>::grid();
          auto ext = grid.extent();
          ExecPolicy<Conf>::loop(
            idx_t(0, ext_total),
            idx_t(ext_total.size(), ext_total),
            [&] LAMBDA(auto idx) {
              // pos is a 2D index, first index in p, second index in x
              auto pos = get_pos(idx, ext_total);
              typename Conf::idx_t idx_x(pos[1], ext);
              auto pos_x = get_pos(idx_x, ext);

              if (grid.is_in_bound(pos_x)) {

                // Update f in x, 0th direction
                value_t p = p_lower[0] + pos[0] * p_delta[0];
                value_t v = p / std::sqrt(1.0 + p*p);
                if (v > 0.0) {
                  df[idx] = dt * v * (f[sp][idx.dec_y()] - f[sp][idx]) * grid.inv_delta[0];
                } else {
                  df[idx] = dt * v * (f[sp][idx.inc_y()]- f[sp][idx]) * grid.inv_delta[0];
                }

                // Update f in p
                if (E[0][idx_x] > 0.0) {
                  df[idx] += dt * E[0][idx_x] * (f[sp][idx.inc_x()] - f[sp][idx]) / p_delta[0];
                } else {
                  df[idx] += dt * E[0][idx_x] * (f[sp][idx.dec_x()] - f[sp][idx]) / p_delta[0];
                }

                // // add df to f
                // f[sp][idx] += df[idx];

                //update J
                // J[0][idx_x] -= v * (f[sp][idx] + df[idx]);
                atomic_add(&J[0][idx_x], -v * (f[sp][idx] + df[idx]));

                //update E
                // E[0][idx_x] -= J[0][idx_x]* dt;
              }
            });
        }, f, *df_tmp, E, J);
    ExecPolicy<Conf>::sync();

    add(typename ExecPolicy<Conf>::exec_tag{}, f[sp]->data, df_tmp->data, index_t<Conf::dim + Dim_P>{},
        index_t<Conf::dim + Dim_P>{}, m_ext_total);
  }

}

template <typename Conf,
          int Dim_P,
          template <class> class ExecPolicy,
          template <class> class CoordPolicy>
void
vlasov_solver<Conf, Dim_P, ExecPolicy, CoordPolicy>::register_data_components() {
  // Declare required data components here
  f.resize(m_num_species);
  for (int i = 0; i < m_num_species; i++) {
    f.set(i,
        sim_env().register_data<phase_space<Conf, Dim_P>>(
                std::string("f_") + ptc_type_name(i), m_grid, 1, m_momentum_ext.data(),
                m_momentum_lower.data(), m_momentum_upper.data()));
    f[i]->include_in_snapshot(true);
  }

  E = sim_env().register_data<vector_field<Conf>>(
      "E", m_grid, field_type::edge_centered,
      ExecPolicy<Conf>::data_mem_type());
//   E->include_in_snapshot(true);
  B = sim_env().register_data<vector_field<Conf>>(
      "B", m_grid, field_type::face_centered,
      ExecPolicy<Conf>::data_mem_type());
//   B->include_in_snapshot(true);
  J = sim_env().register_data<vector_field<Conf>>(
      "J", m_grid, field_type::edge_centered,
      ExecPolicy<Conf>::data_mem_type());
  J->include_in_snapshot(true);

}


}
