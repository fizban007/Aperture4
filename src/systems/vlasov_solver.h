#pragma once

#include <string>
#include "data/data_array.hpp"
#include "data/phase_space_vlasov.hpp"
#include "framework/system.h"
#include "systems/grid.h"
#include "systems/domain_comm.h"
#include "utils/nonown_ptr.hpp"

namespace Aperture {

template <typename Conf,
          int Dim_P,
          template <class> class ExecPolicy,
          template <class> class CoordPolicy>
class vlasov_solver : public system_t {
 public:
  using value_t = typename Conf::value_t;
  static std::string name() { return "vlasov_solver"; }

  vlasov_solver(const grid_t<Conf>& grid,
                const domain_comm<Conf, ExecPolicy>* comm = nullptr);
  ~vlasov_solver();

  void init() override;
  void update(double dt, uint32_t step) override;
  void register_data_components() override;

 protected:
  const grid_t<Conf>& m_grid;
  const domain_comm<Conf, ExecPolicy>* m_comm = nullptr;

  // Data components
  data_array<phase_space_vlasov<Conf, Dim_P>> f;
  nonown_ptr<vector_field<Conf>> E, B, J;
  // data_array<scalar_field<Conf>> Rho;

  // Temp data
  std::unique_ptr<phase_space_vlasov<Conf, Dim_P>> df_tmp;

  // Parameters
  uint32_t m_num_species = 2;
  uint32_t m_data_interval = 1;
  uint32_t m_filter_times = 0;

  vec_t<value_t, max_ptc_types> m_charges;
  vec_t<value_t, max_ptc_types> m_masses;

  vec_t<int, Dim_P> m_momentum_ext;
  vec_t<value_t, Dim_P> m_momentum_lower;
  vec_t<value_t, Dim_P> m_momentum_upper;
  vec_t<value_t, Dim_P> m_momentum_delta;
  extent_t<Conf::dim + Dim_P> m_ext_total;
};

}
