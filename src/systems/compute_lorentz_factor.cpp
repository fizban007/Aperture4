#include "compute_lorentz_factor.h"
#include "framework/environment.h"
#include "framework/config.h"
#include "framework/params_store.h"

namespace Aperture {

template <typename Conf>
void
compute_lorentz_factor<Conf>::register_data_impl(MemType type) {
  m_env.params().get_value("num_species", m_num_species);

  gamma.resize(m_num_species);
  for (int n = 0; n < m_num_species; n++) {
    gamma[n] = m_env.register_data<scalar_field<Conf>>(
        std::string("gamma_") + ptc_type_name(n), m_grid,
        field_type::cell_centered, type);
  }
}

template <typename Conf>
void
compute_lorentz_factor<Conf>::register_data_components() {
  register_data_impl(MemType::host_only);
}

template <typename Conf>
void
compute_lorentz_factor<Conf>::init() {
  m_env.get_data("particles", &ptc);
  m_env.params().get_value("fld_output_interval", m_data_interval);
}

template <typename Conf>
void
compute_lorentz_factor<Conf>::update(double dt, uint32_t step) {}

INSTANTIATE_WITH_CONFIG(compute_lorentz_factor);

}  // namespace Aperture
