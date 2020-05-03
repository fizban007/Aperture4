#include "radiative_transfer.h"
#include "framework/environment.h"
#include <exception>

namespace Aperture {

template <typename Conf>
radiative_transfer<Conf>::radiative_transfer(sim_environment& env,
                                             const grid_t<Conf>& grid,
                                             const domain_comm<Conf>* comm)
    : system_t(env), m_grid(grid), m_comm(comm) {}

template <typename Conf>
void
radiative_transfer<Conf>::init() {}

template <typename Conf>
void
radiative_transfer<Conf>::register_dependencies() {
  m_env.get_data("particles", &ptc);
  if (ptc == nullptr)
    throw std::runtime_error("Data component 'particles' is not found!");

  size_t max_ph_num = 10000;
  m_env.params().get_value("max_ph_num", max_ph_num);

  ph = m_env.register_data<photon_data_t>("photons", max_ph_num,
                                          MemType::host_only);
  rho_ph = m_env.register_data<scalar_field<Conf>>(
      "Rho_ph", m_grid, field_type::vert_centered, MemType::host_only);
}

template <typename Conf>
void
radiative_transfer<Conf>::update(double dt, uint32_t step) {}

template <typename Conf>
void
radiative_transfer<Conf>::move_photons(double dt) {}

template <typename Conf>
void
radiative_transfer<Conf>::emit_photons(double dt) {}

template <typename Conf>
void
radiative_transfer<Conf>::produce_pairs(double dt) {}

}  // namespace Aperture
