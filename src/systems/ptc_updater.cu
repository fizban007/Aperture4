#include "core/constant_mem_func.h"
#include "data/field_helpers.h"
#include "framework/config.h"
#include "ptc_updater.h"
#include "utils/interpolation.hpp"
#include "utils/kernel_helper.hpp"
#include "utils/range.hpp"
#include "utils/util_functions.h"

namespace Aperture {

template <typename Conf>
void
ptc_updater_cu<Conf>::init() {
  this->init_charge_mass();
  init_dev_charge_mass(this->m_charges, this->m_masses);

  this->Etmp = vector_field<Conf>(this->m_grid, MemType::device_only);
  this->Btmp = vector_field<Conf>(this->m_grid, MemType::device_only);
}

template <typename Conf>
void
ptc_updater_cu<Conf>::register_dependencies() {
  size_t max_ptc_num = 1000000;
  get_from_store("max_ptc_num", max_ptc_num, this->m_env.params());
  // Prefer device_only, but can take other possibilities if data is already
  // there
  this->ptc = this->m_env.template register_data<particle_data_t>(
      "particles", max_ptc_num, MemType::device_only);

  this->E = this->m_env.template register_data<vector_field<Conf>>(
      "E", this->m_grid, field_type::edge_centered, MemType::host_device);
  this->B = this->m_env.template register_data<vector_field<Conf>>(
      "B", this->m_grid, field_type::face_centered, MemType::host_device);
  this->J = this->m_env.template register_data<vector_field<Conf>>(
      "J", this->m_grid, field_type::edge_centered, MemType::host_device);

  get_from_store("num_species", this->m_num_species, this->m_env.params());
  this->Rho.resize(this->m_num_species);
  for (int i = 0; i < this->m_num_species; i++) {
    this->Rho[i] = this->m_env.template register_data<scalar_field<Conf>>(
        std::string("Rho_") + ptc_type_name(i), this->m_grid,
        field_type::vert_centered, MemType::host_device);
  }
}

template <typename Conf>
void
ptc_updater_cu<Conf>::update(double dt, uint32_t step) {
  if (this->m_pusher == Pusher::boris) {
    push<boris_pusher>(dt);
  } else if (this->m_pusher == Pusher::vay) {
    push<vay_pusher>(dt);
  } else if (this->m_pusher == Pusher::higuera) {
    push<higuera_pusher>(dt);
  }
}

template <typename Conf>
template <typename P>
void
ptc_updater_cu<Conf>::push(double dt) {
  // First interpolate E and B fields to vertices and store them in Etmp
  // and Btmp
  resample_dev((*(this->E))[0], this->Etmp[0], this->m_grid.guards(),
               this->E->stagger(0), this->Etmp.stagger(0));
  resample_dev((*(this->E))[1], this->Etmp[1], this->m_grid.guards(),
               this->E->stagger(1), this->Etmp.stagger(1));
  resample_dev((*(this->E))[2], this->Etmp[2], this->m_grid.guards(),
               this->E->stagger(2), this->Etmp.stagger(2));
  resample_dev((*(this->B))[0], this->Btmp[0], this->m_grid.guards(),
               this->B->stagger(0), this->Btmp.stagger(0));
  resample_dev((*(this->B))[1], this->Btmp[1], this->m_grid.guards(),
               this->B->stagger(1), this->Btmp.stagger(1));
  resample_dev((*(this->B))[2], this->Btmp[2], this->m_grid.guards(),
               this->B->stagger(2), this->Btmp.stagger(2));

  auto num = this->ptc->number();
  auto ext = this->m_grid.extent();
  P pusher;

  auto pusher_kernel = [dt, num, ext] __device__(auto ptrs, auto E, auto B,
                                                 auto pusher) {
    for (auto n : grid_stride_range(0ul, num)) {
      uint32_t cell = ptrs.cell[n];
      if (cell == empty_cell) continue;
      auto idx = E[0].idx_at(cell, ext);
      // auto pos = idx.get_pos();

      auto interp = interpolator<bspline<1>, Conf::dim>{};
      auto flag = ptrs.flag[n];
      int sp = get_ptc_type(flag);

      Scalar qdt_over_2m = dt * 0.5f * dev_charges[sp] / dev_masses[sp];

      auto x = vec_t<Pos_t, 3>(ptrs.x1[n], ptrs.x2[n], ptrs.x3[n]);
      //  Grab E & M fields at the particle position
      Scalar E1 = interp(E[0], x, idx);
      Scalar E2 = interp(E[1], x, idx);
      Scalar E3 = interp(E[2], x, idx);
      Scalar B1 = interp(B[0], x, idx);
      Scalar B2 = interp(B[1], x, idx);
      Scalar B3 = interp(B[2], x, idx);

      //  Push particles
      Scalar p1 = ptrs.p1[n], p2 = ptrs.p2[n], p3 = ptrs.p3[n],
             gamma = ptrs.E[n];
      if (p1 != p1 || p2 != p2 || p3 != p3) {
        printf(
            "NaN detected in push! p1 is %f, p2 is %f, p3 is %f, gamma "
            "is %f\n",
            p1, p2, p3, gamma);
        asm("trap;");
      }

      if (!check_flag(flag, PtcFlag::ignore_EM)) {
        pusher(p1, p2, p3, gamma, E1, E2, E3, B1, B2, B3, qdt_over_2m,
               (Scalar)dt);
      }

      // if (dev_params.rad_cooling_on && sp != (int)ParticleType::ion) {
      //   sync_kill_perp(p1, p2, p3, gamma, B1, B2, B3, E1, E2, E3,
      //                  q_over_m);
      // }
      ptrs.p1[n] = p1;
      ptrs.p2[n] = p2;
      ptrs.p3[n] = p3;
      ptrs.E[n] = gamma;
    }
  };

  if (num > 0) {
    kernel_launch(pusher_kernel, this->ptc->dev_ptrs(), this->Etmp.get_ptrs(),
                  this->Btmp.get_ptrs(), pusher);
  }
}

template class ptc_updater_cu<Config<1>>;
template class ptc_updater_cu<Config<2>>;
template class ptc_updater_cu<Config<3>>;

}  // namespace Aperture
