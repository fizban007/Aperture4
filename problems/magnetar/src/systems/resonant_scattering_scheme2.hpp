/*
 * Copyright (c) 2024 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "core/particle_structs.h"
#include "core/random.h"
#include "data/fields.h"
#include "data/phase_space.hpp"
#include "framework/environment.h"
#include "systems/grid.h"
#include "utils/interpolation.hpp"
#include "utils/util_functions.h"
// #include <cstdlib>

template <typename T>
HD_INLINE int
sgn(T val) {
  // if (val == 0) return 1;// To hopefully stop division by zero errors
  return (T(0) <= val) - (val < T(0));
}

namespace Aperture {
template <typename T>
static HD_INLINE T nonan(T val,T a, T epsilon = 1e-10) {// regularized_denominator
      // Ensures value stays at least epsilon away from 'a' to prevent numerical instabilities
    if (math::abs(val-a) < epsilon) {
        return a+ ((val>a)?epsilon:-epsilon);
    }
    return val;
}

template <typename Conf>
struct resonant_scattering_scheme{
  using value_t = typename Conf::value_t;
  HD_INLINE void check_nan(const char* name, value_t val) {
      //  unsigned int bits = *(unsigned int*)&val;
    unsigned int bits = *(unsigned int*)&val;
    
    
    if (isnan(val)) {
        printf("NaN detected in %s: %f (isnan=%d, isinf=%d)\n", 
               name, val, (int)isnan(val), (int)isinf(val));
        asm("trap;");
    }
    if (isinf(val)) {
        printf("Inf detected in %s: %f (isnan=%d, isinf=%d)\n", 
               name, val, (int)isnan(val), (int)isinf(val));
        asm("trap;");
    }
        
    }
    static HD_INLINE value_t max_a(value_t val, value_t a) {
        // Ensures value does not exceed magnitude 'a'
        if (val > a) return a;
        if (val < -a) return -a;
        return val;
    }
    

  const grid_t<Conf> &m_grid;
  value_t BQ = 1.0e7;
  value_t star_kT = 1.0e-3;
  value_t gamma_thr = 10.0;
  value_t gamma_pair = 3.0;
  value_t chi_factor = 100.0;
  value_t res_drag_coef = 4.72e13; // This is the default value from Beloborodov
                                   // 2013, normalized to time unit Rstar/c
                                   // I/e alpha *c/4/lambda_bar*(Rstar/c), alpha = 1/137
  value_t ph_path = 0.0f;
  // Variables describing the reflected photons
  // value_t reflect_fraction = 0.0f; // Fraction of interacting photons that are reflected vs simply thermal from the star
  // value_t reflect_R = 12.0f; // The emitting point of "reflected" photons 

  int downsample = 8;
  // int num_bins[Conf::dim];
  // value_t lower[Conf::dim];
  // value_t upper[Conf::dim];
  vec_t<int, 2> num_bins; // The two dimensions are energy and theta, in that order
  vec_t<value_t, 2> lower;
  vec_t<value_t, 2> upper;
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_B;
  ndptr<value_t, Conf::dim> m_ph_poisson_diag;
  ndptr<value_t, Conf::dim> m_ph_poisson_diag_eq1;
  ndptr<value_t, Conf::dim + 2> m_ph_flux;
  extent_t<Conf::dim + 2> m_ext_flux;

  resonant_scattering_scheme(const grid_t<Conf> &grid) : m_grid(grid) {}

  void init() {
    sim_env().params().get_value("B_Q", BQ);
    sim_env().params().get_value("star_kT", star_kT);
    sim_env().params().get_value("res_drag_coef", res_drag_coef);
    sim_env().params().get_value("gamma_thr", gamma_thr);
    sim_env().params().get_value("gamma_pair", gamma_pair);
    sim_env().params().get_value("chi_factor", chi_factor);
    // sim_env().params().get_value("reflect_fraction", reflect_fraction);
    // sim_env().params().get_value("reflect_R", reflect_R);

    nonown_ptr<vector_field<Conf>> B;
    // This is total B field, i.e. B0 + Bdelta
    sim_env().get_data("B", B);
#ifdef GPU_ENABLED
    m_B[0] = B->at(0).dev_ndptr_const();
    m_B[1] = B->at(1).dev_ndptr_const();
    m_B[2] = B->at(2).dev_ndptr_const();
#else
    m_B[0] = B->at(0).host_ndptr_const();
    m_B[1] = B->at(1).host_ndptr_const();
    m_B[2] = B->at(2).host_ndptr_const();
#endif

    sim_env().params().get_value("ph_flux_downsample", downsample);
    sim_env().params().get_vec_t("ph_flux_bins", num_bins);
    sim_env().params().get_vec_t("ph_flux_lower", lower);
    sim_env().params().get_vec_t("ph_flux_upper", upper);

    nonown_ptr<phase_space<Conf, 2>> ph_flux; // 2D for theta and energy
    ph_flux = sim_env().register_data<phase_space<Conf, 2>>(
        std::string("resonant_ph_flux"), m_grid, downsample, num_bins.data(),
        lower.data(), upper.data(), false, default_mem_type);
    ph_flux->reset_after_output(true);

#ifdef GPU_ENABLED
    m_ph_flux = ph_flux->data.dev_ndptr();
#else
    m_ph_flux = ph_flux->data.host_ndptr();
#endif
    m_ext_flux = ph_flux->data.extent();

    nonown_ptr<scalar_field<Conf>> ph_poisson;
    
    nonown_ptr<scalar_field<Conf>> ph_poisson_eq1;
    ph_poisson = sim_env().register_data<scalar_field<Conf>>(
      std::string("ph_poisson_diag"), m_grid, default_mem_type
    );
    ph_poisson_eq1 = sim_env().register_data<scalar_field<Conf>>(
      std::string("ph_poisson_diag_eq1"), m_grid, default_mem_type
    );
    ph_poisson->reset_after_output(true);
    ph_poisson_eq1->reset_after_output(true);

#ifdef GPU_ENABLED
    m_ph_poisson_diag = ph_poisson->dev_ndptr();
#else
    m_ph_poisson_diag = ph_poisson->host_ndptr();
#endif
#ifdef GPU_ENABLED
    m_ph_poisson_diag_eq1 = ph_poisson->dev_ndptr();
#else
    m_ph_poisson_diag_eq1 = ph_poisson->host_ndptr();
#endif
  }

  value_t absorption_rate(value_t b, value_t eph, value_t sinth) {
    return 0.0f;
  }

  HOST_DEVICE size_t emit_photon(const Grid<Conf::dim, value_t> &grid,
                                 const extent_t<Conf::dim> &ext, ptc_ptrs &ptc,
                                 size_t tid, ph_ptrs &ph, size_t ph_num,
                                 unsigned long long int *ph_pos,
                                 rand_state &state, value_t dt) {
    auto flag = ptc.flag[tid];
    if (check_flag(flag, PtcFlag::ignore_radiation)) {
      return 0;
    }

    // get particle information
    auto cell = ptc.cell[tid];
    auto idx = Conf::idx(cell, ext);
    auto pos = get_pos(idx, ext);
    
    value_t gamma = ptc.E[tid];
    value_t p1 = ptc.p1[tid];
    value_t p2 = ptc.p2[tid];
    value_t p3 = ptc.p3[tid];
    check_nan("p1", p1);
    check_nan("p2", p2);
    check_nan("p3", p3);
    check_nan("gamma", gamma);
  
    vec_t<value_t, 3> rel_x(ptc.x1[tid], ptc.x2[tid], ptc.x3[tid]);
    // x_global gives the global coordinate of the particle
    auto x_global = grid.coord_global(pos, rel_x);
    value_t r = grid_sph_t<Conf>::radius(x_global[0]);
    value_t th = grid_sph_t<Conf>::theta(x_global[1]);

    // Get local B field
    vec_t<value_t, 3> B;
    auto interp = interp_t<1, Conf::dim>{};
    B[0] = interp(rel_x, m_B[0], idx, ext, stagger_t(0b001));
    B[1] = interp(rel_x, m_B[1], idx, ext, stagger_t(0b010));
    B[2] = interp(rel_x, m_B[2], idx, ext, stagger_t(0b100));
    value_t B_mag = math::sqrt(B.dot(B));
    value_t b = B_mag / BQ;
    value_t p = math::sqrt(p1 * p1 + p2 * p2 + p3 * p3);
    value_t pdot_Bhat = (p1 * B[0] + p2 * B[1] + p3 * B[2])/B_mag;
    // The component of momentum parallel to the magnetic field as a vector
    value_t p_para1 = pdot_Bhat *B[0]/ (B_mag);
    value_t p_para2 = pdot_Bhat *B[1]/ (B_mag);
    value_t p_para3 = pdot_Bhat *B[2]/ (B_mag);
    
    value_t p_para_mag = math::sqrt(p_para1*p_para1 + p_para2*p_para2 + p_para3*p_para3);
    // We need to account for a motionless particle
    // In this case the lorentz boost is undefined
    // and mu is meaningless
    value_t no_p = false;
    if (p_para_mag == 0.0f) {
      no_p = true;
    }
    value_t force_dir1 = p_para1/p_para_mag;
    value_t force_dir2 = p_para2/p_para_mag;
    value_t force_dir3 = p_para3/p_para_mag;
    if (no_p) {
      // If the particle is motionless, we use the B field direction as the force direction
      // But ensure the force direction is always towards the equatorial plane
      // i.e in the southern hemisphere we want positive force to have neg theta
      force_dir1 = B[0]/B_mag * sgn<value_t>(B[0]);
      force_dir2 = B[1]/B_mag * sgn<value_t>(B[0]);
      force_dir3 = B[2]/B_mag * sgn<value_t>(B[0]);
    }
    value_t beta_para = p_para_mag / gamma;// This is strictly positive
    // if (beta_para > 0.9999999) {
    //   printf("beta_para is %f", beta_para);
    //   beta_para = 0.9999999;
    // }
    // value_t gamma_para;
    // if (beta_para > 0.999999) {
    //   value_t delta_beta = 1.0 - beta_para;
    //   // printf("delta_beta is %f", delta_beta);
    //   check_nan("delta_beta close 0", 1/delta_beta);
    //   // We approximate The lorentz factor as 1/(sqrt(2delta_beta))
    //   gamma_para = 1.0 / math::sqrt(2.0 * delta_beta);
    //   // value_t beta_para = 1.0 - 1.0 / (gamma_para * gamma_para);
    // }
    // else {
      // beta_para = max_a(beta_para, 0.9999999);// gamma ~2,236 
      // This is basically the gamma associated with the frame of the center of gyration
      // And so is the gamma for lorenz boosting
    value_t gamma_para = 1.0 / math::sqrt(1.0 - beta_para * beta_para);
    // }
    // check_nan("B_mag", B_mag);
    // check_nan("b", b);
    // check_nan("p", p);
    // check_nan("pdotB", pdotB);
    

    // check_nan("p_para", p_para);
    // check_nan("p_para_signed", p_para_signed);
    check_nan("beta_para", beta_para);
    check_nan("gamma_para", gamma_para);
    
    // Compute resonant cooling and emit photon if necessary

    // mu for photons emitted from the surface of the star
    // B_mag = nonan(B_mag, 0.0f);
    // value_t mu = sgn(pdotB)*B[0] / (B_mag); // mu is already absolute value
    // p_para_mag = nonan(p_para_mag, 0.0f);
    // if (p_para_mag != 0.0) {
    value_t mu = force_dir1;// default mu is Fr for motionless particles, this should only change lorentz boost
    if (not no_p) {
      mu = p_para1/p_para_mag;
    }
    // value_t mu = B[0]/B_mag*sgn(B[0]);
    // else {value_t mu = B[0]/B_mag} // mu is already absolute value
    // check_nan("mu", mu);

    // computing mu_R for reflected photons (reflected i.e emitted from r_R on equatorial plane)
    // Could easily be updated for a more general photon emission point
    // value_t A = math::sqrt(r*r + reflect_R*reflect_R - 2*r*reflect_R*math::sin(th));
    // A = nonan(A, 0.0f);
    // // if (A < 1e-3) { // to take care of the region RIGHT by emission where we could divide by zero
    // //   return 0;
    // // }
    // // th might be broken out of the range [0, pi] due to numerical errors
    // if (th > M_PI) th = M_PI;
    // if (th < 0.0f) th = 0.0f;
    // value_t Bx =B[0]*math::sin(th)+B[1]*math::cos(th);//Used in constructing mu_R from /vec(B)dot vec(n) where n is unit vector pointing from reflect_R to r
    // value_t By = B[0]*math::cos(th)-B[1]*math::sin(th);
    // value_t mu_R = (Bx*(r*math::sin(th)-reflect_R)+By*r*math::cos(th))/A/B_mag; // Bdotn/(AB) where n is the vector pointing from  reflect_R to r
    // check_nan("mu_R", mu_R);
    // Swap the mu over to reflected computation
    // mu = mu_R;


    // TODO: check whether this definition of y is correct
    // value_t y = math::abs(b / (star_kT * (gamma_para - p_para_signed * mu)));
    value_t y = math::abs(b / (star_kT * gamma_para * (1.0 - beta_para * mu))); // abs from beta dot photon direction
    // check_nan("y", y);
    // if (y > 30.0f || y <= 0.0f)
    //   return 0; // Way out of resonance, do not do anything
    // printf("in resonance");
      
    // This is based on Beloborodov 2013, Eq. B4. The resonant drag coefficient
    // is the main rescaling parameter, res_drag_coef = alpha * c / 4 / lambda_bar
    value_t exp_term_y = math::exp(y)-1.0f;
    // exp_term_y = nonan(exp_term_y, 0.0f);
    value_t coef = res_drag_coef * square(star_kT) * y * y /
        (r * r * exp_term_y);
    // check_nan("coef", coef);
    value_t Nph = math::abs(coef) * dt / gamma;
    // check_nan("Nph", Nph);
    // This is the general energy of the outgoing photon, in the electron rest frame for reference
    // value_t Eph =
    //     std::min(g - 1.0f, g * (1.0f - 1.0f / math::sqrt(1.0f + 2.0f * B_mag / BQ)));

    // Now we need to compute the outgoing photon energy. It is a fixed energy
    // in the electron rest frame, but needs to be Lorenz transformed to the lab
    // frame. We start with generating a random cos theta from -1 to 1
    float u = 2.0f * rng_uniform<float>(state) - 1.0f;// u= cos(theta) for isotropic emission

    // In electron rest frame Eph = m_e c^2*(1-1/sqrt(1+2b)) emitted isotropically
    value_t Eph = math::abs(gamma_para * (1.0f + beta_para * u) *
                            (1.0f - 1.0f / math::sqrt(1.0f + 2.0f * b))); // lorenz boosted with emission angle dependence (i.e beamed)
    value_t Emax = math::abs(gamma_para * (1.0f + beta_para) * (1.0f - 1.0f / math::sqrt(1.0f + 2.0f * b)));//Emax when u=1
    value_t Eavg = math::abs(gamma_para * (1.0f - 1.0f / math::sqrt(1.0f + 2.0f * b)));//Eavg when u=0
    // check_nan("Eph", Eph);
    // check_nan("Emax", Emax);
    // check_nan("Eavg", Eavg);
    // Photon direction
    float phi_p = 2.0f * M_PI * rng_uniform<float>(state);
    float cphi = math::cos(phi_p);
    float sphi = math::sin(phi_p);
    // Lorentz transform u to the lab frame
    // TODO: Check whether this is correct
    // value_t u_boost_den =(1 + math::abs(beta_para) * u);// effectively the energy boost
    // u_boost_den = nonan(u_boost_den,0.0f);
    u = (u + beta_para) / (1 + beta_para * u); 
    u = max_a(u,0.9999999);

    value_t sth = sqrt(1.0f - u * u);
    // value_t n1 = p1 / p;
    // value_t n2 = p2 / p;
    // value_t n3 = p3 / p;
    // Defining the boost direction
    // TODO: Set up the boost direction for the motionless particle case

    value_t n1 = p_para1 / p_para_mag;
    value_t n2 = p_para2 / p_para_mag;
    value_t n3 = p_para3 / p_para_mag;
    value_t np = math::sqrt(n1 * n1 + n2 * n2);
    if (no_p) { // TODO: check if this is correct
      n1 = force_dir1;
      n2 = force_dir2;
      n3 = force_dir3;
      np = 1.0f;
    }
    // np = nonan(np, 0.0f);
    
    // check_nan("n1", n1);
    // check_nan("n2", n2);
    // check_nan("n3", n3);
    // check_nan("np", np);
  // plane perpendicular to momentum direction defined by
  // {n2/np, -n1/np, 0} and {n3*n1/np, -n3*n2/np, np}
  // We use phi to access a specific direction in this plane
    // value_t n_ph1 = n1 * u + sth * (n2 * cphi + n1 * n3 * sphi) / np;
    // value_t n_ph2 = n2 * u + sth * (-n1 * cphi + n2 * n3 * sphi) / np;
    // value_t n_ph3 = n3 * u + sth * (-np * sphi);
    value_t n_ph1 = n1;
    value_t n_ph2 = n2;
    value_t n_ph3 = n3;
      // check_nan("n_ph1", n_ph1);
      // check_nan("n_ph2", n_ph2);
      // check_nan("n_ph3", n_ph3);
    
    bool produce_photon = false;
    bool deposit_photon = false;
    // Need to take Nph < 1 and > 1 differently, since the photon production
    // may take a significant amount of energy from the emitting particle
    //old if (Eph > 2.0f) {//>2m_ec^2 // Photon energy larger than 1MeV, treat as discrete photon
    // if (Emax > 2.0f) {//>2m_ec^2 // max Photon energy larger than 1MeV, treat as discrete photon
    if (gamma > gamma_thr && y / 100.0f < 30.0f) {
      produce_photon = true;
    } else {
      // Just do drag and deposit the photon into an angle bin

      // Compute analytically the drag force on the particle and apply it. This is taken
      // from Beloborodov 2013, Eq. B6. Need to check the sign. TODO
      // defining the direction of positive drag force
      // value_t B_out1 = B[0]/B_mag *sgn(B[0]);
      // value_t B_out2 = B[1]/B_mag *sgn(B[0]);
      // B_out3 = B[2]/B_mag *sgn(B[0]);
      if (y > 30.0f || y <= 0.0f)
        return 0; // Way out of resonance, do not do anything
      // value_t mu_min_beta =((1-p_para1/gamma_para)*p_para1 // Since photon direction is purely radial
      //                       +(p_para2/gamma_para)* p_para2
      //                       + (p_para3/gamma_para) * p_para3)/p_para_mag;
      // value_t mu_min_beta = force_dir1;
      value_t drag_force = coef * star_kT * y * gamma_para * (mu-beta_para);
      if (no_p) {drag_force = -drag_force;}// TODO: check if this is correct my sign might be weird
      // I think its flipped because force_dir I might have defined as negative for motionless particles
      // printf("mu_min_beta is %f, drag_force is %f\n", mu_min_beta, drag_force);
      p1 += force_dir1 * dt * drag_force;
      p2 += force_dir2 * dt * drag_force;
      p3 += force_dir3 * dt * drag_force;

      ptc.p1[tid] = p1;
      ptc.p2[tid] = p2;
      ptc.p3[tid] = p3;
      ptc.E[tid] = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

      if (Eph < 2.0f) {
        deposit_photon = true;
      }
    }

    if (produce_photon) {
      // value_t ph1 = Eph * (n1 * u + sth * (n2 * cphi + n1 * n3 * sphi) / np);
      // value_t ph2 = Eph * (n2 * u + sth * (-n1 * cphi + n2 * n3 * sphi) / np);
      // value_t ph3 = Eph * (n3 * u - sth * (-np * sphi));
      // printf("beta_para is %f, Emax is %f, gamma_para is %f, Eph is %f\n", beta_para, Emax, gamma_para, Eph);
      Eph = gamma_pair * 2.0;

      ptc.p1[tid] = (p1 -= Eph * n_ph1);
      ptc.p2[tid] = (p2 -= Eph * n_ph2);
      ptc.p3[tid] = (p3 -= Eph * n_ph3);
      ptc.E[tid] = math::sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);

      // Actually produce the photons
      size_t offset = ph_num + atomic_add(ph_pos, 1);
      ph.x1[offset] = ptc.x1[tid];
      ph.x2[offset] = ptc.x2[tid];
      ph.x3[offset] = ptc.x3[tid];
      ph.p1[offset] = Eph * n_ph1;
      ph.p2[offset] = Eph * n_ph2;
      ph.p3[offset] = Eph * n_ph3;
      ph.E[offset] = Eph;
      ph.weight[offset] = ptc.weight[tid];
      ph.path_left[offset] = ph_path;
      ph.cell[offset] = ptc.cell[tid];
      // TODO: Set polarization

      return offset;
    } else if (deposit_photon) {
      // TODO: deposit the outgoing photon into some array
      // This array is ph_flux, which is a 2D array of theta and energy
      // Start off with a simple c=inf case, i.e it gets deposited into the
      // edge of the simulation box immeditely
      value_t cos_ph = n_ph1 * math::cos(th) - n_ph2 * math::sin(th);
      value_t th_ph = math::acos(cos_ph);
      //value_t th_ph = math::acos(n_ph1 * math::sin(th) + n_ph2 * math::cos(th));
      // Figure out the spatial index in the ph_flux array
      index_t<Conf::dim + 2> pos_flux;

      for (int i = 2; i < Conf::dim + 2; i++) {
        // pos_flux[i] = pos[i - 2] / downsample;// didn't include guard cells
        pos_flux[i] = (pos[i - 2] - grid.guard[i - 2]) / downsample;
      }

      // Figure out the theta, Eph index in the ph_flux array
      int n_Eph = std::floor((math::log(Eph) - math::log(lower[0])) /
                             (math::log(upper[0]) - math::log(lower[0])) * num_bins[0]);
      pos_flux[0] = clamp(n_Eph, 0, num_bins[0] - 1);
      pos_flux[1] = clamp(std::floor(th_ph / M_PI * num_bins[1]), 0, num_bins[1] - 1);
      // pos_flux[1] = clamp(std::floor((cos_ph + 1.0) * 0.5 * num_bins[1]), 0, num_bins[1] - 1);
      idx_col_major_t<Conf::dim + 2> idx_flux(pos_flux, m_ext_flux);
      atomic_add(&m_ph_flux[idx_flux], Nph * ptc.weight[tid]);
      return 0;
    }
    return 0;
  }

  HOST_DEVICE size_t produce_pair(const Grid<Conf::dim, value_t> &grid,
                                  const extent_t<Conf::dim> &ext, ph_ptrs &ph,
                                  size_t tid, ptc_ptrs &ptc, size_t ptc_num,
                                  unsigned long long int *ptc_pos,
                                  rand_state &state, value_t dt) {
    // Get the magnetic field vector at the particle location
    auto cell = ph.cell[tid];
    auto idx = Conf::idx(cell, ext);
    auto x = vec_t<value_t, 3>(ph.x1[tid], ph.x2[tid], ph.x3[tid]);
    auto p = vec_t<value_t, 3>(ph.p1[tid], ph.p2[tid], ph.p3[tid]);
    auto pos = get_pos(idx, ext);

    // x_global gives the cartesian coordinate of the photon.
    auto x_global = grid.coord_global(pos, x);

    vec_t<value_t, 3> B;
    auto interp = interp_t<1, Conf::dim>{};
    B[0] = interp(x, m_B[0], idx, ext, stagger_t(0b001));
    B[1] = interp(x, m_B[1], idx, ext, stagger_t(0b010));
    B[2] = interp(x, m_B[2], idx, ext, stagger_t(0b100));

    // Compute the angle between photon and B field and compute the quantum
    // parameter chi value_t chi = quantum_chi(p, B, m_BQ);
    value_t B_mag = math::sqrt(B.dot(B));
    if (B_mag <= 0.0) 
      return 0;
    value_t eph = ph.E[tid];
    auto pxB = cross(p, B);
    auto pdotB = p.dot(B);
    value_t sinth = math::abs(math::sqrt(pxB.dot(pxB)) / B_mag / eph);
    value_t b = B_mag / BQ;

    // There are two requirements for pair production: 1. The photon energy
    // needs to exceed the threshold E_thr = 2me c^2 / sin th; 2. The absorption
    // rate is proportional to 4.3e7 b exp(-8/3bE\sin\theta).
    // TODO: This threshold energy depends on polarization!
    value_t Ethr = 2.0f / sinth;
    if (eph * chi_factor < Ethr)
      return 0;

    value_t chi = 0.5f * eph * b * sinth * chi_factor;
    // TODO: Is there any rescaling that we need to do? Also check units. The
    // actual prefactor doesn't seem to matter much, since the exponential dependence
    // on chi is so strong.
    value_t prob = 4.3e7 * 1e6 * b * math::exp(-4.0 / 3.0 / chi) * dt;
    // value_t prob = 1.0;

    value_t u = rng_uniform<value_t>(state);
    if (u < prob) {
      // Actually produce the electron-positron pair
      size_t offset = ptc_num + atomic_add(ptc_pos, 2);
      size_t offset_e = offset;
      size_t offset_p = offset + 1;

      value_t p_ptc =
          math::sqrt(0.25f - 1.0f / square(eph)) * math::abs(pdotB) / B_mag;
      // printf("sinth is %f, path is %f, eph is %f, prob is %f, chi is %f,
      // p_ptc is %f\n", sinth, ph.path_left[tid], eph, prob,
      //        0.5f * eph * B_mag/m_BQ * sinth * m_zeta, p_ptc);

      // Immediately cool to zero magnetic moment and reduce Lorentz factor as
      // needed
      // value_t gamma = 0.5f * eph;
      value_t gamma = math::sqrt(1.0f + p_ptc * p_ptc);
      // if (sinth > TINY && gamma > 1.0f / sinth) {
      //   gamma = 1.0f / sinth;
      //   if (gamma < 1.01f) gamma = 1.01;
      //   p_ptc = math::sqrt(gamma * gamma - 1.0f);
      // }

      ptc.x1[offset_e] = ptc.x1[offset_p] = x[0];
      ptc.x2[offset_e] = ptc.x2[offset_p] = x[1];
      ptc.x3[offset_e] = ptc.x3[offset_p] = x[2];

      // Particle momentum is along B, direction is inherited from initial
      // photon direction
      value_t sign = sgn(pdotB);
      ptc.p1[offset_e] = ptc.p1[offset_p] = p_ptc * sign * B[0] / B_mag;
      ptc.p2[offset_e] = ptc.p2[offset_p] = p_ptc * sign * B[1] / B_mag;
      ptc.p3[offset_e] = ptc.p3[offset_p] = p_ptc * sign * B[2] / B_mag;
      ptc.E[offset_e] = ptc.E[offset_p] = gamma;
      ptc.aux1[offset_e] = ptc.aux1[offset_p] = 0.0f;

      ptc.weight[offset_e] = ptc.weight[offset_p] = ph.weight[tid];
      ptc.cell[offset_e] = ptc.cell[offset_p] = cell;
      ptc.flag[offset_e] =
          set_ptc_type_flag(flag_or(PtcFlag::secondary), PtcType::electron);
      ptc.flag[offset_p] =
          set_ptc_type_flag(flag_or(PtcFlag::secondary), PtcType::positron);

      return offset;
    }

    return 0;
  }
};

}  // namespace Aperture
