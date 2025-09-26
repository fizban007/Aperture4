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
#include "systems/policies/coord_policy_spherical_gca.hpp"
// #include <cstdlib>


namespace Aperture {

template <typename T>
HD_INLINE bool isnan(T val) {
    return val != val;
}

template <typename T>
HD_INLINE bool isinf(T val) {
    return (val > std::numeric_limits<T>::max() || val < -std::numeric_limits<T>::max());
}

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
  HD_INLINE void check_nan(const char *name, value_t val) {
    //  unsigned int bits = *(unsigned int*)&val;
    unsigned int bits = *(unsigned int *)&val;

    if (isnan(val)) {
      printf("NaN detected in %s: %f (isnan=%d, isinf=%d)\n", name, val,
             (int)isnan(val), (int)isinf(val));
      // asm("trap;");
    }
    if (isinf(val)) {
      printf("Inf detected in %s: %f (isnan=%d, isinf=%d)\n", name, val,
             (int)isnan(val), (int)isinf(val));
      // asm("trap;");
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
  vec_t<ndptr_const<value_t, Conf::dim>, 3> m_B, m_E;
  ndptr<value_t, Conf::dim> m_ph_poisson_diag;
  ndptr<value_t, Conf::dim> m_ph_poisson_diag_eq1;
  ndptr<value_t, Conf::dim + 2> m_ph_flux;
  extent_t<Conf::dim + 2> m_ext_flux;

  resonant_scattering_scheme(const grid_t<Conf> &grid) : m_grid(grid) {}

  void init() {
    sim_env().params().get_value("B_Q", BQ);
    sim_env().params().get_value("star_kT", star_kT);
    sim_env().params().get_value("res_drag_coef", res_drag_coef);
    // sim_env().params().get_value("reflect_fraction", reflect_fraction);
    // sim_env().params().get_value("reflect_R", reflect_R);

    nonown_ptr<vector_field<Conf>> B;
    nonown_ptr<vector_field<Conf>> E;
    // This is total B field, i.e. B0 + Bdelta
    sim_env().get_data("B", B);
    sim_env().get_data("E", E);
#ifdef GPU_ENABLED
    m_B[0] = B->at(0).dev_ndptr_const();
    m_B[1] = B->at(1).dev_ndptr_const();
    m_B[2] = B->at(2).dev_ndptr_const();
    m_E[0] = E->at(0).dev_ndptr_const();
    m_E[1] = E->at(1).dev_ndptr_const();
    m_E[2] = E->at(2).dev_ndptr_const();
#else
    m_B[0] = B->at(0).host_ndptr_const();
    m_B[1] = B->at(1).host_ndptr_const();
    m_B[2] = B->at(2).host_ndptr_const();
    m_E[0] = E->at(0).host_ndptr_const();
    m_E[1] = E->at(1).host_ndptr_const();
    m_E[2] = E->at(2).host_ndptr_const();
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
    // Note with the GCA scheme p[0] is the magnitude of p_parallel to B and p[1] is the conserved magnetic moment mu (related to p_perp)
    value_t p_para = ptc.p1[tid];
    value_t moment_mu = ptc.p2[tid];
    vec_t<value_t, 3> rel_x(ptc.x1[tid], ptc.x2[tid], ptc.x3[tid]);
    // x_global gives the global coordinate of the particle
    auto x_global = grid.coord_global(pos, rel_x);
    value_t r = grid_sph_t<Conf>::radius(x_global[0]);
    value_t th = grid_sph_t<Conf>::theta(x_global[1]);

    // Get local B field
    vec_t<value_t, 3> B, E;
    auto interp = interp_t<1, Conf::dim>{};
    B[0] = interp(rel_x, m_B[0], idx, ext, stagger_t(0b001));
    B[1] = interp(rel_x, m_B[1], idx, ext, stagger_t(0b010));
    B[2] = interp(rel_x, m_B[2], idx, ext, stagger_t(0b100));
    E[0] = interp(rel_x, m_E[0], idx, ext, stagger_t(0b110));
    E[1] = interp(rel_x, m_E[1], idx, ext, stagger_t(0b101));
    E[2] = interp(rel_x, m_E[2], idx, ext, stagger_t(0b011));
    
    auto vE = coord_policy_spherical_gca<Conf>::f_v_E(E, B); // 3D vector like E, B
    value_t B_mag = math::sqrt(B.dot(B));
    value_t p_perp = math::sqrt(moment_mu * B_mag * 2); // TODO: IDK ABOUT THIS ONE
    value_t b = B_mag / BQ;
    // The component of momentum parallel to the magnetic field as a vector
    // I just did minimal changes to work with new P[] def in GCA there is probably a better way now,
    // but this should work
    value_t p_para1 = p_para *B[0]/ (B_mag);
    value_t p_para2 = p_para *B[1]/ (B_mag);
    value_t p_para3 = p_para *B[2]/ (B_mag);
    
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
    if (beta_para > 0.9999999) {
      printf("beta_para is %f", beta_para);
      beta_para = 0.9999999;
    }
    
    value_t gamma_para = 1.0 / math::sqrt(1.0 - beta_para * beta_para);
    
    //TODO I think gamma and gamma_para should be the same, so maybe can clean up
     gamma_para = gamma; // This is the gamma associated with the frame of the center of gyration so I think should be the same
    // }
    beta_para = math::sqrt(1.0 - 1.0 / (gamma_para * gamma_para));
    
    // Compute resonant cooling and emit photon if necessary

    value_t mu = force_dir1;// default mu is Fr for motionless particles, this should only change lorentz boost
    if (not no_p) {
      mu = p_para1/p_para_mag;
    }
    // TODO: check whether this definition of y is correct especially with GCA
    // value_t y = math::abs(b / (star_kT * (gamma_para - p_para_signed * mu)));
    value_t y = math::abs(b / (star_kT * gamma_para * (1.0 - beta_para * mu))); // abs from beta dot photon direction
    
    if (y > 30.0f || y <= 0.0f)
      return 0; // Way out of resonance, do not do anything
      
    // This is based on Beloborodov 2013, Eq. B4. The resonant drag coefficient
    // is the main rescaling parameter, res_drag_coef = alpha * c / 4 / lambda_bar
    value_t exp_term_y = math::exp(y)-1.0f;
    // exp_term_y = nonan(exp_term_y, 0.0f);
    value_t coef = res_drag_coef * square(star_kT) * y * y /
        (r * r * exp_term_y);
    value_t Nph = math::abs(coef) * dt / gamma;

    // Now we need to compute the outgoing photon energy. This has heavy dependence
    // on Angles, so I am very careful with naming things in appropriate frames
    // We start with generating a random nu' = cos phi' from -1 to 1
    float nu_prime = (2.0f * rng_uniform<float>(state) - 1.0f) * 0.999f; // setting for 0.9999 to avoid numerical issues
    value_t nu = (nu_prime + beta_para) / (1.0f + beta_para * nu_prime); // lorenz boost to lab frame
 
    // Trying to account for Eph more accurately

    //We compute energy of photon in the rest frame when it depends on 1st Landau level, incoming mu' and outgoing u'
    value_t mu_prime = (mu-beta_para)/(1.0 - beta_para * mu); // lorenz transform of mu to rest frame
    if (math::abs(mu_prime) > 0.999f) { // Avoid numerical issues with 0/0
      mu_prime = sgn(mu_prime)*0.999f;
    }
    // These commented out sections are the computation of the energy from total 4-momentum conservation
    // value_t E1 = math::sqrt(1.0f + 2.0f * b); // First landau energy
    // value_t E1 = -mu_prime / (1 - mu_prime*mu_prime) // First landau energy considering momentum
    //               + math::sqrt( mu_prime*mu_prime*mu_prime*mu_prime
    //                             - mu_prime*mu_prime* (1+ 2*b)
    //                             + (1+ 2 * b)) / (1 - mu_prime*mu_prime);// This E1 considers momentum of 1st Landau level not being 0
    // value_t Eph_prime = (E1 - 1.0f) / (1.0f + (E1 - 1.0f) * (1.0f - mu_prime * nu_prime + math::sqrt((1.0f + mu_prime * mu_prime) * (1.0f + nu_prime * nu_prime)))); // Outgoing photon energy in rest frame
    // TODO: CHECK
    // I think that Eph_max occurs when mu_prime = mu_prime. as mu_prime is given for any specific scattering event
    // Okay so this form does not have a simple nu_prime leading to max it HEAVILY depends on b so I approximated nuprime(b) as decaying power law to yield max
    // nu_inf = 0.4325, A = 0.5729, b0 = 1.2242, alpha = 1.5944 from my analytic plotting in python
    // value_t nu_max_prime = 0.4325 + 0.5729 / std::pow(1+b/1.2242,1.5944);
    // value_t Emax_prime = (E1 - 1.0f) / (1.0f + (E1 - 1.0f) * (1.0f - mu_prime*nu_max_prime  + math::sqrt((1.0f + mu_prime * mu_prime) *  (1.0f + nu_max_prime * nu_max_prime)))); // Outgoing photon energy in rest frame
  

    //Kun's derivation from Energy conservation and ONLY parallel momentum conservation, I believe it assumes p_perp is 0
    value_t Eph_incoming = 1 / (1-mu_prime * mu_prime) * (math::sqrt(1 + 2*b *(1-mu_prime * mu_prime)) -1); // incoming photon energy in rest frame for resonant scattering
    value_t sin_phi_primesqr = 1.0f - nu_prime * nu_prime;
    value_t Eph_prime = 1/ sin_phi_primesqr * ( 1 +Eph_incoming * (1 - mu_prime * nu_prime) - math::sqrt(1 + 2 * Eph_incoming * nu_prime * (nu_prime - mu_prime) + Eph_incoming * Eph_incoming * (nu_prime - mu_prime) * (nu_prime - mu_prime))); // outgoing photon energy in rest frame
    // From testing previous line it seems like there is a b/gamma dependence to the vlaue of nu' that maximizes. Roughly if b/gamma > 1 then nu' = -1 if b/gamma < 1 then nu' = 1. This breaks down at low gamma < 10
    value_t nu_max_prime = (b/gamma_para > 1.0f)?-0.999f:0.999f;
    
    value_t Emax_prime = 1/ (1-nu_max_prime * nu_max_prime) * ( 1 +Eph_incoming * (1 - mu_prime * nu_max_prime) - math::sqrt(1 + 2 * Eph_incoming * nu_max_prime * (nu_max_prime - mu_prime) + Eph_incoming * Eph_incoming * (nu_max_prime - mu_prime) * (nu_max_prime - mu_prime))); // outgoing photon energy in rest frame
    // This blows up when mu' -> -1 and nu' -> 1 which is the condition near equator at rmax ~1.1 where I get issues, the true limit via l'hopitals is 0
    // if ( mu_prime < -0.999 && nu_max_prime > 0.999) {
    //   Emax_prime = 0.0f;
    // }

    
    value_t Eph = gamma_para * (1.0f + beta_para * nu_prime) * Eph_prime; // Lorenz boost to lab frame
    value_t Emax = gamma_para * (1.0f + beta_para * nu_max_prime) * Emax_prime; // Lorenz boost to lab frame
    if ((Eph - Emax)/Emax > 1e-3 && gamma > 5.0f) {
      printf("Warning Eph > Emax, Eph = %e, Emax = %e, gamma = %e, b = %e, mu = %e, nu' = %e, nu_max' = %e, Eph_inc = %e, r = %e, th = %e\n", Eph, Emax, gamma_para, b, mu_prime, nu_prime, nu_max_prime,Eph_incoming,r,th);
      asm("trap;");
    }

    // Photon direction
    float phi_p = 2.0f * M_PI * rng_uniform<float>(state);
    float cphi = math::cos(phi_p);
    float sphi = math::sin(phi_p);
    value_t sth = sqrt(1.0f - nu * nu);

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
  
  // In the new GCA scheme we want to change the particle momentum more accurately for singly tracked photons
  // {n2/np, -n1/np, 0} and {n3*n1/np, -n3*n2/np, np}
  // We use phi to access a specific direction in this plane
    value_t n_ph1 = n1 * nu + sth * (n2 * cphi + n1 * n3 * sphi) / np;
    value_t n_ph2 = n2 * nu + sth * (-n1 * cphi + n2 * n3 * sphi) / np;
    value_t n_ph3 = n3 * nu + sth * (-np * sphi);
  // Because we redefined ptc.p1[tid] = p_para, ptc.p2[tid] = mu, we need to
  // compute p1,p2,p3 in the n1,n2,n3 basis
  // then subtract the photon energy from the particle
  // and finally convert back to the new p_para, p_mu definition
  
    
    bool produce_photon = false;
    bool deposit_photon = false;
    // Need to take Nph < 1 and > 1 differently, since the photon production
    // may take a significant amount of energy from the emitting particle
    if (Emax > 2.0f) {//>2m_ec^2 // max Photon energy larger than 1MeV, treat as discrete photon
    //  if (false) {
    // incident photon energy to compute the change in momentum
      value_t Eph_inc = (E1 - 1.0f) * gamma_para * (1.0f + beta_para * mu_prime); // Incoming photon energy in rest frame, assuming head on collision mu' = -1
      
      // value_t del_p_para = -1.0* Eph * nu; // Only Eph
      value_t del_p_para = Eph_inc * mu - Eph * nu; // considering incoming
      
      if (!std::isfinite(del_p_para)) {
        printf("del_p_para is nan, Eph is %e, nu is %e, mu is %e, Eph_inc is %e\n", Eph, nu, mu, Eph_inc);
        printf("gamma is %e, gamma_para is %e, beta_para is %e, b is %e, p_para is %e, p_perp is %e, r is %e, th is %e\n", gamma, gamma_para, beta_para, b, p_para, p_perp, r, th);
        asm("trap;");
      }
      
      // always produce a photon if Nph > 1, otherwise draw from poisson?
      
        // Always produce a photon (we are assuming Nph is very close to 1)
        // Technically real number of photons is drawn from a poisson
        // But this code doesn't know how to deal with multiple photon creations
        int N_pois = rng_poisson<float>(state, Nph);
        if (N_pois >= 1) {
          value_t sgn = 1.0f;
          if (p_para < 0.0f) {sgn = -1.0f;}// ensure drag force always points along B
          p_para += sgn*del_p_para; // change p_para appropriately
          ptc.p1[tid] = p_para; // Remember to not get confused by ptc.p1=p_para 
          // p2 being moment_mu is always set to 0
          value_t new_E = coord_policy_spherical_gca<Conf>::f_Gamma(p_para, moment_mu, B_mag, vE);
          if (!std::isfinite(new_E)) {
            printf("new_E is nan, p_para is %e, moment_mu is %e, B_mag is %e\n", p_para, moment_mu, B_mag);
            printf("gamma is %e, gamma_para is %e, beta_para is %e, b is %e, r is %e, th is %e\n", gamma, gamma_para, beta_para, b, r, th);
            asm("trap;");
          }
          ptc.E[tid] = new_E;
          if (Eph > 2.0f) {
            produce_photon = true;
          } else {
            deposit_photon = true;
          }
          // Nph = 1.0; // We enforce a max of 1 for now
          if (N_pois > 1) {
            atomic_add(&m_ph_poisson_diag[idx], 1);
          }
          else {
            atomic_add(&m_ph_poisson_diag_eq1[idx], 1);
          }
        }
    } else {
      // Just do drag and deposit the photon into an angle bin

      // Compute analytically the drag force on the particle and apply it. This is taken
      // from Beloborodov 2013, Eq. B6. 
      
      value_t drag_force = coef * star_kT * y * gamma_para * (mu-beta_para);
      if (no_p) {drag_force = -drag_force;}// TODO: check if this is correct my sign might be weird
      value_t sgn = 1.0f;
      if (p_para < 0.0f) {sgn = -1.0f;}// Ensure Drag force always points along B
      p_para += sgn * drag_force * dt ; // change p_para appropriately
      
      ptc.p1[tid] = p_para;
      value_t new_E = coord_policy_spherical_gca<Conf>::f_Gamma(p_para, moment_mu, B_mag, vE);
          if (!std::isfinite(new_E)) {
            printf("new_E is nan, p_para is %e, moment_mu is %e, B_mag is %e\n", p_para, moment_mu, B_mag);
            printf("gamma is %e, gamma_para is %e, beta_para is %e, b is %e, r is %e, th is %e\n", gamma, gamma_para, beta_para, b, r, th);
            asm("trap;");
          }
      ptc.E[tid] = new_E;
      deposit_photon = true;
    }

    if (produce_photon) {
      
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
      // This array is ph_flux, which is a 2D array of theta and energy
      // Start off with a simple c=inf case, i.e it gets deposited into the
      // edge of the simulation box immeditely
      value_t cos_ph = n_ph1 * math::cos(th) - n_ph2 * math::sin(th);
      value_t th_ph = math::acos(cos_ph);
      //value_t th_ph = math::acos(n_ph1 * math::sin(th) + n_ph2 * math::cos(th));
      // Figure out the spatial index in the ph_flux ve
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

    vec_t<value_t, 3> B, E;
    auto interp = interp_t<1, Conf::dim>{};
    B[0] = interp(x, m_B[0], idx, ext, stagger_t(0b001));
    B[1] = interp(x, m_B[1], idx, ext, stagger_t(0b010));
    B[2] = interp(x, m_B[2], idx, ext, stagger_t(0b100));
    E[0] = interp(x, m_E[0], idx, ext, stagger_t(0b001));
    E[1] = interp(x, m_E[1], idx, ext, stagger_t(0b010));
    E[2] = interp(x, m_E[2], idx, ext, stagger_t(0b100));
    
    auto vE = coord_policy_spherical_gca<Conf>::f_v_E(E, B); // 3D vector like E, B

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
    if (eph < Ethr)
      return 0;

    value_t chi = 0.5f * eph * b * sinth;
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
      // value_t gamma = math::sqrt(1.0f + p_ptc * p_ptc);
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
      // ptc.p1[offset_e] = ptc.p1[offset_p] = p_ptc * sign * B[0] / B_mag;
      // ptc.p2[offset_e] = ptc.p2[offset_p] = p_ptc * sign * B[1] / B_mag;
      // ptc.p3[offset_e] = ptc.p3[offset_p] = p_ptc * sign * B[2] / B_mag;
      ptc.p1[offset_e] = ptc.p1[offset_p] = p_ptc * sign;
      ptc.p2[offset_e] = ptc.p2[offset_p] = 0.0f;
      ptc.p3[offset_e] = ptc.p3[offset_p] = 0.0f;
      ptc.E[offset_e] = ptc.E[offset_p] = coord_policy_spherical_gca<Conf>::f_Gamma(p_ptc, 0, B_mag, vE);
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

}  // namespace Apertutr
