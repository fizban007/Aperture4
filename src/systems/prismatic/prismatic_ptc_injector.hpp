#pragma once

#include "core/enum_types.h"
#include "core/math.hpp"
#include "core/random.h"
#include "core/typedefs_and_constants.h"
#include "data/rng_states.h"
#include "framework/environment.h"
#include "systems/prismatic/prismatic_exec_policy.hpp"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_ptc_mesh_local.h"
#include "utils/util_functions.h"
#include "utils/vec.hpp"

#ifdef GPU_ENABLED
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#endif

namespace Aperture {

// Default position validator: accept every candidate placement (same role
// as the base injector's always_valid_pos, defined separately so this
// header does not pull in the structured-grid injector stack).
struct prism_always_valid {
  template <typename Cell, typename X>
  HD_INLINE bool operator()(const Cell&, const X&) const {
    return true;
  }
};

// Cartesian position of the point (l1, l2, zeta) inside prism (tri, k).
// Same convention as the particle push and fill_volume: radially projected
// barycentric combination of the sphere unit vectors (l1 -> tri vertex 0,
// l3 = 1 - l1 - l2 -> vertex 2), linear radius interpolation in the shell.
template <typename MeshPtrs>
HD_INLINE vec_t<Scalar, 3> prism_position(const MeshPtrs& mp, int tri, int k,
                                          Scalar l1, Scalar l2, Scalar zeta) {
  int v0 = mp.tri_verts[tri * 3 + 0];
  int v1 = mp.tri_verts[tri * 3 + 1];
  int v2 = mp.tri_verts[tri * 3 + 2];
  Scalar l3 = Scalar(1) - l1 - l2;
  Scalar sx = l1 * mp.sphere_vx[v0] + l2 * mp.sphere_vx[v1] + l3 * mp.sphere_vx[v2];
  Scalar sy = l1 * mp.sphere_vy[v0] + l2 * mp.sphere_vy[v1] + l3 * mp.sphere_vy[v2];
  Scalar sz = l1 * mp.sphere_vz[v0] + l2 * mp.sphere_vz[v1] + l3 * mp.sphere_vz[v2];
  Scalar sn = Scalar(1) / math::sqrt(sx * sx + sy * sy + sz * sz);
  Scalar r = mp.radii[k] + zeta * (mp.radii[k + 1] - mp.radii[k]);
  return vec_t<Scalar, 3>(r * sn * sx, r * sn * sy, r * sn * sz);
}

// Particle injector for the prismatic mesh.  A faithful port of the base
// ptc_injector (systems/ptc_injector_new.h) onto the unstructured
// triangle-times-shell cell layout: identical count -> exclusive-scan ->
// fill-at-offsets structure, identical functor-driven interface, shared
// rng_states data and particle-flag machinery.  Differences:
//   - cells are prisms indexed cell = k * N_tri + tri (the particle cell
//     encoding), iterated as a flat range; Phase 7C: cells and the mesh
//     ptrs are LOCAL (prismatic_ptc_mesh_local — identity single-rank),
//     and the functors must gate on mp.owns_cell(tri, k) themselves when
//     the rank must not inject into its ghost ring;
//   - f_criteria / f_num receive (tri, k, mp) instead of (pos, grid, ext);
//   - in-cell sampling is uniform in (barycentric, zeta) via the triangle
//     fold (NOT uniform in physical volume -- the radial r^2 weighting and
//     spherical distortion are the f_num / f_weight functors' business,
//     same division of labor as the base injector's f_weight(x_global)).
//
// f_dist(x_global, state, PtcType) -> vec_t<value_t, 3> Cartesian momentum
// and f_weight(x_global, PtcType) -> weight match the base signatures, so
// samplers from core/random.h (maxwell_juttner etc.) work unchanged.
template <typename ExecPolicy>
class prismatic_ptc_injector {
 public:
  using exec_tag = typename ExecPolicy::exec_tag;
  using value_t = Scalar;

  // Environment-coupled constructor (mirrors the base injector): fetches
  // "particles" and "rng_states" registered by prismatic_ptc_updater.
  explicit prismatic_ptc_injector(const prismatic_ptc_mesh_local& lmesh)
      : m_lmesh(lmesh),
        m_num_per_cell(lmesh.max_cell(), ExecPolicy::data_mem_type()),
        m_cum_num_per_cell(lmesh.max_cell(), ExecPolicy::data_mem_type()) {
    nonown_ptr<prismatic_particle_data> ptc;
    nonown_ptr<rng_states_t<exec_tag>> states;
    sim_env().get_data("particles", ptc);
    sim_env().get_data("rng_states", states);
    m_ptc = &(*ptc);
    m_states = &(*states);
    sim_env().params().get_value("tracked_fraction", m_tracked_fraction);
    m_track_rank = static_cast<uint64_t>(sim_env().get_rank()) << 32;
  }

  // Direct-dependency constructor (tests, standalone drivers).
  prismatic_ptc_injector(const prismatic_ptc_mesh_local& lmesh,
                         prismatic_particle_data& ptc,
                         rng_states_t<exec_tag>& states)
      : m_lmesh(lmesh),
        m_num_per_cell(lmesh.max_cell(), ExecPolicy::data_mem_type()),
        m_cum_num_per_cell(lmesh.max_cell(), ExecPolicy::data_mem_type()),
        m_ptc(&ptc),
        m_states(&states) {}

  template <typename FCriteria, typename FNumPerCell, typename FDist,
            typename FWeight, typename FValidate = prism_always_valid>
  void inject_plasma(const FCriteria& f_criteria, const FNumPerCell& f_num,
                     const FDist& f_dist, const FWeight& f_weight,
                     uint32_t flag = 0,
                     const FValidate& f_validate = FValidate{},
                     PtcType pos_type = PtcType::positron) {
    auto mp = m_lmesh.get_ptrs(exec_tag{});
    int N_tri = mp.N_tri;
    int N_cells = N_tri * mp.N_r;

    // Count particles per prism (writes every cell; no pre-zero needed).
    ExecPolicy::launch(
        [N_cells, N_tri, mp] LAMBDA(auto num_per_cell, auto f_criteria,
                                    auto f_num) {
          ExecPolicy::loop(0, N_cells, [&] LAMBDA(int c) {
            int tri, k;
            prism_cell_decode(static_cast<uint32_t>(c), N_tri, tri, k);
            num_per_cell[c] =
                f_criteria(tri, k, mp) ? f_num(tri, k, mp) : 0;
          });
        },
        m_num_per_cell, f_criteria, f_num);
    ExecPolicy::sync();

    // Exclusive scan + total, on the side matching the exec policy.
    int new_particles = exclusive_scan(N_cells);

    auto num = m_ptc->number();
    auto max_num = m_ptc->size();
    Logger::print_debug("prismatic injector: current num {}, injecting {}",
                        num, new_particles);

    auto tracked_fraction = m_tracked_fraction;
    auto track_rank = m_track_rank;
    ExecPolicy::launch(
        [N_cells, N_tri, mp, num, max_num, tracked_fraction, track_rank, flag,
         pos_type] LAMBDA(auto ptc, auto states, auto num_per_cell,
                          auto cum_num_per_cell, auto f_dist, auto f_weight,
                          auto f_validate, auto ptc_id) {
          rng_t<exec_tag> rng(states);
          ExecPolicy::loop(0, N_cells, [&] LAMBDA(int c) {
            int tri, k;
            prism_cell_decode(static_cast<uint32_t>(c), N_tri, tri, k);
            for (int n = 0; n < num_per_cell[c]; n += 2) {
              uint32_t offset_e = num + cum_num_per_cell[c] + n;
              uint32_t offset_p = offset_e + 1;
              if (offset_e >= max_num || offset_p >= max_num) break;

              // Uniform sample over the triangle (fold) and the layer.
              value_t u = rng.template uniform<value_t>();
              value_t v = rng.template uniform<value_t>();
              if (u + v > value_t(1)) {
                u = value_t(1) - u;
                v = value_t(1) - v;
              }
              value_t zeta = rng.template uniform<value_t>();
              auto x_global = prism_position(mp, tri, k, u, v, zeta);

              // Same inert-slot convention as the base injector: offsets
              // are fixed by the prefix sum, so an invalid placement is
              // marked empty_cell and reclaimed at the next sort.
              if (!f_validate(c, x_global)) {
                ptc.cell[offset_e] = empty_cell;
                ptc.cell[offset_p] = empty_cell;
                continue;
              }

              ptc.cell[offset_e] = static_cast<uint32_t>(c);
              ptc.cell[offset_p] = static_cast<uint32_t>(c);
              ptc.x1[offset_e] = ptc.x1[offset_p] = u;
              ptc.x2[offset_e] = ptc.x2[offset_p] = v;
              ptc.x3[offset_e] = ptc.x3[offset_p] = zeta;

              auto p = f_dist(x_global, rng.m_local_state, PtcType::electron);
              ptc.p1[offset_e] = p[0];
              ptc.p2[offset_e] = p[1];
              ptc.p3[offset_e] = p[2];
              ptc.E[offset_e] = math::sqrt(value_t(1) + p.dot(p));

              p = f_dist(x_global, rng.m_local_state, pos_type);
              ptc.p1[offset_p] = p[0];
              ptc.p2[offset_p] = p[1];
              ptc.p3[offset_p] = p[2];
              ptc.E[offset_p] = math::sqrt(value_t(1) + p.dot(p));

              ptc.weight[offset_e] = f_weight(x_global, PtcType::electron);
              ptc.weight[offset_p] = f_weight(x_global, pos_type);
              uint32_t local_flag = flag;
              if (!check_flag(local_flag, PtcFlag::ignore_tracking) &&
                  rng.template uniform<value_t>() < tracked_fraction) {
                set_flag(local_flag, PtcFlag::tracked);
              }
              ptc.flag[offset_e] =
                  set_ptc_type_flag(local_flag, PtcType::electron);
              ptc.flag[offset_p] = set_ptc_type_flag(local_flag, pos_type);
              ptc.id[offset_e] = track_rank + atomic_add(ptc_id, 1);
              ptc.id[offset_p] = track_rank + atomic_add(ptc_id, 1);
            }
          });
        },
        *m_ptc, *m_states, m_num_per_cell, m_cum_num_per_cell, f_dist,
        f_weight, f_validate, m_ptc->ptc_id());
    ExecPolicy::sync();
    m_ptc->add_num(new_particles);
  }

  template <typename FCriteria, typename FNumPerCell, typename FDist,
            typename FWeight, typename FValidate = prism_always_valid>
  void inject_pairs(const FCriteria& f_criteria, const FNumPerCell& f_num,
                    const FDist& f_dist, const FWeight& f_weight,
                    uint32_t flag = 0,
                    const FValidate& f_validate = FValidate{}) {
    inject_plasma(f_criteria, f_num, f_dist, f_weight, flag, f_validate,
                  PtcType::positron);
  }

  template <typename FCriteria, typename FNumPerCell, typename FDist,
            typename FWeight, typename FValidate = prism_always_valid>
  void inject_e_i_plasma(const FCriteria& f_criteria, const FNumPerCell& f_num,
                         const FDist& f_dist, const FWeight& f_weight,
                         uint32_t flag = 0,
                         const FValidate& f_validate = FValidate{}) {
    inject_plasma(f_criteria, f_num, f_dist, f_weight, flag, f_validate,
                  PtcType::ion);
  }

 private:
  // Exclusive scan of m_num_per_cell into m_cum_num_per_cell; returns the
  // total.  Device path scans in place with thrust and copies back only
  // the two tail elements; host path is a running sum.
  int exclusive_scan(int N_cells) {
#ifdef GPU_ENABLED
    if constexpr (std::is_same_v<exec_tag, exec_tags::device>) {
      thrust::device_ptr<int> p_num(m_num_per_cell.dev_ptr());
      thrust::device_ptr<int> p_cum(m_cum_num_per_cell.dev_ptr());
      thrust::exclusive_scan(p_num, p_num + N_cells, p_cum);
      GpuCheckError();
      int tail_num = 0, tail_cum = 0;
      GpuSafeCall(gpuMemcpy(&tail_num, m_num_per_cell.dev_ptr() + N_cells - 1,
                            sizeof(int), gpuMemcpyDeviceToHost));
      GpuSafeCall(gpuMemcpy(&tail_cum,
                            m_cum_num_per_cell.dev_ptr() + N_cells - 1,
                            sizeof(int), gpuMemcpyDeviceToHost));
      return tail_cum + tail_num;
    }
#endif
    int* num = m_num_per_cell.host_ptr();
    int* cum = m_cum_num_per_cell.host_ptr();
    int total = 0;
    for (int c = 0; c < N_cells; c++) {
      cum[c] = total;
      total += num[c];
    }
    return total;
  }

  const prismatic_ptc_mesh_local& m_lmesh;
  buffer<int> m_num_per_cell, m_cum_num_per_cell;
  prismatic_particle_data* m_ptc = nullptr;
  rng_states_t<exec_tag>* m_states = nullptr;
  float m_tracked_fraction = 0.0f;
  uint64_t m_track_rank = 0;
};

using prismatic_ptc_injector_t =
    prismatic_ptc_injector<prismatic_exec_policy_dynamic>;

}  // namespace Aperture
