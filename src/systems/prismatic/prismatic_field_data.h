#pragma once

#include "core/buffer.hpp"
#include "core/data_adapter.h"
#include "core/typedefs_and_constants.h"
#include "framework/data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_particles.h"

namespace Aperture {

// Field data on the prismatic mesh, templated on element type.
// Each instance wraps a single buffer<Scalar> sized to the number of
// elements of that type (vertices, edges, or faces).
//
// Registered as data_t with the environment so multiple systems can
// share the same field object by name.

enum class PrismaticFieldType { vertex, edge, face };

// Describes how an edge 1-cochain buffer should be interpreted.  The flat
// solver stores edge values as the primal 1-cochain ∫_e E·dl; the GR
// Kerr-Schild solver stores D̃[e], the dual 2-cochain, which must be
// multiplied by hodge1_inv[e] to recover the primal 1-cochain ∫_e D·dl.
// Downstream consumers (Whitney interpolation in the sph output,
// particle depositor, ...) query this tag to know whether to apply the
// Hodge conversion.
enum class EdgeCochainKind { primal_1, dual_2 };

// Storage layout (4.1b): ONE contiguous buffer per field with the
// primary sub-type first — h_edges then v_edges for edge fields,
// tri_faces then rect_faces for face fields.  This is exactly the
// global mesh ordering, so single-rank consumers that index the
// combined range (particle kernels, sph output, exporter) are
// untouched, while solver kernels can use the split views below.
// Under a multirank partition (local-sized construction, 4.1b.6) the
// same two-block layout holds with local sizes; the combined-range
// consumers then MUST NOT be used (single-rank assert lives in the
// particle updater).
template <PrismaticFieldType Type>
class prismatic_field : public data_t {
 public:
  prismatic_field(const prismatic_mesh& mesh,
                  MemType mem = default_mem_type)
      : m_data(field_size(mesh), mem) {
    if constexpr (Type == PrismaticFieldType::edge) {
      m_split = (mesh.m_N_r + 1) * mesh.m_N_edge_s;   // N_h_edges
    } else if constexpr (Type == PrismaticFieldType::face) {
      m_split = (mesh.m_N_r + 1) * mesh.m_N_tri;      // N_tri_faces
    }
    // Zero on construction so fields start clean
    m_data.assign(Scalar(0));
  }

  // 4.1b.6 part 2: LOCAL-sized construction for a distributed run —
  // the two blocks are the partition's local (owned + ghost) cochain
  // layouts.  Combined-range consumers (particles, sph output, the
  // exporter) MUST NOT be registered alongside such fields.
  prismatic_field(const prismatic_mesh_partition& mp,
                  MemType mem = default_mem_type)
      : m_data(local_field_size(mp), mem) {
    if constexpr (Type == PrismaticFieldType::edge) {
      m_split = mp.layout(cochain_type::h_edge).local_size();
    } else if constexpr (Type == PrismaticFieldType::face) {
      m_split = mp.layout(cochain_type::tri_face).local_size();
    }
    m_data.assign(Scalar(0));
  }

  // No-op: systems write initial values (e.g. dipole B) in system init,
  // which runs before data init. We must not overwrite those values.
  void init() override {}

  buffer<Scalar>& data() { return m_data; }
  const buffer<Scalar>& data() const { return m_data; }

  Scalar* host_ptr() { return m_data.host_ptr(); }
  const Scalar* host_ptr() const { return m_data.host_ptr(); }

  // Split views (4.1b): first block (h_edges / tri_faces) and second
  // block (v_edges / rect_faces).  Offsets are in elements; valid on
  // host and device pointers alike.
  int split() const { return m_split; }
  Scalar* host_ptr_a() { return m_data.host_ptr(); }
  Scalar* host_ptr_b() { return m_data.host_ptr() + m_split; }
  Scalar* dev_ptr_a() { return m_data.dev_ptr(); }
  Scalar* dev_ptr_b() {
    return m_data.dev_ptr() ? m_data.dev_ptr() + m_split : nullptr;
  }

  // For edge fields only: owning system declares the cochain convention.
  EdgeCochainKind edge_kind() const { return m_edge_kind; }
  void set_edge_kind(EdgeCochainKind k) { m_edge_kind = k; }

 private:
  static int field_size(const prismatic_mesh& mesh) {
    // Explicit narrow: the mesh-sized ctor is the SINGLE-RANK path (a
    // global-sized field above 2^31 elements cannot exist on one rank).
    if constexpr (Type == PrismaticFieldType::vertex) {
      return int(mesh.m_N_verts);
    } else if constexpr (Type == PrismaticFieldType::edge) {
      return int(mesh.m_N_edges);
    } else {
      return int(mesh.m_N_faces);
    }
  }
  static int local_field_size(const prismatic_mesh_partition& mp) {
    if constexpr (Type == PrismaticFieldType::vertex) {
      return mp.layout(cochain_type::vertex).local_size();
    } else if constexpr (Type == PrismaticFieldType::edge) {
      return mp.layout(cochain_type::h_edge).local_size() +
             mp.layout(cochain_type::v_edge).local_size();
    } else {
      return mp.layout(cochain_type::tri_face).local_size() +
             mp.layout(cochain_type::rect_face).local_size();
    }
  }

  buffer<Scalar> m_data;
  int m_split = 0;   // element offset of the second block (v / rect)
  EdgeCochainKind m_edge_kind = EdgeCochainKind::primal_1;
};

using prismatic_vertex_field = prismatic_field<PrismaticFieldType::vertex>;
using prismatic_edge_field = prismatic_field<PrismaticFieldType::edge>;
using prismatic_face_field = prismatic_field<PrismaticFieldType::face>;

// Particle data for the prismatic mesh.
class prismatic_particle_data : public data_t, public prismatic_particles_t {
 public:
  prismatic_particle_data(size_t max_ptc, MemType mem = default_mem_type)
      : prismatic_particles_t(max_ptc, mem) {}

  void init() override { prismatic_particles_t::init(); }
};

// Adapter specializations so ExecPolicy::launch can adapt particle data
// to the correct pointer struct for host/device execution.
template <>
struct host_adapter<prismatic_particle_data> {
  typedef prism_ptc_ptrs type;
  typedef prism_ptc_ptrs const_type;
  static inline type apply(prismatic_particle_data& d) {
    return d.get_host_ptrs();
  }
};

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
template <>
struct gpu_adapter<prismatic_particle_data> {
  typedef prism_ptc_ptrs type;
  typedef prism_ptc_ptrs const_type;
  static inline type apply(prismatic_particle_data& d) {
    return d.get_dev_ptrs();
  }
};
#endif

}  // namespace Aperture
