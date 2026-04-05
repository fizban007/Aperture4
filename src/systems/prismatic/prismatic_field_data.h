#pragma once

#include "core/buffer.hpp"
#include "core/data_adapter.h"
#include "core/typedefs_and_constants.h"
#include "framework/data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_particles.h"

namespace Aperture {

// Field data on the prismatic mesh, templated on element type.
// Each instance wraps a single buffer<Scalar> sized to the number of
// elements of that type (vertices, edges, or faces).
//
// Registered as data_t with the environment so multiple systems can
// share the same field object by name.

enum class PrismaticFieldType { vertex, edge, face };

template <PrismaticFieldType Type>
class prismatic_field : public data_t {
 public:
  prismatic_field(const prismatic_mesh& mesh,
                  MemType mem = default_mem_type)
      : m_data(field_size(mesh), mem) {
    // Zero on construction so fields start clean
    m_data.assign(Scalar(0));
  }

  // No-op: systems write initial values (e.g. dipole B) in system init,
  // which runs before data init. We must not overwrite those values.
  void init() override {}

  buffer<Scalar>& data() { return m_data; }
  const buffer<Scalar>& data() const { return m_data; }

  Scalar* host_ptr() { return m_data.host_ptr(); }
  const Scalar* host_ptr() const { return m_data.host_ptr(); }

 private:
  static int field_size(const prismatic_mesh& mesh) {
    if constexpr (Type == PrismaticFieldType::vertex) return mesh.m_N_verts;
    else if constexpr (Type == PrismaticFieldType::edge) return mesh.m_N_edges;
    else return mesh.m_N_faces;
  }

  buffer<Scalar> m_data;
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
