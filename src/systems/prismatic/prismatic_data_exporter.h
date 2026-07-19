#pragma once

#include "core/typedefs_and_constants.h"
#include "framework/system.h"
#include "systems/prismatic/prismatic_field_data.h"
#include "systems/prismatic/prismatic_mesh.h"
#include "systems/prismatic/prismatic_aggregation.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include "systems/prismatic/prismatic_mpi_comm.h"
#include "systems/prismatic/prismatic_owned_runs.h"
#include "utils/hdf_wrapper.h"
#include "utils/nonown_ptr.hpp"
#include <string>
#include <vector>

namespace Aperture {

class prismatic_data_exporter : public system_t {
 public:
  static std::string name() { return "prismatic_data_exporter"; }

  // Phase 5: pass the solver's partition + comm (dec_field_solver::
  // mesh_partition() / the same comm given to the solver) to run
  // distributed — snapshots become single global-indexed files written
  // collectively via parallel HDF5 hyperslabs, bit-identical to the
  // single-rank output.  The solver MUST be registered first so the
  // "E"/"B" data components are local-sized.  mesh.h5 is written by
  // world rank 0 alone (every rank holds the full mesh).
  prismatic_data_exporter(const prismatic_mesh& mesh,
                          const prismatic_mesh_partition* mp = nullptr,
                          const prismatic_mpi_comm* comm = nullptr);
  ~prismatic_data_exporter() = default;

  void register_data_components() override;
  void init() override;
  void update(double dt, uint32_t step) override;

  // Restart support: seed the time accumulator (it integrates += dt).
  void set_time(double t) { m_time = t; }

 private:
  void write_mesh();
  void write_snapshot(uint32_t step, double time);
  void write_meta(H5File& file);

  const prismatic_mesh& m_mesh;
  const prismatic_mesh_partition* m_mp = nullptr;
  const prismatic_mpi_comm* m_comm = nullptr;
  bool m_distributed = false;
  nonown_ptr<prismatic_edge_field> m_E;
  nonown_ptr<prismatic_face_field> m_B;
  // Moment fields (7D: present when a particle updater is registered;
  // the exporter is now the only production output system).
  nonown_ptr<prismatic_edge_field> m_J;
  nonown_ptr<prismatic_vertex_field> m_rho;
  nonown_ptr<prismatic_vertex_field> m_rho_abs;
  nonown_ptr<prismatic_vertex_field> m_gamma_wsum;

  // Distributed mode: contiguous owned runs per combined dataset
  // (E_e = [h|v] edges, B_f = [tri|rect] faces), built once at init.
  // Run i writes field_buf[mem_off[i] .. +len[i]) to dataset position
  // file_off[i] — see H5File::write_parallel_runs.
  prismatic_run_set m_E_runs, m_B_runs, m_V_runs;

  int m_output_interval = 100;

  // Structured downsampling of raw cochain output. The radial stride
  // applies to the shell index k; the angular stride applies to the
  // sphere-element index (sphere triangle for triangular faces, sphere
  // edge for rectangular faces / horizontal edges, sphere vertex for
  // vertical edges). Both default to 1 (full output, identical to
  // pre-existing behavior).
  //
  // Setting (radial=2, angular=4) downsamples by a factor of 8 in a
  // way that mirrors exactly one level of refinement coarsening: every
  // other shell × the corner-child of every parent triangle (the
  // subdivide() routine pushes the (a, m_ab, m_ac) corner child first
  // for every parent, so stride 4 in t selects one specific child per
  // parent — see prismatic_mesh.cpp:117).
  //
  // The kept indices for each element type are stored in mesh.h5 as
  // output_face_idx / output_edge_idx, plus the strides themselves and
  // the unique vertices referenced by the kept set so external tools
  // can render the downsampled mesh standalone.
  int m_output_radial_stride  = 1;
  int m_output_angular_stride = 1;
  // 7D F9: TRUE coarse-cochain aggregation (fld_output_aggregate with
  // fld_output_angular_level = j and the radial stride as R).  Replaces
  // full snapshots with bona fide level-(L−j) DEC dumps; works in both
  // modes (distributed ranks sum partials over owned fine elements and
  // MPI-reduce — no slab-alignment constraint needed at all).
  bool m_aggregate = false;
  int m_agg_level = 0;
  prismatic_coarse_aggregator m_agg;
  std::vector<double> m_agg_E, m_agg_B, m_agg_J;
  std::vector<double> m_agg_rho, m_agg_ra, m_agg_gw;
  void write_aggregated(uint32_t step, double time);
  std::string m_output_dir = "Data";
  double m_time = 0.0;

  // Built once in init(); empty when both strides are 1.
  std::vector<int> m_out_edge_idx;
  std::vector<int> m_out_face_idx;
  // Unique vertex indices referenced by the kept faces and edges,
  // sorted ascending. Lets visualization tools build a self-contained
  // vertex list for the downsampled mesh.
  std::vector<int> m_out_vert_idx;
  // Per-snapshot scratch buffers (avoid reallocation each call).
  std::vector<Scalar> m_out_E_buf;
  std::vector<Scalar> m_out_B_buf;
};

}  // namespace Aperture
