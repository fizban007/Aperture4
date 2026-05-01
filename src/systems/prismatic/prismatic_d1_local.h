#pragma once

#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh_partition.h"
#include <vector>

namespace Aperture {

class prismatic_mesh;

// =========================================================================
// Local d1 / d1^T sparse incidence, partitioned into per-cochain blocks.
//
// The global mesh stores d1 (face → bounding edges, ±1 sign) and d1^T
// (edge → adjacent faces) as flat CSR matrices over combined index
// spaces.  Under partitioning, faces and edges live in 4 separate
// cochain layouts (tri_face, rect_face, h_edge, v_edge), so the
// natural local form is a set of cochain-paired sparse blocks:
//
//   d1_tri_h   : owned tri_face  → local h_edge   (degree 3 per row)
//   d1_rect_h  : owned rect_face → local h_edge   (degree 2 per row)
//   d1_rect_v  : owned rect_face → local v_edge   (degree 2 per row)
//   d1t_h_tri  : owned h_edge    → local tri_face (degree 2 per row)
//   d1t_h_rect : owned h_edge    → local rect_face (degree 2 per row)
//   d1t_v_rect : owned v_edge    → local rect_face (degree = vertex valence)
//
// Two empty cross-blocks are dropped (d1_tri_v, d1t_v_tri).
//
// Row indices are LOCAL into the row-side cochain layout, restricted
// to the owned range — the solver iterates owned cells only.
// Column indices are LOCAL into the column-side cochain layout's full
// range (owned + ghost), so any neighbor a stencil reaches can be
// addressed without further translation, including halo-resident peers.
//
// Sign convention is preserved unchanged from the global d1 / d1^T:
// val[j] is ±1 just like the global mesh's d1_val / d1t_val.  Phase
// 4.1b solver kernels can use these blocks directly in place of the
// flat global CSRs.
// =========================================================================
template <typename T>
struct sparse_csr {
  std::vector<int> row_ptr;   // size n_rows + 1
  std::vector<int> col_idx;   // size nnz
  std::vector<T> val;         // size nnz

  int n_rows() const { return int(row_ptr.size()) - 1; }
  int nnz() const { return int(col_idx.size()); }
};

class prismatic_d1_local {
 public:
  // Build local d1 / d1^T blocks from a globally-constructed mesh and
  // its partition bundle.  The partition's halo plans must already
  // halo every neighbor referenced by d1 / d1^T on owned cells (this
  // is automatic for the standard angular + radial plan builders).
  static prismatic_d1_local build(const prismatic_mesh& mesh,
                                  const prismatic_mesh_partition& mp);

  const prismatic_mesh_partition& partition() const { return *m_partition; }

  // d1 blocks: face row → edge col.
  sparse_csr<Scalar> d1_tri_h;
  sparse_csr<Scalar> d1_rect_h;
  sparse_csr<Scalar> d1_rect_v;

  // d1^T blocks: edge row → face col.
  sparse_csr<Scalar> d1t_h_tri;
  sparse_csr<Scalar> d1t_h_rect;
  sparse_csr<Scalar> d1t_v_rect;

 private:
  const prismatic_mesh_partition* m_partition = nullptr;
};

}  // namespace Aperture
