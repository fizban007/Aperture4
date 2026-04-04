#pragma once

#include "core/typedefs_and_constants.h"
#include "systems/prismatic/prismatic_mesh.h"

namespace Aperture {

// Deposit current from a particle trajectory within a single prism.
//
// Uses the closed-form Whitney 1-form path integrals (see
// docs/icosahedral_prismatic_pic/flat_space_pic.tex, §4.4):
//
//   Horizontal: J_{h,ij,k} = (q/dt) * A_ij * phi_bar_k
//     where A_ij = l_old[i]*dl[j] - l_old[j]*dl[i]  (angular factor)
//           phi_bar_k = average of hat function along trajectory
//
//   Vertical: J_{v,i} = (q/dt) * dz * (l_old[i] + l_new[i]) / 2
//
// The deposited current satisfies exact charge conservation:
//   Delta(rho_v) / dt = sum_e (d0)_{e,v} J_e   at every vertex v.
//
// Parameters:
//   mesh       - the prismatic mesh
//   tri_idx    - sphere triangle index (0 to N_tri-1)
//   layer_idx  - radial layer index (0 to N_r-1)
//   l_old[3]   - barycentric coordinates at start (l1, l2, l3)
//   zeta_old   - normalized radial coordinate at start, in [0,1]
//   l_new[3]   - barycentric coordinates at end
//   zeta_new   - normalized radial coordinate at end
//   q_over_dt  - particle charge / timestep
//   J          - current buffer to accumulate into (size N_edges)
void deposit_current_single_prism(
    const prismatic_mesh& mesh,
    int tri_idx, int layer_idx,
    const Scalar l_old[3], Scalar zeta_old,
    const Scalar l_new[3], Scalar zeta_new,
    Scalar q_over_dt,
    Scalar* J);

// Detect if the new coordinates leave the current prism.
// Returns the type of crossing:
//   0 = no crossing (still inside)
//   1 = radial crossing (through triangular face, bottom or top)
//   2 = angular crossing (through rectangular face, one of 3 edges)
//
// On return:
//   crossing_s  - parameter s in [0,1] at which boundary is hit
//   cross_idx   - for radial: -1 (bottom) or +1 (top)
//                 for angular: local edge index 0, 1, or 2
int detect_crossing(const Scalar l_old[3], Scalar zeta_old,
                    const Scalar l_new[3], Scalar zeta_new,
                    Scalar& crossing_s, int& cross_idx);

// Deposit current for a particle that may cross prism boundaries.
// Splits the trajectory at each crossing point and deposits independently
// in each prism. Handles up to max_crossings prism transitions.
//
// Parameters:
//   mesh        - the prismatic mesh
//   tri_idx     - starting sphere triangle index
//   layer_idx   - starting radial layer index
//   l_old[3]    - barycentric coordinates at old position
//   zeta_old    - normalized radial coordinate at old position
//   l_new[3]    - barycentric coordinates at new position (in starting prism)
//   zeta_new    - normalized radial coordinate at new position
//   q_over_dt   - particle charge / timestep
//   J           - current buffer to accumulate into
//   new_tri     - [output] triangle index after all crossings
//   new_layer   - [output] layer index after all crossings
void deposit_current(
    const prismatic_mesh& mesh,
    int tri_idx, int layer_idx,
    const Scalar l_old[3], Scalar zeta_old,
    const Scalar l_new[3], Scalar zeta_new,
    Scalar q_over_dt,
    Scalar* J,
    int& new_tri, int& new_layer);

// Interpolate E and B fields at a particle position using Whitney forms.
//
// E(x) = sum_e E_e * W^1_e(x)   (Whitney 1-form expansion)
// B(x) = sum_f B_f * W^2_f(x)   (Whitney 2-form expansion)
//
// Returns the 3D Cartesian field components at the given position.
void interpolate_fields(
    const prismatic_mesh& mesh,
    int tri_idx, int layer_idx,
    const Scalar l[3], Scalar zeta,
    const Scalar* E_e, const Scalar* B_f,
    Scalar& Ex, Scalar& Ey, Scalar& Ez,
    Scalar& Bx, Scalar& By, Scalar& Bz);

}  // namespace Aperture
