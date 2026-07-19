#pragma once

#include "systems/prismatic/prismatic_cochain_layout.h"
#include "hdf5.h"
#include <vector>

namespace Aperture {

// Contiguous owned runs of a distributed cochain layout, for scattered
// hyperslab I/O against the combined global datasets (see
// H5File::write_parallel_runs / read_parallel_runs).  Run i maps
// buffer[mem_off[i] .. +len[i]) to dataset [file_off[i] .. +len[i]).
// Shared by the exporter (snapshots) and the checkpointer.
struct prismatic_run_set {
  std::vector<hsize_t> mem_off, file_off, len;
};

// Compress a layout's owned set (locals 0..n_owned in ascending-global
// order) into contiguous runs.  mem_base/file_base place a sub-block
// inside a combined buffer/dataset ([h|v] edges, [tri|rect] faces).
inline void
append_owned_runs(const distributed_cochain_layout& L, size_t mem_base,
                  size_t file_base, prismatic_run_set& rs) {
  const int n = L.owned_size();
  int l = 0;
  while (l < n) {
    const int g0 = L.to_global(l);
    int run = 1;
    while (l + run < n && L.to_global(l + run) == g0 + run) run++;
    rs.mem_off.push_back(mem_base + l);
    rs.file_off.push_back(file_base + g0);
    rs.len.push_back(run);
    l += run;
  }
}

}  // namespace Aperture
