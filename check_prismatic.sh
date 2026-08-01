#!/usr/bin/env bash
# Fast regression check for the prismatic subsystem: builds every target
# that compiles prismatic code and runs the prismatic-tagged test suites.
# Usage: ./check_prismatic.sh [build_dir]   (default: build)
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
build_dir="${1:-$repo_root/build}"

if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
  echo "error: $build_dir is not a configured build directory" >&2
  exit 1
fi

targets=(
  tests
  prismatic_dipole
  vacuum_dipole
  cavity_resonator
  streaming_test
  prismatic_interp
  prismatic_wald   # shelved GR problem — kept compiling on purpose
  ptc_orbit_test
  test_prismatic_mpi_backend_multirank
)

cmake --build "$build_dir" --target "${targets[@]}" -j"$(nproc)"

"$repo_root/bin/tests" '[prismatic]'
"$repo_root/bin/tests" '[deposit]'
"$repo_root/bin/tests" '[mesh_ptrs]'

echo "check_prismatic: all builds and tests passed"
