#!/bin/bash
# L6 Deutsch movie series: alpha = 0, 30, 60, 90 deg, one clean period each.
set -e
cd "$(dirname "$0")"
for a in 90 60 30 00; do
  echo "=== Running alpha=${a} ==="
  ./bin/vacuum_dipole -c config_deutsch_L6_movie_a${a}.toml > run_a${a}.log 2>&1
  rm -f Data_deutsch_L6_movie_a${a}/mesh.h5 Data_deutsch_L6_movie_a${a}/step_*.h5
  echo "=== alpha=${a} done: $(ls Data_deutsch_L6_movie_a${a}/sph_0*.h5 | wc -l) sph frames ==="
done
echo "ALL RUNS COMPLETE"
