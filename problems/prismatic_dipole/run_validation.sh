#!/bin/bash
# Run vacuum dipole validation at multiple refinement levels.
# N_r is scaled with L so that dr ~ r*dtheta.

set -e

LEVELS="${@:-3 4 5 6}"
DIRS=""

for L in $LEVELS; do
  # N_r ~ 5 * 2^(L-1) to match angular resolution
  NR=$((5 * (1 << (L - 1))))
  DIR="Data_validation_L${L}"
  DIRS="$DIRS $DIR"
  mkdir -p "$DIR"

  echo "=== Running L=$L, N_r=$NR ==="
  cat > config.toml << EOF
dt = 0.01
max_steps = 1
subdivision_level = $L
N_r = $NR
r_min = 1.0
r_max = 20.0
Bp = 1.0
Omega = 1.0
obliquity = 0.7854
damping_length = $((NR / 4))
damping_coef = 0.05
fld_output_interval = 1
output_dir = "$DIR"
sph_N_theta = 90
sph_N_phi = 180
EOF
  ./bin/vacuum_dipole
  echo ""
done

echo "=== Running validation ==="
python3 validate_vacuum_dipole.py $DIRS
