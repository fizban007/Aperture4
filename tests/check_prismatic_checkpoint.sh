#!/usr/bin/env bash
# Checkpoint/restart acceptance for the distributed prismatic PIC code
# (CHECKPOINT_RESTART_PLAN.md step 4).  Drives test_prismatic_pic_multirank
# through the full matrix:
#
#   continuity   2N straight vs N + checkpoint + fresh process + N, same
#                rank count (1 and 8).  HOST builds must be BITWISE
#                identical; GPU builds compare fields at the atomic
#                FP-reorder tolerance (bitwise is unattainable there by
#                construction) and LIVE counts exactly.
#   elasticity   checkpoint at 8 ranks, restart at 1 / 8-with-node-tiling /
#                20 ranks: LIVE exact, ESUM to ~7 digits, continued
#                evolution within cross-rank tolerance.
#   crash-safety a truncated generation + garbage tmp/ must be skipped by
#                restart_from = auto in favor of the last complete one.
#
# Usage: tests/check_prismatic_checkpoint.sh <test_binary> [--bitwise]
#   <test_binary>  path to test_prismatic_pic_multirank
#   --bitwise      require bit-identical continuity dumps (host builds)
#
# Everything runs in a temp directory; exits nonzero on any failure.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
bin="${1:?usage: $0 <test_prismatic_pic_multirank> [--bitwise]}"
bitwise="${2:-}"
cfg_src="$repo_root/tests/config_prismatic_pic_multirank.toml"
work="$(mktemp -d "${TMPDIR:-/tmp}/prismatic_ckpt.XXXXXX")"
trap 'rm -rf "$work"' EXIT
cd "$work"

datasets=(E_e B_f J_e rho rho_abs gamma_wsum)
# GPU float atomics reorder deposits: two INDEPENDENT straight runs of
# this config differ by up to ~2e-4 absolute in J_e (measured), so the
# delta must sit above the run-to-run envelope.  Real restart bugs show
# up at O(field scale), orders of magnitude above this.
delta=1e-3

run() {  # run <nranks> <config> -> prints LIVE and ESUM lines
  local n="$1" cfg="$2"
  if [[ "$n" == 1 ]]; then
    "$bin" -c "$cfg"
  else
    mpirun --oversubscribe -n "$n" "$bin" -c "$cfg"
  fi
}

mkcfg() {  # mkcfg <name> <output_dir> [extra lines...]
  local name="$1" out="$2"; shift 2
  sed "s|output_dir = .*|output_dir = \"$out\"|" "$cfg_src" > "$name"
  local line; for line in "$@"; do echo "$line" >> "$name"; done
}

check_counts() {  # check_counts <log> <live>
  grep -q "^LIVE $2\$" "$1" || { echo "FAIL: LIVE mismatch in $1"; exit 1; }
  grep -q "^MISOWNED 0\$" "$1" || { echo "FAIL: MISOWNED != 0 in $1"; exit 1; }
  grep -q "ALL PASS" "$1" || { echo "FAIL: $1 did not pass"; exit 1; }
}

esum_of() { grep "^ESUM" "$1" | awk '{print $2}'; }

check_esum() {  # check_esum <a> <b>  (~7 significant digits)
  python3 - "$1" "$2" <<'PY'
import sys
a, b = float(sys.argv[1]), float(sys.argv[2])
rel = abs(a - b) / max(abs(a), 1e-30)
sys.exit(0 if rel < 1e-6 else 1)
PY
}

check_dumps() {  # check_dumps <fileA> <fileB> <mode: exact|delta>
  local a="$1" b="$2" mode="$3" ds
  for ds in "${datasets[@]}"; do
    if [[ "$mode" == exact ]]; then
      h5diff "$a" "$b" "/$ds" > /dev/null ||
        { echo "FAIL: $ds differs (bitwise) between $a and $b"; exit 1; }
    else
      h5diff --delta="$delta" "$a" "$b" "/$ds" > /dev/null ||
        { echo "FAIL: $ds differs beyond $delta between $a and $b"; exit 1; }
    fi
  done
}

dump_mode=delta
[[ "$bitwise" == "--bitwise" ]] && dump_mode=exact

echo "== continuity (1 rank) =="
mkcfg c1a.toml D1_straight
run 1 c1a.toml > log_1a.txt 2>&1; check_counts log_1a.txt 630
mkcfg c1b.toml D1_ck "checkpoint_interval = 20"
run 1 c1b.toml > log_1b.txt 2>&1; check_counts log_1b.txt 630
mkcfg c1c.toml D1_rs 'restart_from = "D1_ck/ckpt/ckpt_21"'
run 1 c1c.toml > log_1c.txt 2>&1; check_counts log_1c.txt 630
check_esum "$(esum_of log_1b.txt)" "$(esum_of log_1c.txt)" ||
  { echo "FAIL: 1-rank continuity ESUM"; exit 1; }
check_dumps D1_straight/step_000040.h5 D1_rs/step_000040.h5 "$dump_mode"
echo "   OK ($dump_mode dumps)"

echo "== continuity (8 ranks) =="
mkcfg c8a.toml D8_straight
run 8 c8a.toml > log_8a.txt 2>&1; check_counts log_8a.txt 630
mkcfg c8b.toml D8_ck "checkpoint_interval = 20"
run 8 c8b.toml > log_8b.txt 2>&1; check_counts log_8b.txt 630
mkcfg c8c.toml D8_rs 'restart_from = "D8_ck/ckpt/ckpt_21"'
run 8 c8c.toml > log_8c.txt 2>&1; check_counts log_8c.txt 630
check_esum "$(esum_of log_8b.txt)" "$(esum_of log_8c.txt)" ||
  { echo "FAIL: 8-rank continuity ESUM"; exit 1; }
check_dumps D8_straight/step_000040.h5 D8_rs/step_000040.h5 "$dump_mode"
echo "   OK ($dump_mode dumps)"

echo "== elasticity (8 -> 1, 8 -> 8 tiled, 8 -> 20) =="
mkcfg e1.toml E_8to1 'restart_from = "D8_ck/ckpt/ckpt_21"'
run 1 e1.toml > log_e1.txt 2>&1; check_counts log_e1.txt 630
grep -q "REDISTRIBUTING" log_e1.txt || { echo "FAIL: no redistribution log"; exit 1; }
mkcfg e2.toml E_8tile 'restart_from = "D8_ck/ckpt/ckpt_21"' "ranks_per_node = 4"
run 8 e2.toml > log_e2.txt 2>&1; check_counts log_e2.txt 630
mkcfg e3.toml E_8to20 'restart_from = "D8_ck/ckpt/ckpt_21"'
run 20 e3.toml > log_e3.txt 2>&1; check_counts log_e3.txt 630
for l in log_e1.txt log_e2.txt log_e3.txt; do
  check_esum "$(esum_of log_8a.txt)" "$(esum_of $l)" ||
    { echo "FAIL: elasticity ESUM in $l"; exit 1; }
done
check_dumps D8_straight/step_000040.h5 E_8to1/step_000040.h5 delta
check_dumps D8_straight/step_000040.h5 E_8tile/step_000040.h5 delta
check_dumps D8_straight/step_000040.h5 E_8to20/step_000040.h5 delta
echo "   OK"

echo "== crash safety (auto skips truncated generation) =="
mkdir -p D8_ck/ckpt/tmp && echo garbage > D8_ck/ckpt/tmp/checkpoint.h5
mkdir -p D8_ck/ckpt/ckpt_99
head -c 100000 D8_ck/ckpt/ckpt_41/checkpoint.h5 > D8_ck/ckpt/ckpt_99/checkpoint.h5
mkcfg cs.toml D_auto 'checkpoint_dir = "D8_ck/ckpt"' 'restart_from = "auto"'
run 8 cs.toml > log_cs.txt 2>&1; check_counts log_cs.txt 630
grep -q "Skipping incomplete generation" log_cs.txt ||
  { echo "FAIL: truncated generation not skipped"; exit 1; }
grep -q "ckpt_41/checkpoint.h5" log_cs.txt ||
  { echo "FAIL: auto did not pick the last complete generation"; exit 1; }
echo "   OK"

echo "ALL CHECKPOINT TESTS PASS"
