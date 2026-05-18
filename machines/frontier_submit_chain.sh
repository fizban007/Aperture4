#!/bin/bash
# Submit N copies of an sbatch script as an afterok-dependent chain.
#
# Usage:
#   ./frontier_submit_chain.sh <num_jobs> <sbatch_script> [extra sbatch args...]
#
# Example:
#   ./frontier_submit_chain.sh 6 frontier_sbatch_chain_template.sh
#
# The first job runs immediately; each subsequent job runs only if the
# previous one exited cleanly (afterok). If a job is killed by SLURM
# (timeout, OOM) it exits non-zero and the chain stops automatically.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <num_jobs> <sbatch_script> [extra sbatch args...]" >&2
  exit 1
fi

N=$1; shift
SCRIPT=$1; shift

if [[ ! -f "$SCRIPT" ]]; then
  echo "error: sbatch script '$SCRIPT' not found" >&2
  exit 1
fi

prev=""
for i in $(seq 1 "$N"); do
  if [[ -z "$prev" ]]; then
    jid=$(sbatch --parsable "$SCRIPT" "$@")
    echo "Submitted chain link $i/$N: jobid $jid"
  else
    jid=$(sbatch --parsable --dependency=afterok:"$prev" "$SCRIPT" "$@")
    echo "Submitted chain link $i/$N: jobid $jid (after $prev)"
  fi
  prev=$jid
done
