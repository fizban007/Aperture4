#!/bin/bash
#SBATCH -A ast229
#SBATCH -J bondi
#SBATCH -o %x-%j.out
#SBATCH -t 0:20:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2

cd $PROJWORK/ast229/mehlhaff/Aperture4/problems/gr_2d_kerr_schild/bin/
module reset
source $PROJWORK/ast229/mehlhaff/Aperture4/machines/frontier.20260408

# Use OLCF-recommended CPU-GPU affinity binding.  Each Frontier node has
# 8 GCDs and 8 NUMA domains (L3 regions).  This maps each local rank to
# the correct CPU cores + closest GPU, respecting the NUMA topology:
#   local rank 0 → GPU 4, cores 48-55  (NUMA 7 — note the non-sequential
#   local rank 1 → GPU 5, cores 56-63   GCD-to-NUMA mapping on Frontier)
#   local rank 2 → GPU 2, cores 16-23
#   local rank 3 → GPU 3, cores 24-31
#   local rank 4 → GPU 6, cores 32-39
#   local rank 5 → GPU 7, cores 40-47
#   local rank 6 → GPU 0, cores  0-7
#   local rank 7 → GPU 1, cores  8-15
srun -n 16 --ntasks-per-node 8 --gpus-per-node 8 \
     --gpu-bind closest \
     --cpu-bind=map_cpu:49,57,17,25,33,41,1,9 \
     ./bondi -c config_bondi.toml
