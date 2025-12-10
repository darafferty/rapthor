#!/bin/bash

###############################################################################
# Rapthor Benchmark Script for Single Node run on SKAO DP cluster on AWS
#
# SLURM settings:
# ---------------------------------------------------------------------------- #
#SBATCH --job-name=sp-5859-rapthor-benchmark-test
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=150
#SBATCH --time 168:00:00
#SBATCH --output=%x-%j-leader-slurm.out
#SBATCH --error=%x-%j-leader-slurm.out
# ---------------------------------------------------------------------------- #
set -euo pipefail

# Set the required environmental variables to configure the rapthor run

export RAPTHOR_BATCH_SYSTEM="single node"
export RAPTHOR_MAX_NODES=1

# the number of processors and amount of memory per node to
# request from SLURM can be specified with the cpus_per_task (default = 0 = all)
# and mem_per_node_gb options (default = 0 = all). By setting the cpus_per_task
# value to the number of processors per node, one can ensure that each task gets
# the entire node to itself, which is the recommended way of running Rapthor
export RAPTHOR_CPUS_PER_TASK=0
export RAPTHOR_MEM_PER_NODE_GB=0

# Maximum number of cores and threads per task to use on each node (default = 0
# = all)
export RAPTHOR_MAX_CORES=0
export RAPTHOR_MAX_THREADS=0

# Number of threads to use by WSClean during deconvolution (default = 0 = 2/5
# of max_threads, but not more than 14). Higher values will speed up imaging at
# the expense of higher memory usage
export RAPTHOR_DECONV_THREADS=0

# Number of threads to use by WSClean for parallel gridding (default = 0 = 2/5
# of max_threads, but not more than 6)
export RAPTHOR_PARALLEL_GRIDDING_THREADS=0

# run benchmark
source rapthor_skao_benchmark_runner.sh