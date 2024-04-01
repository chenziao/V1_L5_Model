#!/bin/sh

#SBATCH --partition compute
#SBATCH --nodes=2
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=60
#SBATCH --account=umc113

#SBATCH -J V1_sim 
#SBATCH -o  ./stdout/V1_sim.o%j.out
#SBATCH -e  ./stdout/V1_sim.e%j.error
#SBATCH -t 0-48:00:00  # days-hours:minutes

#SBATCH --mem-per-cpu=2G  # memory per core; default is 1GB/core

## send mail to this address, alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zc963@mail.missouri.edu

START=$(date)
echo "Started running at $START."

export HDF5_USE_FILE_LOCKING=FALSE
unset DISPLAY

mpirun ./components/mechanisms/x86_64/special -mpi -python run_network.py config_baseline.json True # args: config file, whether use coreneuron

END=$(date)
echo "Done running simulation at $END"

TRIALNAME="baseline_1"
mkdir ../Analysis/simulation_results/"$TRIALNAME"
cp -a output/. ../Analysis/simulation_results/"$TRIALNAME"
cp -a ecp_tmp/. ../Analysis/simulation_results/"$TRIALNAME"
