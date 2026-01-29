#!/bin/bash
#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH -J benchmarka30 #Give it something meaningful.
#SBATCH -o output.%J.out
#SBATCH -e error.%J.err
#SBATCH -p dine2
#SBATCH -A do015
#SBATCH --constraint=gpu
#SBATCH -t 0:15:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=wgfr58@durham.ac.uk

module load nvhpc
module load intel_comp/2024.2.0 compiler-rt tbb compiler mpi
module load ucx/1.17.0
module load parallel_hdf5/1.14.4
module load fftw/3.3.10
module load parmetis/4.0.3-64bit
module load gsl/2.8

# Run SWIFT
../../../swift --cosmology --self-gravity --threads=16 -n 30 small_cosmo_volume_dm.yml

