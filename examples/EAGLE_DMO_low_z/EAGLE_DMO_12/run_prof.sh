#!/bin/bash

set -euo pipefail

swift="../../../swift"
params="eagle_12.yml"
steps=250

threads_list=(32)
ncells_list=(512)
cellsize_list=(15000)

for threads in "${threads_list[@]}"; do
    for ncells in "${ncells_list[@]}"; do
        for cellsize in "${cellsize_list[@]}"; do
            echo "Running: threads=${threads}, ncells=${ncells}, cellsize=${cellsize}"
            "${swift}" --cosmology --self-gravity \
                --param="GPU:ncells_per_gpu_grav_pack:${ncells}" \
                --param="GPU:gpu_grav_cell_size:${cellsize}" \
                --threads="${threads}" -n "${steps}" "${params}"
            out="timesteps_${ncells}_${cellsize}_${threads}_recursion.txt"
            echo "Saving timesteps -> ${out}"
            mv timesteps.txt "${out}"
        done
    done
done
