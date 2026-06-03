#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include "gpu_mapping.h"
#include <unistd.h>

#include "externalfunctions.cu"
#include "multipole_struct.h"
#include "periodic.h"

/**
 * @brief Launch the GPU self P-P gravity kernel for a packed batch.
 *
 * Selects the full or truncated self-interaction kernel depending on the
 * periodic boundary conditions and the cell truncation criterion, then launches
 * the tiled GPU kernel on the provided stream.
 *
 * @param periodic Whether periodic boundary conditions are enabled.
 * @param rmax_i Bounding radius for the packed cell.
 * @param min_trunc Minimum truncation radius for periodic forces.
 * @param r_s_inv Inverse splitting scale for periodic mesh forces.
 * @param gcount_i Number of particles in the packed cell.
 * @param gcount_padded_i Padded particle count in the packed cell.
 * @param gravity_gpu_values_send_d Device send buffer.
 * @param gravity_gpu_values_recv_d Device receive buffer.
 * @param ncells Number of packed cells in this batch.
 * @param max_cell_size Maximum number of particles per packed cell.
 * @param stream GPU stream used for the kernel launch.
 */
extern "C" void self_pp_offload_new(
    int periodic,
    const float *rmax_d,
    double min_trunc,
    const float *r_s_inv,
    const int *counts_d,
    const int *offsets_d,
    const int *active_counts_d,
    const int *active_offsets_d,
    const int *active_index_d,
    struct gravity_gpu_values_send *send_d,
    struct gravity_gpu_values_recv *recv_d,
    int ncells,
    int max_cell_size,
    int max_active_count,
    GPUStream stream) {

  int threads = 256;
  dim3 block(threads);
  dim3 grid(ncells, (max_active_count + threads - 1) / threads);
  size_t shmem = threads * sizeof(gravity_gpu_values_send);

  self_grav_pp_kernel_tiled<<<grid, block, shmem, stream>>>(
      send_d, recv_d, counts_d, offsets_d, active_counts_d, active_offsets_d, active_index_d,
      rmax_d, *r_s_inv, periodic, (float)min_trunc, ncells, max_cell_size);

  GPUError err = GPUGetLastError();
  if (err != GPU_SUCCESS)
    printf("Error - self_pp: %s\n", GPUGetErrorString(err));
}

/**
 * @brief Launch the GPU pair P-P gravity kernel for a packed batch.
 *
 * Launches the tiled pair-interaction kernel for the i-side particles of each
 * packed pair, and optionally for the j-side particles as well when symmetric
 * updates are requested.
 *
 * @param periodic Whether periodic boundary conditions are enabled.
 * @param rmax_i Bounding radius for cell i.
 * @param rmax_j Bounding radius for cell j.
 * @param min_trunc Minimum truncation radius for periodic forces.
 * @param r_s_inv Inverse splitting scale for periodic mesh forces.
 * @param gcount_i Number of particles in cell i.
 * @param gcount_padded_i Padded particle count for cell i.
 * @param gcount_j Number of particles in cell j.
 * @param gcount_padded_j Padded particle count for cell j.
 * @param ci_active Whether cell i is active on this rank.
 * @param cj_active Whether cell j is active on this rank.
 * @param dim_0 Domain size in the x dimension.
 * @param dim_1 Domain size in the y dimension.
 * @param dim_2 Domain size in the z dimension.
 * @param symmetric Whether to update both cells.
 * @param gravity_gpu_values_send_d Device send buffer.
 * @param gravity_gpu_values_recv_d Device receive buffer.
 * @param ncells Number of packed cells in this batch.
 * @param max_cell_size Maximum number of particles per packed cell.
 * @param stream GPU stream used for the kernel launch.
 */
extern "C" void pair_pp_offload_new(
    int periodic,
    double min_trunc,
    const float *r_s_inv,
    const int *pair_use_full_d,
    const int *pair_side_active_offsets_d,
    const int *pair_counts_d,
    const int *pair_offsets_d,
    const int *pair_active_counts_d,
    const int *pair_active_offsets_d,
    const int *pair_active_index_d,
    const int *pair_pair_i_d,
    const int *pair_pair_j_d,
    int npairs,
    int nslots,
    float dim_0, float dim_1, float dim_2,
    const int *pair_cell_flags_d,
    const float4 *send_pair_pos_mass_d,
    const float *send_pair_h_d,
    struct gravity_gpu_values_recv *gravity_gpu_values_recv_d,
    int ncells,
    int max_cell_size,
    int max_active_count,
    GPUStream stream) {

  (void)min_trunc;

  const int threads = 256;
  dim3 block(threads);

  dim3 grid(npairs, (max_active_count + threads - 1) / threads);

  const size_t shmem =
      threads * sizeof(float4) +
      threads * sizeof(float);

  pair_grav_pp_kernel_tiled<<<grid, block, shmem, stream>>>(
      pair_cell_flags_d,
      pair_use_full_d,
      pair_side_active_offsets_d,
      send_pair_pos_mass_d,
      send_pair_h_d,
      gravity_gpu_values_recv_d,
      pair_counts_d,
      pair_offsets_d,
      pair_active_counts_d,
      pair_active_offsets_d,
      pair_active_index_d,
      pair_pair_i_d,
      pair_pair_j_d,
      npairs,
      nslots,
      periodic,
      *r_s_inv,
      0,
      dim_0, dim_1, dim_2,
      nslots);

  pair_grav_pp_kernel_tiled<<<grid, block, shmem, stream>>>(
      pair_cell_flags_d,
      pair_use_full_d,
      pair_side_active_offsets_d,
      send_pair_pos_mass_d,
      send_pair_h_d,
      gravity_gpu_values_recv_d,
      pair_counts_d,
      pair_offsets_d,
      pair_active_counts_d,
      pair_active_offsets_d,
      pair_active_index_d,
      pair_pair_i_d,
      pair_pair_j_d,
      npairs,
      nslots,
      periodic,
      *r_s_inv,
      1,
      dim_0, dim_1, dim_2,
      nslots);
}
