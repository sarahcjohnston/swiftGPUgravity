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
    float rmax_i,
    double min_trunc,
    const float *r_s_inv,
    const int *counts_d,
    const int *offsets_d,
    struct gravity_gpu_values_send *send_d,
    struct gravity_gpu_values_recv *recv_d,
    int ncells,
    int max_cell_size,
    GPUStream stream){

	/* memory allocation was here - this is all done in runner_main now */

	//call kernel function 
	//int nblocks = gcount_i/256;
	int threads = 256;
	dim3 block(threads);
	dim3 grid(ncells, (max_cell_size + threads - 1) / threads);
	size_t shmem = threads * sizeof(gravity_gpu_values_send);
	
	int max_r_decision = 0;
	
	if (!periodic) {

    /* Not periodic -> Can always use Newtonian potential */
    doself_grav_pp_full_new_refactor_tiled<<<grid, block, shmem, stream>>>(
    send_d, recv_d, counts_d, offsets_d, *r_s_inv,
    periodic, max_r_decision, ncells);

  } else {

    /* Do we need to use the truncated interactions ? */
    if (rmax_i > min_trunc) {

      /* Periodic but far-away cells must use the truncated potential */
      doself_grav_pp_truncated_new_refactor_tiled<<<grid, block, shmem, stream>>>(
    send_d, recv_d, counts_d, offsets_d, *r_s_inv,
    periodic, max_r_decision, ncells);
                                    
    } else {
    
    max_r_decision = 1;

      /* Periodic but close-by cells can use the full Newtonian potential */
      doself_grav_pp_full_new_refactor_tiled<<<grid, block, shmem, stream>>>(
    send_d, recv_d, counts_d, offsets_d, *r_s_inv,
    periodic, max_r_decision, ncells);
    }
  }

	GPUError err2 = GPUGetLastError();
	    if (err2 != GPU_SUCCESS)
	printf("Error - self_pp: %s\n", GPUGetErrorString(err2));
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
    int periodic, float rmax_i, float rmax_j, double min_trunc,
    const float *r_s_inv,
    const int *pair_counts_d,
    const int *pair_offsets_d,
    int ci_active, int cj_active,
    float dim_0, float dim_1, float dim_2, int symmetric,
    struct gravity_gpu_values_send *gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv *gravity_gpu_values_recv_d,
    int ncells, int max_cell_size, GPUStream stream) {

	  int threads = 256;
  dim3 block(threads);
  int npairs = ncells / 2;
  dim3 grid(npairs, (max_cell_size + threads - 1) / threads);
  size_t shmem = threads * sizeof(gravity_gpu_values_send);

  pair_grav_pp_kernel_tiled<<<grid, block, shmem, stream>>>(
      gravity_gpu_values_send_d, gravity_gpu_values_recv_d,
      pair_counts_d, pair_offsets_d,
      periodic, *r_s_inv, symmetric, /*swap=*/0,
      dim_0, dim_1, dim_2, max_cell_size, ncells);

  GPUError err1 = GPUGetPeekAtLastError();
  if (err1 != GPU_SUCCESS)
    printf("KERNEL LAUNCH ERROR (swap = 0): %s\n",
           GPUGetErrorString(err1));

  if (symmetric) {
    pair_grav_pp_kernel_tiled<<<grid, block, shmem, stream>>>(
        gravity_gpu_values_send_d, gravity_gpu_values_recv_d,
        pair_counts_d, pair_offsets_d,
        periodic, *r_s_inv, symmetric, /*swap=*/1,
        dim_0, dim_1, dim_2, max_cell_size, ncells);
  }

  GPUError err2 = GPUGetPeekAtLastError();
  if (err2 != GPU_SUCCESS)
    printf("KERNEL LAUNCH ERROR (swap = 1): %s\n",
           GPUGetErrorString(err2));
}
