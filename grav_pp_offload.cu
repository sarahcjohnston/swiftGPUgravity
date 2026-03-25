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

//self grav pp offload
extern "C" void self_pp_offload_new(int periodic, float rmax_i, double min_trunc, const float *r_s_inv, const int *gcount_i, const int *gcount_padded_i, int ci_active, struct gravity_gpu_values_send *gravity_gpu_values_send_d, struct gravity_gpu_values_recv *gravity_gpu_values_recv_d, int ncells, int max_cell_size, GPUStream stream){

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
    doself_grav_pp_full_new_refactor_tiled<<<grid,block, shmem, stream>>>(gravity_gpu_values_send_d, gravity_gpu_values_recv_d, *r_s_inv, *gcount_i, *gcount_padded_i, periodic, ci_active, max_r_decision, ncells, max_cell_size);

  } else {

    /* Do we need to use the truncated interactions ? */
    if (rmax_i > min_trunc) {

      /* Periodic but far-away cells must use the truncated potential */
      doself_grav_pp_truncated_new_refactor_tiled<<<grid,block, shmem, stream>>>(gravity_gpu_values_send_d, gravity_gpu_values_recv_d, *r_s_inv, *gcount_i, *gcount_padded_i, periodic, ci_active, max_r_decision, ncells, max_cell_size);
                                    

    } else {
    
    max_r_decision = 1;

      /* Periodic but close-by cells can use the full Newtonian potential */
      doself_grav_pp_full_new_refactor_tiled<<<grid,block, shmem, stream>>>(gravity_gpu_values_send_d, gravity_gpu_values_recv_d, *r_s_inv, *gcount_i, *gcount_padded_i, periodic, ci_active, max_r_decision, ncells, max_cell_size);
    }
  }

	GPUError err2 = GPUGetLastError();
	    if (err2 != GPU_SUCCESS)
	printf("Error - self_pp: %s\n", GPUGetErrorString(err2));
}

//self grav pp offload
extern "C" void pair_pp_offload_new(int periodic, float rmax_i, float rmax_j, double min_trunc, const float *r_s_inv, const int *gcount_i, const int *gcount_padded_i, const int *gcount_j, const int *gcount_padded_j, int ci_active, int cj_active, float dim_0, float dim_1, float dim_2, int symmetric, struct gravity_gpu_values_send *gravity_gpu_values_send_d, struct gravity_gpu_values_recv *gravity_gpu_values_recv_d, int ncells, int max_cell_size, GPUStream stream){

	int threads = 256;
	dim3 block(threads);
	int npairs = ncells/2;
	dim3 grid(npairs, (max_cell_size + threads - 1) / threads);
	size_t shmem = threads * sizeof(gravity_gpu_values_send);
	
	// update ci
	pair_grav_pp_kernel_tiled<<<grid, block, shmem, stream>>>(gravity_gpu_values_send_d, gravity_gpu_values_recv_d, periodic, *r_s_inv, symmetric, /*swap=*/0, dim_0, dim_1, dim_2, max_cell_size, ncells);
	
	GPUError err1 = GPUGetPeekAtLastError();
	if (err1 != GPU_SUCCESS) printf("KERNEL LAUNCH ERROR (swap = 0): %s\n", GPUGetErrorString(err1));

	// update cj
  	if (symmetric) {
    	pair_grav_pp_kernel_tiled<<<grid, block, shmem, stream>>>(gravity_gpu_values_send_d, gravity_gpu_values_recv_d, periodic, *r_s_inv, symmetric, /*swap=*/1, dim_0, dim_1, dim_2, max_cell_size, ncells);
  	}
  	
  	GPUError err2 = GPUGetPeekAtLastError();
	if (err2 != GPU_SUCCESS) printf("KERNEL LAUNCH ERROR (swap = 1): %s\n", GPUGetErrorString(err2));

}
