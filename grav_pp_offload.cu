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

/* Self gravity kernel. This is called by the pp_offload function */
//PP ALL INTERACTIONS
__global__ void self_grav_pp(int periodic, float rmax_i, double min_trunc, int *active_i, float *h_i, float *mass_i_arr, float r_s_inv, const float *x_i, const float *y_i, const float *z_i, float *a_x_i, float *a_y_i, float *a_z_i, float *pot_i, int gcount_i, int gcount_padded_i, int ci_active, int ncells, int max_cell_size, int *gcounts, int *cell_active) {

  //printf("ON GPU 1: %f %f %f %f %f %f %f \n", h_i[0], mass_i_arr[0], x_i[0], y_i[0], z_i[0], a_x_i[0], a_y_i[0]);

  int max_r_decision = 0;
  
  if (!periodic) {

    /* Not periodic -> Can always use Newtonian potential */
    doself_grav_pp_full(active_i, h_i, mass_i_arr, x_i, y_i, z_i, a_x_i, a_y_i, a_z_i, pot_i, gcount_i, gcount_padded_i, periodic, ci_active, max_r_decision, ncells, max_cell_size, gcounts, cell_active);

  } else {

    /* Do we need to use the truncated interactions ? */
    if (rmax_i > min_trunc) {

      /* Periodic but far-away cells must use the truncated potential */
      doself_grav_pp_truncated(active_i, h_i, mass_i_arr, r_s_inv, x_i, y_i, z_i, a_x_i, a_y_i, a_z_i, pot_i, gcount_i, gcount_padded_i, periodic, ci_active, max_r_decision, ncells, max_cell_size, gcounts, cell_active);
                                    

    } else {
    
    max_r_decision = 1;

      /* Periodic but close-by cells can use the full Newtonian potential */
      doself_grav_pp_full(active_i, h_i, mass_i_arr, x_i, y_i, z_i, a_x_i, a_y_i, a_z_i, pot_i, gcount_i, gcount_padded_i, periodic, ci_active, max_r_decision, ncells, max_cell_size, gcounts, cell_active);
    }
  }
  
  //printf("ON GPU 2: %f %f %f %f %f %f %f \n", h_i[0], mass_i_arr[0], x_i[0], y_i[0], z_i[0], a_x_i[0], a_y_i[0]);
}


//self grav pp offload
extern "C" void self_pp_offload(int periodic, float rmax_i, double min_trunc, const float *r_s_inv, const int *gcount_i, const int *gcount_padded_i, int ci_active, float *d_h_i, float *d_mass_i, float *d_x_i, float *d_y_i, float *d_z_i, float *d_a_x_i, float *d_a_y_i, float *d_a_z_i, float *d_pot_i, int *d_active_i, int ncells, int max_cell_size, int *gcounts, int *cell_active, GPUStream stream){

	/* memory allocation was here - this is all done in runner_main now */

	//call kernel function 
	//int nblocks = gcount_i/256;
	self_grav_pp<<<32,256, 0, stream>>>(periodic, rmax_i, min_trunc, d_active_i, d_h_i, d_mass_i, *r_s_inv, d_x_i, d_y_i, d_z_i, d_a_x_i, d_a_y_i, d_a_z_i, d_pot_i, *gcount_i, *gcount_padded_i, ci_active, ncells, max_cell_size, gcounts, cell_active);
	//check if thread idx has a particle
	
	GPUError err2 = GPUGetLastError();
	    if (err2 != GPU_SUCCESS)
	printf("Error - self_pp: %s\n", GPUGetErrorString(err2));
	
	/* memory transfer was here - this is all done in runner_main now */
}


//self grav pp offload
extern "C" void self_pp_offload_new(int periodic, float rmax_i, double min_trunc, const float *r_s_inv, const int *gcount_i, const int *gcount_padded_i, int ci_active, struct gravity_gpu_values_send *gravity_gpu_values_send_d, struct gravity_gpu_values_recv *gravity_gpu_values_recv_d, int ncells, int max_cell_size, GPUStream stream){

	/* memory allocation was here - this is all done in runner_main now */

	//call kernel function 
	//int nblocks = gcount_i/256;
	int threads = 256;
	dim3 block(threads);
	dim3 grid(ncells, (max_cell_size + threads - 1) / threads);
	size_t shmem = 0;//threads * sizeof(gravity_gpu_values_send);
	
	int max_r_decision = 0;
	
	if (!periodic) {

    /* Not periodic -> Can always use Newtonian potential */
    doself_grav_pp_full_new_refactor<<<grid,block, shmem, stream>>>(gravity_gpu_values_send_d, gravity_gpu_values_recv_d, *r_s_inv, *gcount_i, *gcount_padded_i, periodic, ci_active, max_r_decision, ncells, max_cell_size);

  } else {

    /* Do we need to use the truncated interactions ? */
    if (rmax_i > min_trunc) {

      /* Periodic but far-away cells must use the truncated potential */
      doself_grav_pp_truncated_new_refactor<<<grid,block, shmem, stream>>>(gravity_gpu_values_send_d, gravity_gpu_values_recv_d, *r_s_inv, *gcount_i, *gcount_padded_i, periodic, ci_active, max_r_decision, ncells, max_cell_size);
                                    

    } else {
    
    max_r_decision = 1;

      /* Periodic but close-by cells can use the full Newtonian potential */
      doself_grav_pp_full_new_refactor<<<grid,block, shmem, stream>>>(gravity_gpu_values_send_d, gravity_gpu_values_recv_d, *r_s_inv, *gcount_i, *gcount_padded_i, periodic, ci_active, max_r_decision, ncells, max_cell_size);
    }
  }

	//self_grav_pp_new<<<grid,block, shmem, stream>>>(periodic, rmax_i, min_trunc, *r_s_inv,*gcount_i, *gcount_padded_i, ci_active, gravity_gpu_values_send_d,  gravity_gpu_values_recv_d, ncells, max_cell_size, stream);
	//check if thread idx has a particle

	GPUError err2 = GPUGetLastError();
	    if (err2 != GPU_SUCCESS)
	printf("Error - self_pp: %s\n", GPUGetErrorString(err2));
	
	/* memory transfer was here - this is all done in runner_main now */
}

//self grav pp offload
extern "C" void pair_pp_offload_new(int periodic, float rmax_i, float rmax_j, double min_trunc, const float *r_s_inv, const int *gcount_i, const int *gcount_padded_i, const int *gcount_j, const int *gcount_padded_j, int ci_active, int cj_active, float dim_0, float dim_1, float dim_2, int symmetric, struct gravity_gpu_values_send *gravity_gpu_values_send_d, struct gravity_gpu_values_recv *gravity_gpu_values_recv_d, int ncells, int max_cell_size, GPUStream stream){

	int threads = 256;
	dim3 block(threads);
	int npairs = ncells/2;
	dim3 grid(npairs, (max_cell_size + threads - 1) / threads);
	size_t shmem = 0;//threads * sizeof(gravity_gpu_values_send);
	
	// update ci
	pair_grav_pp_kernel<<<grid, block, 0, stream>>>(gravity_gpu_values_send_d, gravity_gpu_values_recv_d, periodic, *r_s_inv, symmetric, /*swap=*/0, dim_0, dim_1, dim_2, max_cell_size, ncells);
	
	//printf("PAIRRR \n");
	
	GPUError err = GPUGetPeekAtLastError();
	if (err != GPU_SUCCESS) printf("KERNEL LAUNCH ERROR: %s\n", GPUGetErrorString(err));

	// update cj
  	if (symmetric) {
    	pair_grav_pp_kernel<<<grid, block, 0, stream>>>(gravity_gpu_values_send_d, gravity_gpu_values_recv_d, periodic, *r_s_inv, symmetric, /*swap=*/1, dim_0, dim_1, dim_2, max_cell_size, ncells);
  	}

}
