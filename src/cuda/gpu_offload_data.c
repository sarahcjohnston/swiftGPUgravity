/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2025 Abouzied M. A. Nasar (abouzied.nasar@manchester.ac.uk)
 *                    Mladen Ivkovic (mladen.ivkovic@durham.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

/**
 * @file cuda/gpu_offload_data.c
 * @brief functions related to the gpu_offload_data struct, containing data
 * required for offloading
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "gpu_offload_data.h"

#include "cuda_config.h"

#include "task.h"

#include "gpu_mapping.h"

#include "error.h"

/**
 * @brief initialise GPU data buffers (including their associated metadata)
 *
 * @param buf: the buffers to be initialised
 * @param params: global gpu packing parameters
 * @param send_struct_size: size of struct used for send arrays (both host and
 * device)
 * @param recv_struct_size: size of struct used for recv arrays (both host and
 * device)
 */
void gpu_data_buffers_init(struct gpu_offload_data *buf,
                           const struct gpu_global_pack_params *params,
                           const size_t send_struct_size,
                           const size_t recv_struct_size) {

  /* Grab some handles */
  /*TODO: Do this properly rather than set it to 1*/
  const size_t n_bundles = 1;//params->n_bundles;
  const size_t part_buffer_size = params->part_buffer_size;

  /* Initialise and set up metadata */
  struct gpu_pack_metadata *md = &(buf->md);
  gpu_pack_metadata_init(md, params);

  /* Now allocate arrays */
  GPUError cu_error;

  /* Now allocate memory for Buffer and GPU particle arrays */
  cu_error = GPUMalloc((void **)&buf->d_parts_send_d,
                        part_buffer_size * send_struct_size);
  assert(cu_error == GPUSuccess);

  cu_error = GPUMalloc((void **)&buf->d_parts_recv_d,
                        part_buffer_size * recv_struct_size);
  assert(cu_error == GPUSuccess);

  cu_error = GPUHostMalloc((void **)&buf->parts_send_d,
                            part_buffer_size * send_struct_size);
  assert(cu_error == GPUSuccess);

  cu_error = GPUHostMalloc((void **)&buf->parts_recv_d,
                            part_buffer_size * recv_struct_size);
  assert(cu_error == GPUSuccess);

  /* Create space for cuda events */
  buf->event_end = (GPUEvent *)malloc(n_bundles * sizeof(GPUEvent));

  for (size_t i = 0; i < n_bundles; i++) {
    cu_error = GPUEventCreate(&(buf->event_end[i]));
    assert(cu_error == GPUSuccess);
  }

#ifdef SWIFT_DEBUG_CHECKS
  md->send_struct_size = send_struct_size;
  md->recv_struct_size = recv_struct_size;
#endif

  /* Now allocate arrays GPU-only metadata arrays*/

  /*Data required for unique sorting*/
  int size_of_cell_start_end = sizeof(int4) * params->cell_start_end_buffer_size;

  /*Allocate memory for cell start and end data on host*/
  cu_error = GPUHostMalloc((void **)&buf->gpu_md.cell_i_j_start_end,
                            size_of_cell_start_end);
  assert(cu_error == GPUSuccess);

  /*Allocate memory for cell start and end data on device*/
  cu_error = GPUMalloc((void **)&buf->gpu_md.d_cell_i_j_start_end,
                            size_of_cell_start_end);
  assert(cu_error == GPUSuccess);

  /*Since we have unique sorting this should be quite a bit less than
   * the space needed to store enough blocks to work on part_buffer_size
   * particles. Possibly problematic as 10 pulled out of the air...*/
  const size_t n_blocks = params->cuda_blockid_buffer_size;//(10 * part_buffer_size + GPU_THREAD_BLOCK_SIZE - 1)/GPU_THREAD_BLOCK_SIZE;

  /*Allocate memory for array containing
   * leaf_computation_id for each cuda block*/
  cu_error = GPUHostMalloc((void **)&buf->gpu_md.block_leaf_id,
                            n_blocks * sizeof(int2));
  assert(cu_error == GPUSuccess);

  /*Repeat for device copy*/
  cu_error = GPUMalloc((void **)&buf->gpu_md.d_block_leaf_id,
                            n_blocks * sizeof(int2));
  assert(cu_error == GPUSuccess);

}

/**
 * @brief perform the initialisations required at the start of each step
 */
void gpu_data_buffers_init_step(struct gpu_offload_data *buf) {

  struct gpu_pack_metadata *md = &buf->md;
  gpu_pack_metadata_init_step(md);
  gpu_data_buffers_reset(buf);
}

/**
 * @brief reset (zero out) the data buffers.
 */
void gpu_data_buffers_reset(struct gpu_offload_data *buf) {

#ifdef SWIFT_DEBUG_CHECKS

  /* In principle, if our book-keeping is correct, we shouldn't ever
   * need to zero out the contents. So we don't do it outside of debug
   * mode. */

  const struct gpu_global_pack_params pars = buf->md.params;
  const struct gpu_pack_metadata md = buf->md;

  memset(buf->parts_send_d, 0, pars.part_buffer_size * md.send_struct_size);
  memset(buf->parts_recv_d, 0, pars.part_buffer_size * md.recv_struct_size);

  /* Can't do this from the host side, would need to launch cuda kernel
   * We can actually. cudaMemSet() */
  /* initialise to just some meaningless value to silence the compiler */
//  cudaError_t cu_error = cudaErrorMemoryAllocation;
//
//  cu_error =
//      cudaMemset(&buf->d_parts_send_d,
//      0, pars.part_buffer_size * md.send_struct_size);
//  assert(cu_error == cudaSuccess);
//
//  cu_error =
//      cudaMemset(&buf->d_parts_recv_d,
//      0, pars.part_buffer_size * md.recv_struct_size);
//  assert(cu_error == cudaSuccess);

#endif
}

/**
 * @brief Free everything you allocated.
 */
void gpu_data_buffers_free(struct gpu_offload_data *buf) {

  struct gpu_pack_metadata *md = &(buf->md);
  gpu_pack_metadata_free(md);

  GPUError cu_error = GPUSuccess;

  cu_error = GPUFree(buf->d_parts_send_d);
  assert(cu_error == GPUSuccess);

  cu_error = GPUFree(buf->d_parts_recv_d);
  assert(cu_error == GPUSuccess);

  cu_error = GPUFreeHost(buf->parts_send_d);
  assert(cu_error == GPUSuccess);

  cu_error = GPUFreeHost(buf->parts_recv_d);
  assert(cu_error == GPUSuccess);

  free((void *)buf->event_end);
}

#ifdef __cplusplus
}
#endif
