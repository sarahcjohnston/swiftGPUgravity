#include "engine.h"
#include "error.h"
#include "gpu_functions.h"
#include "runner.h"

#include <hip/hip_runtime_api.h>
#include <stdlib.h>

/**
 * @brief Initialise the GPU-specific state attached to a runner.
 *
 * @param r The runner whose GPU state to initialise.
 */
void runner_gpu_init(struct runner* r) {

  struct gpu_runner* gpu = &r->gpu;
  struct engine* e = r->e;

  hipSetDevice(0);

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);

  const float tot_gpu_mem = (float)prop.totalGlobalMem;
  const float avail_gpu_mem = 0.8f * tot_gpu_mem;
  const int max_cell_size = space_subsize_self_grav + 100;
  const float allarray = 4 * max_cell_size * 24;
  const int ncells_tot = avail_gpu_mem / allarray;
  const int n_threads = e->nr_threads;
  const int ncells_queue = ncells_tot / n_threads;

  gpu->grav_batch_ncells = ncells_queue / 50;
  gpu->grav_max_cell_size = max_cell_size;

  if (gpu->grav_batch_ncells == 0) gpu->grav_batch_ncells = 8;
  gpu->grav_batch_ncells = 4;

  gpu->grav_batch_self_count = 0;
  gpu->grav_batch_pair_count = 0;

  hipMalloc((void**)&gpu->gravity_gpu_values_send_self_d,
            gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                sizeof(struct gravity_gpu_values_send));
  hipHostMalloc((void**)&gpu->gravity_gpu_values_send_self,
                gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                    sizeof(struct gravity_gpu_values_send),
                0);

  hipMalloc((void**)&gpu->gravity_gpu_values_send_pair_d,
            gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                sizeof(struct gravity_gpu_values_send));
  hipHostMalloc((void**)&gpu->gravity_gpu_values_send_pair,
                gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                    sizeof(struct gravity_gpu_values_send),
                0);

  hipMalloc((void**)&gpu->gravity_gpu_values_recv_self_d,
            gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                sizeof(struct gravity_gpu_values_recv));
  hipHostMalloc((void**)&gpu->gravity_gpu_values_recv_self,
                gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                    sizeof(struct gravity_gpu_values_recv),
                0);

  hipMalloc((void**)&gpu->gravity_gpu_values_recv_pair_d,
            gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                sizeof(struct gravity_gpu_values_recv));
  hipHostMalloc((void**)&gpu->gravity_gpu_values_recv_pair,
                gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                    sizeof(struct gravity_gpu_values_recv),
                0);

  gpu->grav_cells_self = malloc(gpu->grav_batch_ncells * sizeof(struct cell*));
  gpu->grav_cells_pair = malloc(gpu->grav_batch_ncells * sizeof(struct cell*));
  gpu->grav_tasks_self = malloc(gpu->grav_batch_ncells * sizeof(struct task*));
  gpu->grav_tasks_pair = malloc(gpu->grav_batch_ncells * sizeof(struct task*));
  gpu->cell_active = malloc(gpu->grav_batch_ncells * sizeof(int));

  if (gpu->grav_cells_self == NULL || gpu->grav_cells_pair == NULL ||
      gpu->grav_tasks_self == NULL || gpu->grav_tasks_pair == NULL ||
      gpu->cell_active == NULL)
    error("Failed to allocate runner GPU host metadata arrays.");
}

/**
 * @brief Clean the GPU-specific state attached to a runner.
 *
 * @param r The runner whose GPU state to clean.
 */
void runner_gpu_clean(struct runner* r) {

  struct gpu_runner* gpu = &r->gpu;

  hipFreeHost(gpu->gravity_gpu_values_send_self);
  hipFreeHost(gpu->gravity_gpu_values_recv_self);
  hipFree(gpu->gravity_gpu_values_send_self_d);
  hipFree(gpu->gravity_gpu_values_recv_self_d);

  hipFreeHost(gpu->gravity_gpu_values_send_pair);
  hipFreeHost(gpu->gravity_gpu_values_recv_pair);
  hipFree(gpu->gravity_gpu_values_send_pair_d);
  hipFree(gpu->gravity_gpu_values_recv_pair_d);

  free(gpu->grav_cells_self);
  free(gpu->grav_tasks_self);
  free(gpu->grav_cells_pair);
  free(gpu->grav_tasks_pair);
  free(gpu->cell_active);

  gpu->gravity_gpu_values_send_self = NULL;
  gpu->gravity_gpu_values_send_self_d = NULL;
  gpu->gravity_gpu_values_send_pair = NULL;
  gpu->gravity_gpu_values_send_pair_d = NULL;
  gpu->gravity_gpu_values_recv_self = NULL;
  gpu->gravity_gpu_values_recv_self_d = NULL;
  gpu->gravity_gpu_values_recv_pair = NULL;
  gpu->gravity_gpu_values_recv_pair_d = NULL;
  gpu->grav_cells_self = NULL;
  gpu->grav_tasks_self = NULL;
  gpu->grav_cells_pair = NULL;
  gpu->grav_tasks_pair = NULL;
  gpu->cell_active = NULL;
  gpu->grav_batch_self_count = 0;
  gpu->grav_batch_pair_count = 0;
  gpu->grav_batch_ncells = 0;
  gpu->grav_max_cell_size = 0;
}
