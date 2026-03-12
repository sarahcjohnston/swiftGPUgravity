#include "active.h"
#include "engine.h"
#include "error.h"
#include "gpu_functions.h"
#include "runner.h"
#include "runner_doiact_grav.h"
#include "scheduler.h"
#include "timers.h"

#include <hip/hip_runtime_api.h>
#include <stdlib.h>

extern void pair_pp_offload_new(
    int periodic, float rmax_i, float rmax_j, double min_trunc,
    const float* r_s_inv, const int* gcount_i, const int* gcount_padded_i,
    const int* gcount_j, const int* gcount_padded_j, int ci_active,
    int cj_active, float dim_0, float dim_1, float dim_2, int symmetric,
    struct gravity_gpu_values_send* gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_d, int ncells,
    int max_cell_size, hipStream_t stream);

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

  const size_t send_bytes = gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                            sizeof(struct gravity_gpu_values_send);
  const size_t recv_bytes = gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                            sizeof(struct gravity_gpu_values_recv);
  const size_t total_device_bytes = 2 * send_bytes + 2 * recv_bytes;
  const size_t total_host_pinned_bytes = 2 * send_bytes + 2 * recv_bytes;

  if (r->id == 0) {
    message("GPU device: %s", prop.name);
    message("Total GPU memory: %.2f B", (float)prop.totalGlobalMem);
    message("Max cell size: %i", gpu->grav_max_cell_size);
    message("ncells per pack: %i", gpu->grav_batch_ncells);
    message("Per-runner device buffer bytes: %zu", total_device_bytes);
    message("Per-runner host pinned bytes: %zu", total_host_pinned_bytes);
  }

  gpu->grav_batch_self_count = 0;
  gpu->grav_batch_pair_count = 0;

  hipStreamCreate(&gpu->stream);

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

  const hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    error("runner_gpu_init failed: %s", hipGetErrorString(err));
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

  hipStreamDestroy(gpu->stream);

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
  gpu->stream = NULL;
  gpu->grav_batch_self_count = 0;
  gpu->grav_batch_pair_count = 0;
  gpu->grav_batch_ncells = 0;
  gpu->grav_max_cell_size = 0;
}

/**
 * @brief Complete a self-gravity task at task level.
 *
 * @param r The runner completing the task.
 * @param sched The scheduler owning the task.
 * @param t The task to complete.
 */
struct task* runner_gpu_complete_self_task(struct runner* r,
                                           struct scheduler* sched,
                                           struct task* t) {

  lock_lock(&sched->queues[r->qid].lock);
  sched->queues[r->qid].gpu_self_tasks_left--;
  (void)lock_unlock(&sched->queues[r->qid].lock);

  return scheduler_done(sched, t);
}

/**
 * @brief Complete a pair-gravity task at task level.
 *
 * @param r The runner completing the task.
 * @param sched The scheduler owning the task.
 * @param t The task to complete.
 */
struct task* runner_gpu_complete_pair_task(struct runner* r,
                                           struct scheduler* sched,
                                           struct task* t) {

  lock_lock(&sched->queues[r->qid].lock);
  sched->queues[r->qid].gpu_pair_tasks_left--;
  (void)lock_unlock(&sched->queues[r->qid].lock);

  return scheduler_done(sched, t);
}

/**
 * @brief Complete all tasks in the current self-gravity GPU batch.
 *
 * @param r The runner owning the batch.
 * @param sched The scheduler owning the tasks.
 */
void runner_gpu_complete_self_batch(struct runner* r, struct scheduler* sched) {

  const int count = r->gpu.grav_batch_self_count;

  for (int i = 0; i < count; i++) {
    runner_gpu_complete_self_task(r, sched, r->gpu.grav_tasks_self[i]);
    r->gpu.grav_cells_self[i] = NULL;
    r->gpu.grav_tasks_self[i] = NULL;
  }

  r->gpu.grav_batch_self_count = 0;
}

/**
 * @brief Complete all unique tasks in the current pair-gravity GPU batch.
 *
 * @param r The runner owning the batch.
 * @param sched The scheduler owning the tasks.
 */
void runner_gpu_complete_pair_batch(struct runner* r, struct scheduler* sched) {

  const int count = r->gpu.grav_batch_pair_count;
  struct task* prev_task = NULL;

  for (int i = 0; i < count; i += 2) {
    if (r->gpu.grav_tasks_pair[i / 2] != prev_task) {
      runner_gpu_complete_pair_task(r, sched, r->gpu.grav_tasks_pair[i / 2]);
      prev_task = r->gpu.grav_tasks_pair[i / 2];
    }

    r->gpu.grav_cells_pair[i] = NULL;
    r->gpu.grav_cells_pair[i + 1] = NULL;
    r->gpu.grav_tasks_pair[i / 2] = NULL;
  }

  r->gpu.grav_batch_pair_count = 0;
}

/**
 * @brief Flush any leftover packed self-gravity work owned by a runner.
 *
 * @param r The runner whose GPU batch should be flushed.
 * @param sched The scheduler owning the queued tasks.
 */
int runner_gpu_flush_leftover_self(struct runner* r) {

  const int ncells_flush_self = r->gpu.grav_batch_self_count;
  const int max_cell_size = r->gpu.grav_max_cell_size;

  if (ncells_flush_self == 0) return 0;

  {
    TIMER_TIC;

    hipMemcpyAsync(r->gpu.gravity_gpu_values_send_self_d,
                   r->gpu.gravity_gpu_values_send_self,
                   ncells_flush_self * max_cell_size *
                       sizeof(struct gravity_gpu_values_send),
                   hipMemcpyHostToDevice, r->gpu.stream);
    hipMemcpyAsync(r->gpu.gravity_gpu_values_recv_self_d,
                   r->gpu.gravity_gpu_values_recv_self,
                   ncells_flush_self * max_cell_size *
                       sizeof(struct gravity_gpu_values_recv),
                   hipMemcpyHostToDevice, r->gpu.stream);

    hipError_t err4 = hipGetLastError();
    if (err4 != hipSuccess) printf("Error4: %s\n", hipGetErrorString(err4));

    runner_doself_recursive_grav_new(
        r, r->gpu.grav_cells_self[0], 1, r->gpu.gravity_gpu_values_send_self_d,
        r->gpu.gravity_gpu_values_recv_self_d, ncells_flush_self, max_cell_size,
        r->gpu.stream);

    hipMemcpyAsync(r->gpu.gravity_gpu_values_recv_self,
                   r->gpu.gravity_gpu_values_recv_self_d,
                   ncells_flush_self * max_cell_size *
                       sizeof(struct gravity_gpu_values_recv),
                   hipMemcpyDeviceToHost, r->gpu.stream);

    hipStreamSynchronize(r->gpu.stream);

    TIMER_TOC(timer_doself_grav_pp);
  }

  hipError_t err5 = hipGetLastError();
  if (err5 != hipSuccess) printf("Error5: %s\n", hipGetErrorString(err5));

  {
    TIMER_TIC;

    for (int j = 0; j < ncells_flush_self; j++) {
      while (cell_glocktree(r->gpu.grav_cells_self[j])) {
        ;
      }
      for (int i = 0;
           i < r->gpu.gravity_gpu_values_send_self[j * max_cell_size].gcounts;
           i++) {
        r->gpu.grav_cells_self[j]->grav.parts[i].a_grav[0] +=
            r->gpu.gravity_gpu_values_recv_self[i + j * max_cell_size].a_x_i;
        r->gpu.grav_cells_self[j]->grav.parts[i].a_grav[1] +=
            r->gpu.gravity_gpu_values_recv_self[i + j * max_cell_size].a_y_i;
        r->gpu.grav_cells_self[j]->grav.parts[i].a_grav[2] +=
            r->gpu.gravity_gpu_values_recv_self[i + j * max_cell_size].a_z_i;
        r->gpu.grav_cells_self[j]->grav.parts[i].potential +=
            r->gpu.gravity_gpu_values_recv_self[i + j * max_cell_size].pot_i;
      }
      cell_gunlocktree(r->gpu.grav_cells_self[j]);
    }

    TIMER_TOC(timer_doself_grav_pp);
  }

  return 1;
}

/**
 * @brief Flush any leftover packed pair-gravity work owned by a runner.
 *
 * @param r The runner whose GPU batch should be flushed.
 * @param sched The scheduler owning the queued tasks.
 */
int runner_gpu_flush_leftover_pair(struct runner* r) {

  const int ncells_flush_pair = r->gpu.grav_batch_pair_count;
  const int max_cell_size = r->gpu.grav_max_cell_size;

  if (ncells_flush_pair == 0) return 0;

  {
    TIMER_TIC;

    hipMemcpyAsync(r->gpu.gravity_gpu_values_send_pair_d,
                   r->gpu.gravity_gpu_values_send_pair,
                   ncells_flush_pair * max_cell_size *
                       sizeof(struct gravity_gpu_values_send),
                   hipMemcpyHostToDevice, r->gpu.stream);
    hipMemcpyAsync(r->gpu.gravity_gpu_values_recv_pair_d,
                   r->gpu.gravity_gpu_values_recv_pair,
                   ncells_flush_pair * max_cell_size *
                       sizeof(struct gravity_gpu_values_recv),
                   hipMemcpyHostToDevice, r->gpu.stream);

    hipError_t err4 = hipGetLastError();
    if (err4 != hipSuccess) printf("Error4: %s\n", hipGetErrorString(err4));

    struct cell* ci_flush = r->gpu.grav_cells_pair[0];
    struct cell* cj_flush = r->gpu.grav_cells_pair[1];

    if (ci_flush == NULL || cj_flush == NULL)
      error("pair flush: NULL packed cells");

    const struct engine* e = r->e;
    const int periodic = e->mesh->periodic;
    const float dim[3] = {(float)e->mesh->dim[0], (float)e->mesh->dim[1],
                          (float)e->mesh->dim[2]};
    const float r_s_inv = e->mesh->r_s_inv;
    const double min_trunc = e->mesh->r_cut_min;

    const int ci_active =
        cell_is_active_gravity(ci_flush, e) && (ci_flush->nodeID == e->nodeID);
    const int cj_active =
        cell_is_active_gravity(cj_flush, e) && (cj_flush->nodeID == e->nodeID);
    const float rmax_i = ci_flush->grav.multipole->r_max;
    const float rmax_j = cj_flush->grav.multipole->r_max;
    const int gcount_i = ci_flush->grav.count;
    const int gcount_j = cj_flush->grav.count;
    const int gcount_padded_i = gcount_i - (gcount_i % VEC_SIZE) + VEC_SIZE;
    const int gcount_padded_j = gcount_j - (gcount_j % VEC_SIZE) + VEC_SIZE;

    pair_pp_offload_new(periodic, rmax_i, rmax_j, min_trunc, &r_s_inv,
                        &gcount_i, &gcount_padded_i, &gcount_j,
                        &gcount_padded_j, ci_active, cj_active, dim[0], dim[1],
                        dim[2], 1, r->gpu.gravity_gpu_values_send_pair_d,
                        r->gpu.gravity_gpu_values_recv_pair_d,
                        ncells_flush_pair, max_cell_size, r->gpu.stream);

    hipMemcpyAsync(r->gpu.gravity_gpu_values_recv_pair,
                   r->gpu.gravity_gpu_values_recv_pair_d,
                   ncells_flush_pair * max_cell_size *
                       sizeof(struct gravity_gpu_values_recv),
                   hipMemcpyDeviceToHost, r->gpu.stream);

    hipStreamSynchronize(r->gpu.stream);

    TIMER_TOC(timer_doself_grav_pp);
  }

  hipError_t err5 = hipGetLastError();
  if (err5 != hipSuccess) printf("Error5: %s\n", hipGetErrorString(err5));

  {
    TIMER_TIC;

    for (int j = 0; j < ncells_flush_pair; j += 2) {
      if (r->gpu.grav_cells_pair[j] == NULL ||
          r->gpu.grav_cells_pair[j + 1] == NULL)
        error("PAIR UNPACK: NULL cell j=%d packed=%d qid=%d", j,
              ncells_flush_pair, r->qid);

      if (r->gpu.grav_tasks_pair[j / 2] == NULL)
        error("PAIR UNPACK: NULL task k=%d (j=%d) packed=%d qid=%d", j / 2, j,
              ncells_flush_pair, r->qid);

      struct cell* ci0 = r->gpu.grav_cells_pair[j];
      struct cell* cj0 = r->gpu.grav_cells_pair[j + 1];
      struct cell *a = ci0, *b = cj0;

      if (a > b) {
        struct cell* tmp = a;
        a = b;
        b = tmp;
      }

      while (cell_glocktree(a)) {
        ;
      }
      for (int i = 0;
           i < r->gpu.gravity_gpu_values_send_pair[j * max_cell_size].gcounts;
           i++) {
        ci0->grav.parts[i].a_grav[0] +=
            r->gpu.gravity_gpu_values_recv_pair[i + j * max_cell_size].a_x_i;
        ci0->grav.parts[i].a_grav[1] +=
            r->gpu.gravity_gpu_values_recv_pair[i + j * max_cell_size].a_y_i;
        ci0->grav.parts[i].a_grav[2] +=
            r->gpu.gravity_gpu_values_recv_pair[i + j * max_cell_size].a_z_i;
        ci0->grav.parts[i].potential +=
            r->gpu.gravity_gpu_values_recv_pair[i + j * max_cell_size].pot_i;
      }
      cell_gunlocktree(a);

      while (cell_glocktree(b)) {
        ;
      }
      for (int i = 0;
           i <
           r->gpu.gravity_gpu_values_send_pair[(j + 1) * max_cell_size].gcounts;
           i++) {
        cj0->grav.parts[i].a_grav[0] +=
            r->gpu.gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size]
                .a_x_i;
        cj0->grav.parts[i].a_grav[1] +=
            r->gpu.gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size]
                .a_y_i;
        cj0->grav.parts[i].a_grav[2] +=
            r->gpu.gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size]
                .a_z_i;
        cj0->grav.parts[i].potential +=
            r->gpu.gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size]
                .pot_i;
      }
      cell_gunlocktree(b);
    }

    TIMER_TOC(timer_doself_grav_pp);
  }

  return 1;
}
