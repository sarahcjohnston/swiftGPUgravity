/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2026 Sarah Johnston (sarah.c.johnston@durham.ac.uk)
 *                    Will Roper (w.roper@sussex.ac.uk)
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

static void runner_gpu_complete_self_task(struct runner* r,
                                          struct scheduler* sched,
                                          struct task* t) {
  lock_lock(&sched->queues[r->qid].lock);
  sched->queues[r->qid].gpu_self_tasks_left--;
  (void)lock_unlock(&sched->queues[r->qid].lock);
  scheduler_done(sched, t);
}

static void runner_gpu_complete_pair_task(struct runner* r,
                                          struct scheduler* sched,
                                          struct task* t) {
  lock_lock(&sched->queues[r->qid].lock);
  sched->queues[r->qid].gpu_pair_tasks_left--;
  (void)lock_unlock(&sched->queues[r->qid].lock);
  scheduler_done(sched, t);
}

void runner_gpu_complete_self_batch(struct runner* r, struct scheduler* sched) {
  const int count = r->gpu.grav_batch_self_count;

  for (int i = 0; i < count; i++) {
    runner_gpu_complete_self_task(r, sched, r->gpu.grav_tasks_self[i]);
    r->gpu.grav_cells_self[i] = NULL;
    r->gpu.grav_tasks_self[i] = NULL;
  }

  r->gpu.grav_batch_self_count = 0;
}

void runner_gpu_complete_pair_batch(struct runner* r, struct scheduler* sched) {
  const int count = r->gpu.grav_batch_pair_count;
  struct task* prev_task = NULL;

  for (int i = 0; i < count; i += 2) {
    struct task* task = r->gpu.grav_tasks_pair[i / 2];
    if (task != prev_task) {
      runner_gpu_complete_pair_task(r, sched, task);
      prev_task = task;
    }
    r->gpu.grav_cells_pair[i] = NULL;
    r->gpu.grav_cells_pair[i + 1] = NULL;
    r->gpu.grav_tasks_pair[i / 2] = NULL;
  }

  r->gpu.grav_batch_pair_count = 0;
}

/**
 * @brief Computes the interaction of all the particles in a cell with all the
 * particles of another cell.
 *
 * This function switches between the full potential and the truncated one
 * depending on needs. It will also use the M2P (multipole) interaction
 * for the subset of particles in either cell for which the distance criterion
 * is valid.
 *
 * This function starts by constructing the require #gravity_cache for both
 * cells and then call the specialised functions doing the actual work on
 * the caches. It then write the data back to the particles.
 *
 * @param r The #runner.
 * @param ci The first #cell.
 * @param cj The other #cell.
 * @param symmetric Are we updating both cells (1) or just ci (0) ?
 * @param allow_mpole Are we allowing the use of M2P interactions ?
 */
enum runner_gpu_task_type runner_dopair_grav_pp_new(
    struct runner* r, struct cell* ci, struct cell* cj, const int symmetric,
    const int allow_mpole,
    struct gravity_gpu_values_send* gravity_gpu_values_send_pair,
    struct gravity_gpu_values_send* gravity_gpu_values_send_pair_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_pair,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_pair_d,
    struct cell** grav_cells_pair, struct task** grav_tasks_pair,
    struct task* t, int ncells, int max_cell_size, hipStream_t stream) {

  /* Recover some useful constants */
  const struct engine* e = r->e;
  const int periodic = e->mesh->periodic;
  const float dim[3] = {(float)e->mesh->dim[0], (float)e->mesh->dim[1],
                        (float)e->mesh->dim[2]};
  const float r_s_inv = e->mesh->r_s_inv;
  const double min_trunc = e->mesh->r_cut_min;

  float dim_0 = dim[0];
  float dim_1 = dim[1];
  float dim_2 = dim[2];

  TIMER_TIC;

  /* Record activity status */
  const int ci_active =
      cell_is_active_gravity(ci, e) && (ci->nodeID == e->nodeID);
  const int cj_active =
      cell_is_active_gravity(cj, e) && (cj->nodeID == e->nodeID);

#ifdef SWIFT_DEBUG_CHECKS
  /* Check that we are not doing something stupid */
  if (ci->split || cj->split) error("Running P-P on splitable cells");

  /* Let's start by checking things are drifted */
  if (!cell_are_gpart_drifted(ci, e)) error("Un-drifted gparts");
  if (!cell_are_gpart_drifted(cj, e)) error("Un-drifted gparts");
  if (cj_active && ci->grav.ti_old_multipole != e->ti_current)
    error("Un-drifted multipole");
  if (ci_active && cj->grav.ti_old_multipole != e->ti_current)
    error("Un-drifted multipole");
#endif

  /* Caches to play with */
  struct gravity_cache* const ci_cache = &r->ci_gravity_cache;
  struct gravity_cache* const cj_cache = &r->cj_gravity_cache;

  /* Shift to apply to the particles in each cell */
  const double shift_i[3] = {0., 0., 0.};
  const double shift_j[3] = {0., 0., 0.};

  /* Recover the multipole info and shift the CoM locations */
  const float rmax_i = ci->grav.multipole->r_max;
  const float rmax_j = cj->grav.multipole->r_max;
  const float CoM_i[3] = {(float)(ci->grav.multipole->CoM[0] - shift_i[0]),
                          (float)(ci->grav.multipole->CoM[1] - shift_i[1]),
                          (float)(ci->grav.multipole->CoM[2] - shift_i[2])};
  const float CoM_j[3] = {(float)(cj->grav.multipole->CoM[0] - shift_j[0]),
                          (float)(cj->grav.multipole->CoM[1] - shift_j[1]),
                          (float)(cj->grav.multipole->CoM[2] - shift_j[2])};

  /* Start by constructing particle caches */

  /* Computed the padded counts */
  const int gcount_i = ci->grav.count;
  const int gcount_j = cj->grav.count;
  const int gcount_padded_i = gcount_i - (gcount_i % VEC_SIZE) + VEC_SIZE;
  const int gcount_padded_j = gcount_j - (gcount_j % VEC_SIZE) + VEC_SIZE;
  const int allow_multipole_i = allow_mpole && ci->grav.count > 1;
  const int allow_multipole_j = allow_mpole && cj->grav.count > 1;

  /* Fill the caches */
  if (ci->nodeID == e->nodeID) {
    gravity_cache_populate(e->max_active_bin, allow_multipole_j, periodic, dim,
                           ci_cache, ci->grav.parts, gcount_i, gcount_padded_i,
                           shift_i, CoM_j, cj->grav.multipole, ci,
                           e->gravity_properties);
  } else {
    gravity_cache_populate_foreign(
        periodic, dim, ci_cache, ci->grav.parts_foreign, gcount_i,
        gcount_padded_i, shift_i, ci, e->gravity_properties);
  }

  if (cj->nodeID == e->nodeID) {
    gravity_cache_populate(e->max_active_bin, allow_multipole_i, periodic, dim,
                           cj_cache, cj->grav.parts, gcount_j, gcount_padded_j,
                           shift_j, CoM_i, ci->grav.multipole, cj,
                           e->gravity_properties);
  } else {
    gravity_cache_populate_foreign(
        periodic, dim, cj_cache, cj->grav.parts_foreign, gcount_j,
        gcount_padded_j, shift_j, cj, e->gravity_properties);
  }

  struct cell* ci0 = ci;
  struct cell* cj0 = cj;
  struct cell *a = ci0, *b = cj0;

  if (a > b) {
    struct cell* tmp = a;
    a = b;
    b = tmp;
  }
  while (cell_glocktree(a)) {
    ;
  }
  while (cell_glocktree(b)) {
    ;
  }

  hipEvent_t startpack, stoppack;
  hipEventCreate(&startpack);
  hipEventCreate(&stoppack);

  hipEventRecord(startpack, stream);

  {
    TIMER_TIC;
    for (int i = 0; i < gcount_i; i++) {
      gravity_gpu_values_send_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .h_i = ci_cache->epsilon[i];
      gravity_gpu_values_send_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .mass_i = ci_cache->m[i];
      gravity_gpu_values_send_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .x_i = ci_cache->x[i];
      gravity_gpu_values_send_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .y_i = ci_cache->y[i];
      gravity_gpu_values_send_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .z_i = ci_cache->z[i];
      gravity_gpu_values_send_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .active_i = ci_cache->active[i];

      gravity_gpu_values_send_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .h_j = ci_cache->epsilon[i];
      gravity_gpu_values_send_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .mass_j = ci_cache->m[i];
      gravity_gpu_values_send_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .x_j = ci_cache->x[i];
      gravity_gpu_values_send_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .y_j = ci_cache->y[i];
      gravity_gpu_values_send_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .z_j = ci_cache->z[i];
      gravity_gpu_values_send_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .active_j = ci_cache->active[i];
    }

    for (int i = 0; i < gcount_j; i++) {
      gravity_gpu_values_send_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .h_j = cj_cache->epsilon[i];
      gravity_gpu_values_send_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .mass_j = cj_cache->m[i];
      gravity_gpu_values_send_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .x_j = cj_cache->x[i];
      gravity_gpu_values_send_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .y_j = cj_cache->y[i];
      gravity_gpu_values_send_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .z_j = cj_cache->z[i];
      gravity_gpu_values_send_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .active_j = cj_cache->active[i];

      gravity_gpu_values_send_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .h_i = cj_cache->epsilon[i];
      gravity_gpu_values_send_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .mass_i = cj_cache->m[i];
      gravity_gpu_values_send_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .x_i = cj_cache->x[i];
      gravity_gpu_values_send_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .y_i = cj_cache->y[i];
      gravity_gpu_values_send_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .z_i = cj_cache->z[i];
      gravity_gpu_values_send_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .active_i = cj_cache->active[i];
    }

    for (int i = 0; i < max_cell_size; i++) {
      gravity_gpu_values_recv_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .a_x_i = 0;
      gravity_gpu_values_recv_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .a_y_i = 0;
      gravity_gpu_values_recv_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .a_z_i = 0;
      gravity_gpu_values_recv_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .pot_i = 0;
      gravity_gpu_values_recv_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .a_x_j = 0;
      gravity_gpu_values_recv_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .a_y_j = 0;
      gravity_gpu_values_recv_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .a_z_j = 0;
      gravity_gpu_values_recv_pair[i +
                                   r->gpu.grav_batch_pair_count * max_cell_size]
          .pot_j = 0;
    }

    for (int i = 0; i < max_cell_size; i++) {
      gravity_gpu_values_recv_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .a_x_i = 0;
      gravity_gpu_values_recv_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .a_y_i = 0;
      gravity_gpu_values_recv_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .a_z_i = 0;
      gravity_gpu_values_recv_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .pot_i = 0;
      gravity_gpu_values_recv_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .a_x_j = 0;
      gravity_gpu_values_recv_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .a_y_j = 0;
      gravity_gpu_values_recv_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .a_z_j = 0;
      gravity_gpu_values_recv_pair[i + (r->gpu.grav_batch_pair_count + 1) *
                                           max_cell_size]
          .pot_j = 0;
    }
    TIMER_TOC(timer_doself_grav_pp);
  }

  /* Store the address of the cells and tasks we are working on */
  grav_cells_pair[r->gpu.grav_batch_pair_count] = ci;
  grav_cells_pair[r->gpu.grav_batch_pair_count + 1] = cj;
  grav_tasks_pair[r->gpu.grav_batch_pair_count / 2] = t;

  gravity_gpu_values_send_pair[r->gpu.grav_batch_pair_count * max_cell_size]
      .cell_active = cell_is_active_gravity(ci, e);
  gravity_gpu_values_send_pair[(r->gpu.grav_batch_pair_count + 1) *
                               max_cell_size]
      .cell_active = cell_is_active_gravity(cj, e);

  gravity_gpu_values_send_pair[r->gpu.grav_batch_pair_count * max_cell_size]
      .gcounts = gcount_i;
  gravity_gpu_values_send_pair[(r->gpu.grav_batch_pair_count + 1) *
                               max_cell_size]
      .gcounts = gcount_j;

  int use_full = 1;
  if (periodic) {
    double d0 = CoM_j[0] - CoM_i[0];
    double d1 = CoM_j[1] - CoM_i[1];
    double d2 = CoM_j[2] - CoM_i[2];
    d0 = nearest(d0, e->mesh->dim[0]);
    d1 = nearest(d1, e->mesh->dim[1]);
    d2 = nearest(d2, e->mesh->dim[2]);
    double r2 = d0 * d0 + d1 * d1 + d2 * d2;
    double max_r = sqrt(r2) + rmax_i + rmax_j;
    use_full = (max_r <= min_trunc);
  }

  // store decision on BOTH blocks
  gravity_gpu_values_send_pair[r->gpu.grav_batch_pair_count * max_cell_size]
      .use_full = use_full;
  gravity_gpu_values_send_pair[(r->gpu.grav_batch_pair_count + 1) *
                               max_cell_size]
      .use_full = use_full;

  // update that we packed a cell into our array
  r->gpu.grav_batch_pair_count += 2;

  gravity_cache_zero_output(ci_cache, gcount_padded_i);
  gravity_cache_zero_output(cj_cache, gcount_padded_j);

  cell_gunlocktree(b);
  cell_gunlocktree(a);

  /* If we have filled our batch, flush it and reset the count. */
  if (r->gpu.grav_batch_pair_count >= ncells) {
    hipEvent_t startcopyH2D, stopcopyH2D;
    hipEventCreate(&startcopyH2D);
    hipEventCreate(&stopcopyH2D);

    hipEventRecord(startcopyH2D, stream);

    {
      TIMER_TIC;

      /* Now copy all the arrays to the device */
      hipMemcpyAsync(
          gravity_gpu_values_send_pair_d, gravity_gpu_values_send_pair,
          ncells * max_cell_size * sizeof(struct gravity_gpu_values_send),
          hipMemcpyHostToDevice, stream);
      hipMemcpyAsync(
          gravity_gpu_values_recv_pair_d, gravity_gpu_values_recv_pair,
          ncells * max_cell_size * sizeof(struct gravity_gpu_values_recv),
          hipMemcpyHostToDevice, stream);

      hipEventRecord(stopcopyH2D, stream);

      hipError_t err2 = hipGetLastError();
      if (err2 != hipSuccess) printf("Error2: %s\n", hipGetErrorString(err2));

      hipEvent_t startker, stopker;
      hipEventCreate(&startker);
      hipEventCreate(&stopker);

      hipEventRecord(startker, stream);

      // run the GPU function
      pair_pp_offload_new(
          periodic, rmax_i, rmax_j, min_trunc, &r_s_inv, &gcount_i,
          &gcount_padded_i, &gcount_j, &gcount_padded_j, ci_active, cj_active,
          dim_0, dim_1, dim_2, symmetric, gravity_gpu_values_send_pair_d,
          gravity_gpu_values_recv_pair_d, ncells, max_cell_size, stream);

      hipEventRecord(stopker, stream);

      // hipDeviceSynchronize();

      hipEvent_t startcopyD2H, stopcopyD2H;
      hipEventCreate(&startcopyD2H);
      hipEventCreate(&stopcopyD2H);

      hipEventRecord(startcopyD2H, stream);

      // copy the arrays from device to host
      // gravity_gpu_D2H(gravity_gpu_values_h, gravity_gpu_values_d, ncells,
      // max_cell_size, stream);
      hipMemcpyAsync(
          gravity_gpu_values_recv_pair, gravity_gpu_values_recv_pair_d,
          ncells * max_cell_size * sizeof(struct gravity_gpu_values_recv),
          hipMemcpyDeviceToHost, stream);

      hipEventRecord(stopcopyD2H, stream);

      hipStreamSynchronize(stream);  // THIS ONE IS NEEDED!

      TIMER_TOC(timer_doself_grav_pp);
    }  // TIMER_TOC(timer_gpu_copycalc);

    // TIMINGS RECORDING
    /*printf("Pack Time: %f ms\n", timer_gpu_pack);
    FILE *f1 = fopen("packtime_a30.txt", "a");
    fprintf(f1, "%f\n", timer_gpu_pack);
    fclose(f1);

    float copytimeH2D = 0;
    hipEventElapsedTime(&copytimeH2D, startcopyH2D, stopcopyH2D);
    printf("Copy Time: %f ms\n", copytimeH2D);
    FILE *f2 = fopen("copytimeH2D_a30.txt", "a");
    fprintf(f2, "%f\n", copytimeH2D);
    fclose(f2);

    float kerneltime = 0;
    hipEventElapsedTime(&kerneltime, startker, stopker);
    printf("Kernel Time: %f ms\n", kerneltime);
    FILE *f3 = fopen("kerneltime_a30.txt", "a");
    fprintf(f3, "%f\n", kerneltime);
    fclose(f3);

    float copytimeD2H = 0;
    hipEventElapsedTime(&copytimeD2H, startcopyD2H, stopcopyD2H);
    printf("Copy Time: %f ms\n", copytimeD2H);
    FILE *f4 = fopen("copytimeD2H_a30.txt", "a");
    fprintf(f4, "%f\n", copytimeD2H);
    fclose(f4);*/

    // hipDeviceSynchronize();
    hipError_t err3 = hipGetLastError();
    if (err3 != hipSuccess) printf("Error3: %s\n", hipGetErrorString(err3));

    {
      TIMER_TIC;

      /*send results back to relevant cell structs*/
      for (int j = 0; j < ncells; j += 2) {

        if (grav_cells_pair[j] == NULL || grav_cells_pair[j + 1] == NULL)
          error("PAIR UNPACK: NULL cell j=%d packed=%d qid=%d", j, ncells,
                r->qid);

        if (grav_tasks_pair[j / 2] == NULL)
          error("PAIR UNPACK: NULL task k=%d (j=%d) packed=%d qid=%d", j / 2, j,
                ncells, r->qid);

        struct cell* ci_pair = grav_cells_pair[j];
        struct cell* cj_pair = grav_cells_pair[j + 1];
        struct cell *a_pair = ci_pair, *b_pair = cj_pair;

        if (a_pair > b_pair) {
          struct cell* tmp = a_pair;
          a_pair = b_pair;
          b_pair = tmp;
        }

        while (cell_glocktree(a_pair)) {
          ;
        }
        // printf("hunting for lock for cell %p\n", (void*)a); }
        for (int i = 0;
             i < gravity_gpu_values_send_pair[j * max_cell_size].gcounts; i++) {
          ci_pair->grav.parts[i].a_grav[0] +=
              gravity_gpu_values_recv_pair[i + j * max_cell_size].a_x_i;
          ci_pair->grav.parts[i].a_grav[1] +=
              gravity_gpu_values_recv_pair[i + j * max_cell_size].a_y_i;
          ci_pair->grav.parts[i].a_grav[2] +=
              gravity_gpu_values_recv_pair[i + j * max_cell_size].a_z_i;
          ci_pair->grav.parts[i].potential +=
              gravity_gpu_values_recv_pair[i + j * max_cell_size].pot_i;
        }
        cell_gunlocktree(a_pair);

        while (cell_glocktree(b_pair)) {
          ;
        }  // {printf("hunting for lock for cell %p\n", (void*)b);}
        for (int i = 0;
             i < gravity_gpu_values_send_pair[(j + 1) * max_cell_size].gcounts;
             i++) {
          cj_pair->grav.parts[i].a_grav[0] +=
              gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size].a_x_i;
          cj_pair->grav.parts[i].a_grav[1] +=
              gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size].a_y_i;
          cj_pair->grav.parts[i].a_grav[2] +=
              gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size].a_z_i;
          cj_pair->grav.parts[i].potential +=
              gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size].pot_i;
        }
        cell_gunlocktree(b_pair);
      }

      TIMER_TOC(timer_doself_grav_pp);
    }

    return flushed_pair_task;
  }

  return packed_task;
}

/**
 * @brief Pack, launch, and unpack a batched self-gravity GPU task.
 *
 * @param r The #runner.
 * @param ci The #cell to pack.
 * @param t The #task being executed.
 * @param ncells The batch capacity in cells.
 * @param max_cell_size The maximum number of particles per packed cell.
 * @return The outcome of the GPU wrapper for this task.
 */
enum runner_gpu_task_type runner_doself_grav_pp_task_new(struct runner* r,
                                                         struct cell* ci,
                                                         struct task* t,
                                                         int ncells,
                                                         int max_cell_size) {

  const struct engine* e = r->e;
  struct gravity_cache* const ci_cache = &r->ci_gravity_cache;
  const int gcount = ci->grav.count;
  const int gcount_padded = gcount - (gcount % VEC_SIZE) + VEC_SIZE;

  if (gcount > max_cell_size)
    error(
        "More particles than allocated memory! %i particles in cell and only "
        "%i slots in memory available. Increase the number of top level "
        "cells!",
        gcount, max_cell_size);

  const double loc[3] = {ci->loc[0] + 0.5 * ci->width[0],
                         ci->loc[1] + 0.5 * ci->width[1],
                         ci->loc[2] + 0.5 * ci->width[2]};

  gravity_cache_populate_no_mpole(e->max_active_bin, ci_cache, ci->grav.parts,
                                  gcount, gcount_padded, loc, ci,
                                  e->gravity_properties);

  while (cell_glocktree(ci)) {
    ;
  }

  hipEvent_t startpack, stoppack;
  hipEventCreate(&startpack);
  hipEventCreate(&stoppack);
  hipEventRecord(startpack, r->gpu.stream);

  {
    TIMER_TIC;
    for (int i = 0; i < gcount; i++) {
      r->gpu
          .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .h_i = ci_cache->epsilon[i];
      r->gpu
          .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .h_j = ci_cache->epsilon[i];
      r->gpu
          .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .mass_i = ci_cache->m[i];
      r->gpu
          .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .mass_j = ci_cache->m[i];
      r->gpu
          .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .x_i = ci_cache->x[i];
      r->gpu
          .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .x_j = ci_cache->x[i];
      r->gpu
          .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .y_i = ci_cache->y[i];
      r->gpu
          .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .y_j = ci_cache->y[i];
      r->gpu
          .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .z_i = ci_cache->z[i];
      r->gpu
          .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .z_j = ci_cache->z[i];
      r->gpu
          .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .active_i = ci_cache->active[i];
      r->gpu
          .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .active_j = ci_cache->active[i];
    }

    for (int i = 0; i < max_cell_size; i++) {
      r->gpu
          .gravity_gpu_values_recv_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .a_x_i = 0;
      r->gpu
          .gravity_gpu_values_recv_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .a_y_i = 0;
      r->gpu
          .gravity_gpu_values_recv_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .a_z_i = 0;
      r->gpu
          .gravity_gpu_values_recv_self[i + r->gpu.grav_batch_self_count *
                                                max_cell_size]
          .pot_i = 0;
    }

    TIMER_TOC(timer_doself_grav_pp);
  }

  r->gpu.grav_cells_self[r->gpu.grav_batch_self_count] = ci;
  r->gpu.grav_tasks_self[r->gpu.grav_batch_self_count] = t;

  for (int i = 0; i < gcount; i++) {
    r->gpu
        .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                              max_cell_size]
        .cell_active = cell_is_active_gravity(ci, e);
    r->gpu
        .gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                              max_cell_size]
        .gcounts = gcount;
  }

  r->gpu.grav_batch_self_count += 1;

  gravity_cache_zero_output(ci_cache, gcount_padded);
  cell_gunlocktree(ci);

#ifdef SWIFT_DEBUG_CHECKS
  for (int j = 0; j < gcount; j++) {
    for (int i = 0; i < gcount; i++) {
      if (i == j) continue;
      accumulate_inc_ll(&ci->grav.parts[j].num_interacted);
    }
  }
#endif

  if (r->gpu.grav_batch_self_count >= ncells) {
    hipEvent_t startcopyH2D, stopcopyH2D;
    hipEventCreate(&startcopyH2D);
    hipEventCreate(&stopcopyH2D);
    hipEventRecord(startcopyH2D, r->gpu.stream);

    {
      TIMER_TIC;

      hipMemcpyAsync(
          r->gpu.gravity_gpu_values_send_self_d,
          r->gpu.gravity_gpu_values_send_self,
          ncells * max_cell_size * sizeof(struct gravity_gpu_values_send),
          hipMemcpyHostToDevice, r->gpu.stream);
      hipMemcpyAsync(
          r->gpu.gravity_gpu_values_recv_self_d,
          r->gpu.gravity_gpu_values_recv_self,
          ncells * max_cell_size * sizeof(struct gravity_gpu_values_recv),
          hipMemcpyHostToDevice, r->gpu.stream);

      hipEventRecord(stopcopyH2D, r->gpu.stream);

      hipError_t err2 = hipGetLastError();
      if (err2 != hipSuccess) printf("Error2: %s\n", hipGetErrorString(err2));

      hipEvent_t startker, stopker;
      hipEventCreate(&startker);
      hipEventCreate(&stopker);
      hipEventRecord(startker, r->gpu.stream);

      runner_doself_recursive_grav_new(r, ci, 1,
                                       r->gpu.gravity_gpu_values_send_self_d,
                                       r->gpu.gravity_gpu_values_recv_self_d,
                                       ncells, max_cell_size, r->gpu.stream);

      hipEventRecord(stopker, r->gpu.stream);

      hipEvent_t startcopyD2H, stopcopyD2H;
      hipEventCreate(&startcopyD2H);
      hipEventCreate(&stopcopyD2H);
      hipEventRecord(startcopyD2H, r->gpu.stream);

      hipMemcpyAsync(
          r->gpu.gravity_gpu_values_recv_self,
          r->gpu.gravity_gpu_values_recv_self_d,
          ncells * max_cell_size * sizeof(struct gravity_gpu_values_recv),
          hipMemcpyDeviceToHost, r->gpu.stream);

      hipEventRecord(stopcopyD2H, r->gpu.stream);
      hipStreamSynchronize(r->gpu.stream);

      TIMER_TOC(timer_doself_grav_pp);
    }

    hipError_t err3 = hipGetLastError();
    if (err3 != hipSuccess) printf("Error3: %s\n", hipGetErrorString(err3));

    {
      TIMER_TIC;

      for (int j = 0; j < ncells; j++) {
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

    return flushed_self_task;
  }

  return packed_task;
}

/**
 * @brief Computes the interaction of all the particles in a cell with all the
 * particles of another cell.
 *
 * This function will try to recurse as far down the tree as possible and only
 * default to direct summation if there is no better option.
 *
 * If using periodic BCs, we will abort the recursion if th distance between the
 * cells is larger than the set threshold.
 *
 * @param r The #runner.
 * @param ci The first #cell.
 * @param cj The other #cell.
 * @param gettimer Are we timing this ?
 */
enum runner_gpu_task_type runner_dopair_recursive_grav_new(
    struct runner* r, struct cell* ci, struct cell* cj, const int gettimer,
    struct gravity_gpu_values_send* gravity_gpu_values_send_pair,
    struct gravity_gpu_values_send* gravity_gpu_values_send_pair_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_pair,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_pair_d,
    struct cell** grav_cells_pair, struct task** grav_tasks_pair,
    struct task* t, int ncells, int max_cell_size, hipStream_t stream) {

  if (ci == NULL || cj == NULL)
    error("runner_dopair_recursive_grav_new got NULL cell");

  const struct engine* e = r->e;

  if (!cell_are_gpart_drifted(ci, e))
    cell_drift_gpart(ci, e, /*force=*/1, NULL);
  if (!cell_are_gpart_drifted(cj, e))
    cell_drift_gpart(cj, e, /*force=*/1, NULL);

  /* Clear the flags */
  runner_clear_grav_flags(ci, e);
  runner_clear_grav_flags(cj, e);

  /* Some constants */
  const int nodeID = e->nodeID;
  const int periodic = e->mesh->periodic;
  const double dim[3] = {e->mesh->dim[0], e->mesh->dim[1], e->mesh->dim[2]};
  const double max_distance = e->mesh->r_cut_max;

  /* Anything to do here? */
  if (!((cell_is_active_gravity(ci, e) && ci->nodeID == nodeID) ||
        (cell_is_active_gravity(cj, e) && cj->nodeID == nodeID)))
    return regular_task;

#ifdef SWIFT_DEBUG_CHECKS

  const int gcount_i = ci->grav.count;
  const int gcount_j = cj->grav.count;

  /* Early abort? */
  if (gcount_i == 0 || gcount_j == 0)
    error("Doing pair gravity on an empty cell !");

  /* Sanity check */
  if (ci == cj) error("Pair interaction between a cell and itself.");

  if (cell_is_active_gravity(ci, e) &&
      ci->grav.ti_old_multipole != e->ti_current)
    error("ci->grav.multipole not drifted.");
  if (cell_is_active_gravity(cj, e) &&
      cj->grav.ti_old_multipole != e->ti_current)
    error("cj->grav.multipole not drifted.");
#endif

  TIMER_TIC;

  /* Recover the multipole information */
  struct gravity_tensors* const multi_i = ci->grav.multipole;
  struct gravity_tensors* const multi_j = cj->grav.multipole;

  /* Get the distance between the CoMs */
  double dx = multi_i->CoM[0] - multi_j->CoM[0];
  double dy = multi_i->CoM[1] - multi_j->CoM[1];
  double dz = multi_i->CoM[2] - multi_j->CoM[2];

  /* Apply BC */
  if (periodic) {
    dx = nearest(dx, dim[0]);
    dy = nearest(dy, dim[1]);
    dz = nearest(dz, dim[2]);
  }
  const double r2 = dx * dx + dy * dy + dz * dz;

  /* Minimal distance between any 2 particles in the two cells */
  const double r_lr_check = sqrt(r2) - (multi_i->r_max + multi_j->r_max);

  /* Are we beyond the distance where the truncated forces are 0? */
  if (periodic && r_lr_check > max_distance) {

#ifdef SWIFT_DEBUG_CHECKS
    if (cell_is_active_gravity(ci, e))
      accumulate_add_ll(&multi_i->pot.num_interacted,
                        multi_j->m_pole.num_gpart);
    if (cell_is_active_gravity(cj, e))
      accumulate_add_ll(&multi_j->pot.num_interacted,
                        multi_i->m_pole.num_gpart);
#endif

#ifdef SWIFT_GRAVITY_FORCE_CHECKS
    /* Need to account for the interactions we missed */
    if (cell_is_active_gravity(ci, e))
      accumulate_add_ll(&multi_i->pot.num_interacted_pm,
                        multi_j->m_pole.num_gpart);
    if (cell_is_active_gravity(cj, e))
      accumulate_add_ll(&multi_j->pot.num_interacted_pm,
                        multi_i->m_pole.num_gpart);
#endif
    return regular_task;
  }

  /* OK, we actually need to compute this pair. Let's find the cheapest
   * option... */

  if (ci->grav.count <= 1 || cj->grav.count <= 1) {

    // printf("BEING CHEAP \n");

    /* We have two cheap cells. Go P-P. */
    runner_dopair_recursive_grav(r, ci, cj, 0);
    return regular_task;

    /* Can we use M-M interactions ? */
  } else if (gravity_M2L_accept_symmetric(e->gravity_properties, multi_i,
                                          multi_j, r2,
                                          /*use_rebuild_sizes=*/0, periodic)) {

    // printf("qid:%i MM\n", r->qid);

    // printf("DOING MM \n");

    /* Go M-M */
    runner_dopair_recursive_grav(r, ci, cj, 0);
    return regular_task;

    /* Did we reach the bottom? */
  } else if (!ci->split && !cj->split) {
    // printf("qid:%i PP here we go!\n", r->qid);

    // printf("qid:%i PP packed:%i\n", r->qid, *packed);
    // fflush(stdout);

    // if (!ci->split && !cj->split){
    // printf("qid:%i tree condition met\n", r->qid);}

    /* We have two leaves. Go P-P. */
    return runner_dopair_grav_pp_new(
        r, ci, cj, /*symmetric*/ 1, /*allow_mpoles=*/1,
        gravity_gpu_values_send_pair, gravity_gpu_values_send_pair_d,
        gravity_gpu_values_recv_pair, gravity_gpu_values_recv_pair_d,
        grav_cells_pair, grav_tasks_pair, t, ncells, max_cell_size, stream);

  } else {

    enum runner_gpu_task_type task_type = regular_task;

    // printf("qid:%i recursing\n", r->qid);

    /* Alright, we'll have to split and recurse. */
    /* We know at least one of ci and cj is splittable */

    const double ri_max = multi_i->r_max;
    const double rj_max = multi_j->r_max;

    /* Split the larger of the two cells and start over again */
    if (ri_max > rj_max) {

      /* Can we actually split that interaction ? */
      if (ci->split) {

        /* Loop over ci's children */
        for (int k = 0; k < 8; k++) {
          if (ci->progeny[k] != NULL) {
            // runner_dopair_recursive_grav(r, ci->progeny[k], cj, 0);
            enum runner_gpu_task_type child_type =
                runner_dopair_recursive_grav_new(
                    r, ci->progeny[k], cj, 0, gravity_gpu_values_send_pair,
                    gravity_gpu_values_send_pair_d,
                    gravity_gpu_values_recv_pair,
                    gravity_gpu_values_recv_pair_d, grav_cells_pair,
                    grav_tasks_pair, t, ncells, max_cell_size, stream);
            if (child_type > task_type) task_type = child_type;
          }
        }

      } else {
        /* cj is split */

        /* MATTHIEU: This could maybe be replaced by P-M interactions ?  */

        /* Loop over cj's children */
        for (int k = 0; k < 8; k++) {
          if (cj->progeny[k] != NULL) {
            // runner_dopair_recursive_grav(r, ci, cj->progeny[k], 0);
            enum runner_gpu_task_type child_type =
                runner_dopair_recursive_grav_new(
                    r, ci, cj->progeny[k], 0, gravity_gpu_values_send_pair,
                    gravity_gpu_values_send_pair_d,
                    gravity_gpu_values_recv_pair,
                    gravity_gpu_values_recv_pair_d, grav_cells_pair,
                    grav_tasks_pair, t, ncells, max_cell_size, stream);
            if (child_type > task_type) task_type = child_type;
          }
        }
      }
    } else {

      /* Can we actually split that interaction ? */
      if (cj->split) {

        /* Loop over cj's children */
        for (int k = 0; k < 8; k++) {
          if (cj->progeny[k] != NULL) {
            // runner_dopair_recursive_grav(r, ci, cj->progeny[k], 0);
            enum runner_gpu_task_type child_type =
                runner_dopair_recursive_grav_new(
                    r, ci, cj->progeny[k], 0, gravity_gpu_values_send_pair,
                    gravity_gpu_values_send_pair_d,
                    gravity_gpu_values_recv_pair,
                    gravity_gpu_values_recv_pair_d, grav_cells_pair,
                    grav_tasks_pair, t, ncells, max_cell_size, stream);
            if (child_type > task_type) task_type = child_type;
          }
        }

      } else {
        /* ci is split */

        /* MATTHIEU: This could maybe be replaced by P-M interactions ?  */

        /* Loop over ci's children */
        for (int k = 0; k < 8; k++) {
          if (ci->progeny[k] != NULL) {
            // runner_dopair_recursive_grav(r, ci->progeny[k], cj, 0);
            enum runner_gpu_task_type child_type =
                runner_dopair_recursive_grav_new(
                    r, ci->progeny[k], cj, 0, gravity_gpu_values_send_pair,
                    gravity_gpu_values_send_pair_d,
                    gravity_gpu_values_recv_pair,
                    gravity_gpu_values_recv_pair_d, grav_cells_pair,
                    grav_tasks_pair, t, ncells, max_cell_size, stream);
            if (child_type > task_type) task_type = child_type;
          }
        }
      }
    }

    if (gettimer) TIMER_TOC(timer_dosub_pair_grav);
    return task_type;
  }

  if (gettimer) TIMER_TOC(timer_dosub_pair_grav);
  return regular_task;
}

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
 * @brief Flush any leftover packed self-gravity work owned by a runner.
 *
 * @param r The runner whose GPU batch should be flushed.
 * @return The outcome of the leftover flush attempt.
 */
enum runner_gpu_task_type runner_gpu_flush_leftover_self(struct runner* r) {

  const int ncells_flush_self = r->gpu.grav_batch_self_count;
  const int max_cell_size = r->gpu.grav_max_cell_size;

  if (ncells_flush_self == 0) return regular_task;

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

  return flushed_self_task;
}

/**
 * @brief Flush any leftover packed pair-gravity work owned by a runner.
 *
 * @param r The runner whose GPU batch should be flushed.
 * @return The outcome of the leftover flush attempt.
 */
enum runner_gpu_task_type runner_gpu_flush_leftover_pair(struct runner* r) {

  const int ncells_flush_pair = r->gpu.grav_batch_pair_count;
  const int max_cell_size = r->gpu.grav_max_cell_size;

  if (ncells_flush_pair == 0) return regular_task;

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

  return flushed_pair_task;
}
