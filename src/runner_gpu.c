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
#include "gpu_mapping.h"
#include "runner.h"
#include "runner_doiact_grav.h"
#include "scheduler.h"
#include "timers.h"

#include <stdlib.h>

/**
 * @brief Launch the GPU P-P gravity kernel for a packed batch.
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
extern void pair_pp_offload_new(
    int periodic, float rmax_i, float rmax_j, double min_trunc,
    const float* r_s_inv, const int* gcount_i, const int* gcount_padded_i,
    const int* gcount_j, const int* gcount_padded_j, int ci_active,
    int cj_active, float dim_0, float dim_1, float dim_2, int symmetric,
    struct gravity_gpu_values_send* gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_d, int ncells,
    int max_cell_size, GPUStream stream);

/**
 * @brief Unpack the GPU parameters from the parameter file.
 *
 * @param e The #engine to unpack the parameters for.
 */
void runner_gpu_params_init(struct engine* e) {

  /* Unpack the number of cells we will pack onto the GPU at a time. */
  e->ncells_per_gpu_grav_pack = parser_get_opt_param_int(
      e->parameter_file, "GPU:ncells_per_gpu_grav_pack", 8);
  if (e->ncells_per_gpu_grav_pack < 2) {
    error("GPU:ncells_per_gpu_grav_pack must be >= 2");
  }
}

/**
 * @brief Mark a packed self-gravity task as complete on the scheduler.
 *
 * @param r The #runner owning the task.
 * @param sched The scheduler tracking the task.
 * @param t The task to complete.
 */
static void runner_gpu_complete_self_task(struct runner* r,
                                          struct scheduler* sched,
                                          struct task* t) {
  lock_lock(&sched->queues[r->qid].lock);
  sched->queues[r->qid].gpu_self_tasks_left--;
  (void)lock_unlock(&sched->queues[r->qid].lock);
  scheduler_done(sched, t);
}

/**
 * @brief Mark a packed pair-gravity task as complete on the scheduler.
 *
 * @param r The #runner owning the task.
 * @param sched The scheduler tracking the task.
 * @param t The task to complete.
 */
void runner_gpu_complete_pair_task(struct runner* r, struct scheduler* sched,
                                   struct task* t) {
  lock_lock(&sched->queues[r->qid].lock);
  sched->queues[r->qid].gpu_pair_tasks_left--;
  (void)lock_unlock(&sched->queues[r->qid].lock);
  scheduler_done(sched, t);
}

/**
 * @brief Complete all self-gravity tasks in the current GPU batch.
 *
 * @param r The #runner owning the batch.
 * @param sched The scheduler tracking the tasks.
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
 * @brief Complete all unique pair-gravity tasks in the current GPU batch.
 *
 * @param r The #runner owning the batch.
 * @param sched The scheduler tracking the tasks.
 */
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
 * @brief Pack one leaf pair-gravity interaction into the runner GPU batch.
 *
 * This function populates the gravity caches for both cells and copies the
 * particle data into the next available slot in the pair batch buffer.  The
 * batch counter is incremented but no GPU work is launched; the caller is
 * responsible for checking whether the batch is full and calling
 * runner_dopair_grav_pp_flush() when appropriate.
 *
 * @param r The #runner.
 * @param ci The first #cell.
 * @param cj The other #cell.
 * @param symmetric Are we updating both cells (1) or just ci (0) ?
 * @param allow_mpole Are we allowing the use of M2P interactions ?
 * @param gravity_gpu_values_send_pair Host send buffer for this batch.
 * @param gravity_gpu_values_recv_pair Host receive buffer for this batch.
 * @param grav_cells_pair Array of cell pointers for this batch.
 * @param grav_tasks_pair Array of task pointers for this batch.
 * @param t The top-level #task currently being processed.
 * @param max_cell_size The maximum number of particles per packed cell.
 * @param stream The GPU stream used for timing events.
 */
static void runner_dopair_grav_pp_pack(
    struct runner* r, struct cell* ci, struct cell* cj, const int symmetric,
    const int allow_mpole,
    struct gravity_gpu_values_send* gravity_gpu_values_send_pair,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_pair,
    struct cell** grav_cells_pair, struct task** grav_tasks_pair,
    struct task* t, int max_cell_size, GPUStream stream) {

  /* Recover some useful constants */
  const struct engine* e = r->e;
  const int periodic = e->mesh->periodic;
  const float dim[3] = {(float)e->mesh->dim[0], (float)e->mesh->dim[1],
                        (float)e->mesh->dim[2]};
  const double min_trunc = e->mesh->r_cut_min;

  TIMER_TIC;

  /* Record activity status */
  const int ci_active =
      cell_is_active_gravity(ci, e) && (ci->nodeID == e->nodeID);
  const int cj_active =
      cell_is_active_gravity(cj, e) && (cj->nodeID == e->nodeID);

  (void)cj_active; /* used only by debug checks below */
  (void)ci_active;

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

  GPUEvent startpack, stoppack;
  GPUEventCreate(&startpack);
  GPUEventCreate(&stoppack);

  GPUEventRecord(startpack, stream);

  /* ---- Pack ci data into the send buffer ---- */
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

    /* ---- Pack cj data into the send buffer ---- */
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

    /* ---- Zero the receive buffer for ci slot ---- */
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

    /* ---- Zero the receive buffer for cj slot ---- */
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

  /* Store decision on BOTH blocks */
  gravity_gpu_values_send_pair[r->gpu.grav_batch_pair_count * max_cell_size]
      .use_full = use_full;
  gravity_gpu_values_send_pair[(r->gpu.grav_batch_pair_count + 1) *
                               max_cell_size]
      .use_full = use_full;

  /* Update that we packed a pair into our array */
  r->gpu.grav_batch_pair_count += 2;

  gravity_cache_zero_output(ci_cache, gcount_padded_i);
  gravity_cache_zero_output(cj_cache, gcount_padded_j);

  cell_gunlocktree(b);
  cell_gunlocktree(a);
}

/**
 * @brief Flush a full pair-gravity GPU batch: H2D copy, kernel launch, D2H
 *        copy, stream synchronisation, and result unpacking to particles.
 *
 * After unpacking, any tasks from *previous* top-level calls that were packed
 * into this batch are completed via scheduler_done().  The task pointed to
 * by @p current_task is assumed to still be in progress and is skipped.
 * The batch counter and metadata arrays are reset to zero so the buffer can
 * be reused immediately.
 *
 * @param r The #runner.
 * @param gravity_gpu_values_send_pair Host send buffer.
 * @param gravity_gpu_values_send_pair_d Device send buffer.
 * @param gravity_gpu_values_recv_pair Host receive buffer.
 * @param gravity_gpu_values_recv_pair_d Device receive buffer.
 * @param grav_cells_pair Array of cell pointers for this batch.
 * @param grav_tasks_pair Array of task pointers for this batch.
 * @param current_task The top-level task currently being walked (not
 * completed).
 * @param ncells The batch capacity (number of cell slots).
 * @param max_cell_size The maximum number of particles per packed cell.
 * @param stream The GPU stream to use.
 */
static void runner_dopair_grav_pp_flush(
    struct runner* r,
    struct gravity_gpu_values_send* gravity_gpu_values_send_pair,
    struct gravity_gpu_values_send* gravity_gpu_values_send_pair_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_pair,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_pair_d,
    struct cell** grav_cells_pair, struct task** grav_tasks_pair,
    struct task* current_task, int ncells, int max_cell_size,
    GPUStream stream) {

  const int ncells_flush = r->gpu.grav_batch_pair_count;
  if (ncells_flush == 0) return;

  /* Retrieve kernel parameters from the first pair in the batch. */
  struct cell* ci_flush = grav_cells_pair[0];
  struct cell* cj_flush = grav_cells_pair[1];

  if (ci_flush == NULL || cj_flush == NULL)
    error("pair flush: NULL packed cells");

  const struct engine* e = r->e;
  const int periodic = e->mesh->periodic;
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
  const float dim_0 = (float)e->mesh->dim[0];
  const float dim_1 = (float)e->mesh->dim[1];
  const float dim_2 = (float)e->mesh->dim[2];

  /* ---- H2D copy ---- */
  {
    TIMER_TIC;

    GPUMemcpyAsync(
        gravity_gpu_values_send_pair_d, gravity_gpu_values_send_pair,
        ncells_flush * max_cell_size * sizeof(struct gravity_gpu_values_send),
        GPU_MEMCPY_HOST_TO_DEVICE, stream);
    GPUMemcpyAsync(
        gravity_gpu_values_recv_pair_d, gravity_gpu_values_recv_pair,
        ncells_flush * max_cell_size * sizeof(struct gravity_gpu_values_recv),
        GPU_MEMCPY_HOST_TO_DEVICE, stream);

    GPUError err2 = GPUGetLastError();
    if (err2 != GPU_SUCCESS)
      printf("Error (flush H2D): %s\n", GPUGetErrorString(err2));

    /* ---- Kernel launch ---- */
    pair_pp_offload_new(
        periodic, rmax_i, rmax_j, min_trunc, &r_s_inv, &gcount_i,
        &gcount_padded_i, &gcount_j, &gcount_padded_j, ci_active, cj_active,
        dim_0, dim_1, dim_2, /*symmetric=*/1, gravity_gpu_values_send_pair_d,
        gravity_gpu_values_recv_pair_d, ncells_flush, max_cell_size, stream);

    /* ---- D2H copy ---- */
    GPUMemcpyAsync(
        gravity_gpu_values_recv_pair, gravity_gpu_values_recv_pair_d,
        ncells_flush * max_cell_size * sizeof(struct gravity_gpu_values_recv),
        GPU_MEMCPY_DEVICE_TO_HOST, stream);

    GPUStreamSynchronize(stream);

    TIMER_TOC(timer_doself_grav_pp);
  }

  GPUError err3 = GPUGetLastError();
  if (err3 != GPU_SUCCESS)
    printf("Error (flush kernel): %s\n", GPUGetErrorString(err3));

  /* ---- Unpack results back to particles ---- */
  {
    TIMER_TIC;

    for (int j = 0; j < ncells_flush; j += 2) {

      if (grav_cells_pair[j] == NULL || grav_cells_pair[j + 1] == NULL)
        error("PAIR UNPACK: NULL cell j=%d packed=%d qid=%d", j, ncells_flush,
              r->qid);

      if (grav_tasks_pair[j / 2] == NULL)
        error("PAIR UNPACK: NULL task k=%d (j=%d) packed=%d qid=%d", j / 2, j,
              ncells_flush, r->qid);

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
      }
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

  /* Complete any tasks from previous top-level calls that were packed
     into this batch, but skip current_task (still being walked). */
  struct scheduler* sched = &r->e->sched;
  struct task* prev_task = NULL;
  for (int j = 0; j < ncells_flush; j += 2) {
    struct task* batch_task = grav_tasks_pair[j / 2];
    if (batch_task != prev_task && batch_task != current_task) {
      runner_gpu_complete_pair_task(r, sched, batch_task);
      prev_task = batch_task;
    }
    grav_cells_pair[j] = NULL;
    grav_cells_pair[j + 1] = NULL;
    grav_tasks_pair[j / 2] = NULL;
  }

  /* Reset the batch counter so the buffer can be reused. */
  r->gpu.grav_batch_pair_count = 0;
}

/**
 * @brief Pack a leaf pair-gravity interaction and flush if the batch is full.
 *
 * This is the main entry point called from runner_dopair_recursive_grav_new()
 * when two unsplit leaf cells are reached. It packs the pair, and if the batch
 * is now full it flushes the GPU work before returning.
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
    struct task* t, int ncells, int max_cell_size, GPUStream stream) {

  /* Pack the pair into the batch buffer. */
  runner_dopair_grav_pp_pack(r, ci, cj, symmetric, allow_mpole,
                             gravity_gpu_values_send_pair,
                             gravity_gpu_values_recv_pair, grav_cells_pair,
                             grav_tasks_pair, t, max_cell_size, stream);

  /* If we have filled our batch, flush it and reset the count. */
  if (r->gpu.grav_batch_pair_count >= ncells) {
    runner_dopair_grav_pp_flush(
        r, gravity_gpu_values_send_pair, gravity_gpu_values_send_pair_d,
        gravity_gpu_values_recv_pair, gravity_gpu_values_recv_pair_d,
        grav_cells_pair, grav_tasks_pair, t, ncells, max_cell_size, stream);
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

  GPUEvent startpack, stoppack;
  GPUEventCreate(&startpack);
  GPUEventCreate(&stoppack);
  GPUEventRecord(startpack, r->gpu.stream);

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
    GPUEvent startcopyH2D, stopcopyH2D;
    GPUEventCreate(&startcopyH2D);
    GPUEventCreate(&stopcopyH2D);
    GPUEventRecord(startcopyH2D, r->gpu.stream);

    {
      TIMER_TIC;

      GPUMemcpyAsync(
          r->gpu.gravity_gpu_values_send_self_d,
          r->gpu.gravity_gpu_values_send_self,
          ncells * max_cell_size * sizeof(struct gravity_gpu_values_send),
          GPU_MEMCPY_HOST_TO_DEVICE, r->gpu.stream);
      GPUMemcpyAsync(
          r->gpu.gravity_gpu_values_recv_self_d,
          r->gpu.gravity_gpu_values_recv_self,
          ncells * max_cell_size * sizeof(struct gravity_gpu_values_recv),
          GPU_MEMCPY_HOST_TO_DEVICE, r->gpu.stream);

      GPUEventRecord(stopcopyH2D, r->gpu.stream);

      GPUError err2 = GPUGetLastError();
      if (err2 != GPU_SUCCESS) printf("Error2: %s\n", GPUGetErrorString(err2));

      GPUEvent startker, stopker;
      GPUEventCreate(&startker);
      GPUEventCreate(&stopker);
      GPUEventRecord(startker, r->gpu.stream);

      runner_doself_recursive_grav_new(r, ci, 1,
                                       r->gpu.gravity_gpu_values_send_self_d,
                                       r->gpu.gravity_gpu_values_recv_self_d,
                                       ncells, max_cell_size, r->gpu.stream);

      GPUEventRecord(stopker, r->gpu.stream);

      GPUEvent startcopyD2H, stopcopyD2H;
      GPUEventCreate(&startcopyD2H);
      GPUEventCreate(&stopcopyD2H);
      GPUEventRecord(startcopyD2H, r->gpu.stream);

      GPUMemcpyAsync(
          r->gpu.gravity_gpu_values_recv_self,
          r->gpu.gravity_gpu_values_recv_self_d,
          ncells * max_cell_size * sizeof(struct gravity_gpu_values_recv),
          GPU_MEMCPY_DEVICE_TO_HOST, r->gpu.stream);

      GPUEventRecord(stopcopyD2H, r->gpu.stream);
      GPUStreamSynchronize(r->gpu.stream);

      TIMER_TOC(timer_doself_grav_pp);
    }

    GPUError err3 = GPUGetLastError();
    if (err3 != GPU_SUCCESS) printf("Error3: %s\n", GPUGetErrorString(err3));

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
    struct task* t, int ncells, int max_cell_size, GPUStream stream) {

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

    /* We have two cheap cells. Go P-P. */
    runner_dopair_recursive_grav(r, ci, cj, 0);
    return regular_task;

    /* Can we use M-M interactions ? */
  } else if (gravity_M2L_accept_symmetric(e->gravity_properties, multi_i,
                                          multi_j, r2,
                                          /*use_rebuild_sizes=*/0, periodic)) {

    /* Go M-M */
    runner_dopair_recursive_grav(r, ci, cj, 0);
    return regular_task;

    /* Did we reach the bottom? */
  } else if (!ci->split && !cj->split) {

    /* We have two leaves. Go P-P. */
    return runner_dopair_grav_pp_new(
        r, ci, cj, /*symmetric*/ 1, /*allow_mpoles=*/1,
        gravity_gpu_values_send_pair, gravity_gpu_values_send_pair_d,
        gravity_gpu_values_recv_pair, gravity_gpu_values_recv_pair_d,
        grav_cells_pair, grav_tasks_pair, t, ncells, max_cell_size, stream);

  } else {

    enum runner_gpu_task_type task_type = regular_task;

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

    /* Determine the return type based on whether *this* walk actually
       produced any leaf pairs.  task_type tracks the highest child return
       value: packed_task or flushed_pair_task means at least one leaf
       pair was generated by this walk. */
    enum runner_gpu_task_type final_type;
    if (task_type >= packed_task) {
      /* This walk produced leaf pairs.  If some are still in the buffer
         they will be flushed later; if all were flushed the task is done. */
      if (r->gpu.grav_batch_pair_count > 0) {
        final_type = packed_task;
      } else {
        final_type = flushed_pair_task;
      }
    } else {
      /* No leaf pairs were produced at all (all M-M or truncated). */
      final_type = regular_task;
    }

    if (gettimer) TIMER_TOC(timer_dosub_pair_grav);
    return final_type;
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

  GPUSetDevice(0);

  GPUDeviceProp prop;
  GPUGetDeviceProperties(&prop, 0);

  const int max_cell_size = space_subsize_self_grav + 100;

  gpu->grav_max_cell_size = max_cell_size;

  gpu->grav_batch_ncells = e->ncells_per_gpu_grav_pack;

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

  GPUStreamCreate(&gpu->stream);

  GPUMalloc((void**)&gpu->gravity_gpu_values_send_self_d,
            gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                sizeof(struct gravity_gpu_values_send));
  GPUHostMalloc((void**)&gpu->gravity_gpu_values_send_self,
                gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                    sizeof(struct gravity_gpu_values_send));

  GPUMalloc((void**)&gpu->gravity_gpu_values_send_pair_d,
            gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                sizeof(struct gravity_gpu_values_send));
  GPUHostMalloc((void**)&gpu->gravity_gpu_values_send_pair,
                gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                    sizeof(struct gravity_gpu_values_send));

  GPUMalloc((void**)&gpu->gravity_gpu_values_recv_self_d,
            gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                sizeof(struct gravity_gpu_values_recv));
  GPUHostMalloc((void**)&gpu->gravity_gpu_values_recv_self,
                gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                    sizeof(struct gravity_gpu_values_recv));

  GPUMalloc((void**)&gpu->gravity_gpu_values_recv_pair_d,
            gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                sizeof(struct gravity_gpu_values_recv));
  GPUHostMalloc((void**)&gpu->gravity_gpu_values_recv_pair,
                gpu->grav_batch_ncells * gpu->grav_max_cell_size *
                    sizeof(struct gravity_gpu_values_recv));

  gpu->grav_cells_self = malloc(gpu->grav_batch_ncells * sizeof(struct cell*));
  gpu->grav_cells_pair = malloc(gpu->grav_batch_ncells * sizeof(struct cell*));
  gpu->grav_tasks_self = malloc(gpu->grav_batch_ncells * sizeof(struct task*));
  gpu->grav_tasks_pair = malloc(gpu->grav_batch_ncells * sizeof(struct task*));
  gpu->cell_active = malloc(gpu->grav_batch_ncells * sizeof(int));

  if (gpu->grav_cells_self == NULL || gpu->grav_cells_pair == NULL ||
      gpu->grav_tasks_self == NULL || gpu->grav_tasks_pair == NULL ||
      gpu->cell_active == NULL)
    error("Failed to allocate runner GPU host metadata arrays.");

  const GPUError err = GPUGetLastError();
  if (err != GPU_SUCCESS)
    error("runner_gpu_init failed: %s", GPUGetErrorString(err));
}

/**
 * @brief Clean the GPU-specific state attached to a runner.
 *
 * @param r The runner whose GPU state to clean.
 */
void runner_gpu_clean(struct runner* r) {

  struct gpu_runner* gpu = &r->gpu;

  GPUFreeHost(gpu->gravity_gpu_values_send_self);
  GPUFreeHost(gpu->gravity_gpu_values_recv_self);
  GPUFree(gpu->gravity_gpu_values_send_self_d);
  GPUFree(gpu->gravity_gpu_values_recv_self_d);

  GPUFreeHost(gpu->gravity_gpu_values_send_pair);
  GPUFreeHost(gpu->gravity_gpu_values_recv_pair);
  GPUFree(gpu->gravity_gpu_values_send_pair_d);
  GPUFree(gpu->gravity_gpu_values_recv_pair_d);

  free(gpu->grav_cells_self);
  free(gpu->grav_tasks_self);
  free(gpu->grav_cells_pair);
  free(gpu->grav_tasks_pair);
  free(gpu->cell_active);

  GPUStreamDestroy(gpu->stream);

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

    GPUMemcpyAsync(r->gpu.gravity_gpu_values_send_self_d,
                   r->gpu.gravity_gpu_values_send_self,
                   ncells_flush_self * max_cell_size *
                       sizeof(struct gravity_gpu_values_send),
                   GPU_MEMCPY_HOST_TO_DEVICE, r->gpu.stream);
    GPUMemcpyAsync(r->gpu.gravity_gpu_values_recv_self_d,
                   r->gpu.gravity_gpu_values_recv_self,
                   ncells_flush_self * max_cell_size *
                       sizeof(struct gravity_gpu_values_recv),
                   GPU_MEMCPY_HOST_TO_DEVICE, r->gpu.stream);

    GPUError err4 = GPUGetLastError();
    if (err4 != GPU_SUCCESS) printf("Error4: %s\n", GPUGetErrorString(err4));

    runner_doself_recursive_grav_new(
        r, r->gpu.grav_cells_self[0], 1, r->gpu.gravity_gpu_values_send_self_d,
        r->gpu.gravity_gpu_values_recv_self_d, ncells_flush_self, max_cell_size,
        r->gpu.stream);

    GPUMemcpyAsync(r->gpu.gravity_gpu_values_recv_self,
                   r->gpu.gravity_gpu_values_recv_self_d,
                   ncells_flush_self * max_cell_size *
                       sizeof(struct gravity_gpu_values_recv),
                   GPU_MEMCPY_DEVICE_TO_HOST, r->gpu.stream);

    GPUStreamSynchronize(r->gpu.stream);

    TIMER_TOC(timer_doself_grav_pp);
  }

  GPUError err5 = GPUGetLastError();
  if (err5 != GPU_SUCCESS) printf("Error5: %s\n", GPUGetErrorString(err5));

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

    GPUMemcpyAsync(r->gpu.gravity_gpu_values_send_pair_d,
                   r->gpu.gravity_gpu_values_send_pair,
                   ncells_flush_pair * max_cell_size *
                       sizeof(struct gravity_gpu_values_send),
                   GPU_MEMCPY_HOST_TO_DEVICE, r->gpu.stream);
    GPUMemcpyAsync(r->gpu.gravity_gpu_values_recv_pair_d,
                   r->gpu.gravity_gpu_values_recv_pair,
                   ncells_flush_pair * max_cell_size *
                       sizeof(struct gravity_gpu_values_recv),
                   GPU_MEMCPY_HOST_TO_DEVICE, r->gpu.stream);

    GPUError err4 = GPUGetLastError();
    if (err4 != GPU_SUCCESS) printf("Error4: %s\n", GPUGetErrorString(err4));

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

    GPUMemcpyAsync(r->gpu.gravity_gpu_values_recv_pair,
                   r->gpu.gravity_gpu_values_recv_pair_d,
                   ncells_flush_pair * max_cell_size *
                       sizeof(struct gravity_gpu_values_recv),
                   GPU_MEMCPY_DEVICE_TO_HOST, r->gpu.stream);

    GPUStreamSynchronize(r->gpu.stream);

    TIMER_TOC(timer_doself_grav_pp);
  }

  GPUError err5 = GPUGetLastError();
  if (err5 != GPU_SUCCESS) printf("Error5: %s\n", GPUGetErrorString(err5));

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
