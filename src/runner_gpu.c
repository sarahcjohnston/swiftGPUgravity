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
 * @param rmavalues_i.x Bounding radius for cell i.
 * @param rmavalues_j.x Bounding radius for cell j.
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
    int periodic,
    double min_trunc,
    const float *r_s_inv,
    const int *pair_counts_d,
    const int *pair_offsets_d,
    const int *pair_active_counts_d,
    const int *pair_active_offsets_d,
    const int *pair_active_index_d,
    float dim_0, float dim_1, float dim_2,
    struct gravity_gpu_values_send *gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv *gravity_gpu_values_recv_d,
    int ncells,
    int max_cell_size,
    int max_active_count,
    GPUStream stream);
    
    
extern void self_pp_offload_new(
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
    GPUStream stream);
/**
 * @brief Unpack the GPU parameters from the parameter file.
 *
 * @param e The #engine to unpack the parameters for.
 */
void runner_gpu_params_init(struct engine *e) {

  /* Unpack the number of cells we will pack onto the GPU at a time. */
  e->ncells_per_gpu_grav_pack = parser_get_opt_param_int(
      e->parameter_file, "GPU:ncells_per_gpu_grav_pack", -1);

  if (e->ncells_per_gpu_grav_pack == 0 ||
      e->ncells_per_gpu_grav_pack < -1) {
    error("GPU:ncells_per_gpu_grav_pack must be >= 1 for user selected or -1 for auto");
  }
}

/**
 * @brief Store the values for packing the self cells without gaps.
 *
 * @param substream The stream the task is on.
 * @param slot The number cell in the pack it is
 * @param count The size of the cell.
 */
static inline void append_packed_self_cell(
    struct gpu_runner_substream *substream, int slot, int count) {

  substream->self_offsets_h[slot] = substream->self_total_count;
  substream->self_counts_h[slot] = count;
  substream->self_total_count += count;
}

/**
 * @brief Store the values for packing the pair cells without gaps.
 *
 * @param substream The stream the task is on.
 * @param slot The number cell in the pack it is
 * @param count The size of the cell.
 */
static inline void append_packed_pair_cell(
    struct gpu_runner_substream *substream, int slot, int count) {

  substream->pair_offsets_h[slot] = substream->pair_total_count;
  substream->pair_counts_h[slot] = count;
  substream->pair_total_count += count;
}

/**
 * @brief Mark a packed self-gravity task as complete on the scheduler.
 *
 * @param r The #runner owning the task.
 * @param sched The scheduler tracking the task.
 * @param t The task to complete.
 */
static void runner_gpu_complete_self_task(struct runner *r,
                                          struct scheduler *sched,
                                          struct task *t) {
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
void runner_gpu_complete_pair_task(struct runner *r, struct scheduler *sched,
                                   struct task *t) {
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
void runner_gpu_complete_self_batch(struct runner *r, struct scheduler *sched,
                                    struct gpu_runner_substream *substream) {
  const int count = substream->grav_batch_self_count;
  struct task *prev_task = NULL;

  for (int i = 0; i < count; i++) {
    struct task *task = substream->grav_tasks_self[i];

    if (task != prev_task && !task->gpu_completed) {
	  runner_gpu_complete_self_task(r, sched, task);
	  task->gpu_completed = 1;
	  prev_task = task;
	}

    substream->grav_cells_self[i] = NULL;
    substream->grav_tasks_self[i] = NULL;
  }

  substream->grav_batch_self_count = 0;
  substream->busy = 0;

  for (int i = 0; i < count; i++) {
    substream->self_counts_h[i] = 0;
    substream->self_offsets_h[i] = 0;
    substream->self_active_counts_h[i] = 0;
    substream->self_rmax_h[i] = 0.f;
  }
  substream->self_total_count = 0;
  substream->self_max_active_count = 0;
  substream->self_total_active_count = 0;
}

/**
 * @brief Complete all unique pair-gravity tasks in the current GPU batch.
 *
 * @param r The #runner owning the batch.
 * @param sched The scheduler tracking the tasks.
 */
void runner_gpu_complete_pair_batch(struct runner *r, struct scheduler *sched,
                                    struct gpu_runner_substream *substream) {
  const int count = substream->grav_batch_pair_count;
  struct task *prev_task = NULL;

  for (int i = 0; i < count; i += 2) {
    const int pair_slot = i / 2;
    struct task *task = substream->grav_tasks_pair[pair_slot];
    const int internal = substream->grav_pair_internal_from_self[pair_slot];

    if (!internal && task != prev_task) {
      runner_gpu_complete_pair_task(r, sched, task);
      prev_task = task;
    }

    substream->grav_cells_pair[i] = NULL;
    substream->grav_cells_pair[i + 1] = NULL;
    substream->grav_tasks_pair[pair_slot] = NULL;
    substream->grav_pair_internal_from_self[pair_slot] = 0;
  }

  substream->grav_batch_pair_count = 0;
  substream->busy = 0;

  for (int i = 0; i < count; i++) {
    substream->pair_counts_h[i] = 0;
    substream->pair_offsets_h[i] = 0;
  }
  substream->pair_total_count = 0;
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
    struct runner *r, struct gpu_runner_substream *substream,
    struct cell *ci, struct cell *cj, const int symmetric,
    const int allow_mpole,
    struct gravity_gpu_values_send *gravity_gpu_values_send_pair,
    struct gravity_gpu_values_recv *gravity_gpu_values_recv_pair,
    struct cell **grav_cells_pair, struct task **grav_tasks_pair,
    unsigned char *grav_pair_internal_from_self,
    struct task *t, int internal_from_self,
    int max_cell_size, GPUStream stream) {

  /* Recover some useful constants */
  const struct engine *e = r->e;
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
  struct gravity_cache *const ci_cache = &r->ci_gravity_cache;
  struct gravity_cache *const cj_cache = &r->cj_gravity_cache;

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
  
    if (gcount_i > max_cell_size)
    error("Pair pack overflow: gcount_i=%d > max_cell_size=%d", gcount_i, max_cell_size);

  if (gcount_j > max_cell_size)
    error("Pair pack overflow: gcount_j=%d > max_cell_size=%d", gcount_j, max_cell_size);

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

  struct cell *ci0 = ci;
  struct cell *cj0 = cj;
  struct cell *a = ci0, *b = cj0;

  if (a > b) {
    struct cell *tmp = a;
    a = b;
    b = tmp;
  }
  while (cell_glocktree(a)) {
    ;
  }
  while (cell_glocktree(b)) {
    ;
  }

  //GPUEvent startpack, stoppack;
  //GPUEventCreate(&startpack);
  //GPUEventCreate(&stoppack);
  
  const int slot_i = substream->grav_batch_pair_count;
  const int slot_j = slot_i + 1;

  append_packed_pair_cell(substream, slot_i, gcount_i);
  append_packed_pair_cell(substream, slot_j, gcount_j);

  const int off_i = substream->pair_offsets_h[slot_i];
  const int off_j = substream->pair_offsets_h[slot_j];

  int active_count_i = 0;
  int active_count_j = 0;
  const int active_base_i = substream->pair_total_active_count;
  substream->pair_active_offsets_h[slot_i] = active_base_i;

  for (int i = 0; i < gcount_i; i++) {
  	if (ci_cache->active[i] > 0) {
    		substream->pair_active_index_h[active_base_i + active_count_i] = i;
    		active_count_i++;
  	}
  }

  substream->pair_total_active_count += active_count_i;
  
  const int active_base_j = substream->pair_total_active_count;
  substream->pair_active_offsets_h[slot_j] = active_base_j;

  for (int i = 0; i < gcount_j; i++) {
  	if (cj_cache->active[i] > 0) {
    		substream->pair_active_index_h[active_base_j + active_count_j] = i;
    		active_count_j++;
  	}
  }

  substream->pair_total_active_count += active_count_j;

  substream->pair_active_counts_h[slot_i] = active_count_i;
  substream->pair_active_counts_h[slot_j] = active_count_j;
  if (active_count_i > substream->pair_max_active_count)
    substream->pair_max_active_count = active_count_i;
  if (active_count_j > substream->pair_max_active_count)
    substream->pair_max_active_count = active_count_j;

  //GPUEventRecord(startpack, stream);

  /* ---- Pack ci data into the send buffer ---- */
  {
    TIMER_TIC;
    for (int i = 0; i < gcount_i; i++) {
    	const int k = off_i + i;

	const float xi = ci_cache->x[i];
    	const float yi = ci_cache->y[i];
    	const float zi = ci_cache->z[i];
    	const float hi = ci_cache->epsilon[i];
    	const float mi = ci_cache->m[i];
    	const int   ai = ci_cache->active[i];

    	 gravity_gpu_values_send_pair[k].values_i.x = xi;
      gravity_gpu_values_send_pair[k].values_i.y = yi;
      gravity_gpu_values_send_pair[k].values_i.z = zi;
      gravity_gpu_values_send_pair[k].values_i.w = hi;

      gravity_gpu_values_send_pair[k].values_j.x = xi;
      gravity_gpu_values_send_pair[k].values_j.y = yi;
      gravity_gpu_values_send_pair[k].values_j.z = zi;
      gravity_gpu_values_send_pair[k].values_j.w = hi;

      gravity_gpu_values_send_pair[k].mass.x = mi;
      gravity_gpu_values_send_pair[k].mass.y = mi;
      gravity_gpu_values_send_pair[k].mass.z = 0.f;
      gravity_gpu_values_send_pair[k].mass.w = 0.f;

      gravity_gpu_values_send_pair[k].flags0.x = ai;
      gravity_gpu_values_send_pair[k].flags0.y = ai;
      gravity_gpu_values_send_pair[k].flags0.z = 0;
      gravity_gpu_values_send_pair[k].flags0.w = 0;

      gravity_gpu_values_send_pair[k].flags1.x = 0;
      gravity_gpu_values_send_pair[k].flags1.y = 0;
      gravity_gpu_values_send_pair[k].flags1.z = 0;
      gravity_gpu_values_send_pair[k].flags1.w = 0;
  	}

  for (int i = 0; i < gcount_j; i++) {
    const int k = off_j + i;
    
    const float xj = cj_cache->x[i];
    const float yj = cj_cache->y[i];
    const float zj = cj_cache->z[i];
    const float hj = cj_cache->epsilon[i];
    const float mj = cj_cache->m[i];
    const int   aj = cj_cache->active[i];

    gravity_gpu_values_send_pair[k].values_j.x = xj;
      gravity_gpu_values_send_pair[k].values_j.y = yj;
      gravity_gpu_values_send_pair[k].values_j.z = zj;
      gravity_gpu_values_send_pair[k].values_j.w = hj;

      gravity_gpu_values_send_pair[k].values_i.x = xj;
      gravity_gpu_values_send_pair[k].values_i.y = yj;
      gravity_gpu_values_send_pair[k].values_i.z = zj;
      gravity_gpu_values_send_pair[k].values_i.w = hj;

      gravity_gpu_values_send_pair[k].mass.x = mj;
      gravity_gpu_values_send_pair[k].mass.y = mj;
      gravity_gpu_values_send_pair[k].mass.z = 0.f;
      gravity_gpu_values_send_pair[k].mass.w = 0.f;

      gravity_gpu_values_send_pair[k].flags0.x = aj;
      gravity_gpu_values_send_pair[k].flags0.y = aj;
      gravity_gpu_values_send_pair[k].flags0.z = 0;
      gravity_gpu_values_send_pair[k].flags0.w = 0;

      gravity_gpu_values_send_pair[k].flags1.x = 0;
      gravity_gpu_values_send_pair[k].flags1.y = 0;
      gravity_gpu_values_send_pair[k].flags1.z = 0;
      gravity_gpu_values_send_pair[k].flags1.w = 0;
	}

    /*for (int i = 0; i < gcount_i; i++) {
    	const int k = off_i + i;
  	gravity_gpu_values_recv_pair[k].values_i.x = 0.0f;
  	gravity_gpu_values_recv_pair[k].values_i.y = 0.0f;
  	gravity_gpu_values_recv_pair[k].values_i.z = 0.0f;
  	gravity_gpu_values_recv_pair[k].values_i.w = 0.0f;

  	gravity_gpu_values_recv_pair[k].values_j.x = 0.0f;
  	gravity_gpu_values_recv_pair[k].values_j.y = 0.0f;
  	gravity_gpu_values_recv_pair[k].values_j.z = 0.0f;
  	gravity_gpu_values_recv_pair[k].values_j.w = 0.0f;
	}

    for (int i = 0; i < gcount_j; i++) {
	const int k = off_j + i;
	gravity_gpu_values_recv_pair[k].values_i.x = 0.0f;
  	gravity_gpu_values_recv_pair[k].values_i.y = 0.0f;
  	gravity_gpu_values_recv_pair[k].values_i.z = 0.0f;
  	gravity_gpu_values_recv_pair[k].values_i.w = 0.0f;

  	gravity_gpu_values_recv_pair[k].values_j.x = 0.0f;
  	gravity_gpu_values_recv_pair[k].values_j.y = 0.0f;
  	gravity_gpu_values_recv_pair[k].values_j.z = 0.0f;
  	gravity_gpu_values_recv_pair[k].values_j.w = 0.0f;
	}*/
    TIMER_TOC(timer_doself_grav_pp);
  }

  /* Store the address of the cells and tasks we are working on */
  grav_cells_pair[substream->grav_batch_pair_count] = ci;
  grav_cells_pair[substream->grav_batch_pair_count + 1] = cj;
  grav_tasks_pair[substream->grav_batch_pair_count / 2] = t;
  grav_pair_internal_from_self[substream->grav_batch_pair_count / 2] =
    (unsigned char)internal_from_self;

  gravity_gpu_values_send_pair[off_i].flags0.z = gcount_i;
  gravity_gpu_values_send_pair[off_j].flags0.z = gcount_j;

  gravity_gpu_values_send_pair[off_i].flags0.w = cell_is_active_gravity(ci, e);
  gravity_gpu_values_send_pair[off_j].flags0.w = cell_is_active_gravity(cj, e);
  
  gravity_gpu_values_send_pair[off_i].mass.z = rmax_i;
  gravity_gpu_values_send_pair[off_j].mass.z = rmax_j;

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
  gravity_gpu_values_send_pair[off_i].flags1.x = use_full;
  gravity_gpu_values_send_pair[off_j].flags1.x = use_full;

  /* Update that we packed a pair into our array */
  substream->grav_batch_pair_count += 2;

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
    struct runner *r, struct gpu_runner_substream *substream,
    struct gravity_gpu_values_send *gravity_gpu_values_send_pair,
    struct gravity_gpu_values_send *gravity_gpu_values_send_pair_d,
    struct gravity_gpu_values_recv *gravity_gpu_values_recv_pair,
    struct gravity_gpu_values_recv *gravity_gpu_values_recv_pair_d,
    struct cell **grav_cells_pair, struct task **grav_tasks_pair,
    struct task *current_task, int ncells, int max_cell_size,
    GPUStream stream){

  const int ncells_flush = substream->grav_batch_pair_count;
  if (ncells_flush == 0) return;

  /* Retrieve kernel parameters from the first pair in the batch. */
    const struct engine *e = r->e;
  const int periodic = e->mesh->periodic;
  const float r_s_inv = e->mesh->r_s_inv;
  const double min_trunc = e->mesh->r_cut_min;

  const float dim_0 = (float)e->mesh->dim[0];
  const float dim_1 = (float)e->mesh->dim[1];
  const float dim_2 = (float)e->mesh->dim[2];
  
  const int pair_capacity = r->gpu.grav_batch_ncells * r->gpu.grav_max_cell_size;

  /* ---- H2D copy ---- */
  {
    TIMER_TIC;

    const int nslots = substream->grav_batch_pair_count;
    const int total = substream->pair_total_count;
    
    if (nslots < 0 || nslots > r->gpu.grav_batch_ncells)
    error("Bad nslots in pair flush: %d (capacity %d)",
          nslots, r->gpu.grav_batch_ncells);

  if (total < 0 || total > pair_capacity)
    error("Bad pair_total_count in pair flush: %d (capacity %d)",
          total, pair_capacity);

	GPUMemcpyAsync(
	    substream->pair_counts_d,
	    substream->pair_counts_h,
	    nslots * sizeof(int),
	    GPU_MEMCPY_HOST_TO_DEVICE, stream);

	GPUMemcpyAsync(
	    substream->pair_offsets_d,
	    substream->pair_offsets_h,
	    nslots * sizeof(int),
	    GPU_MEMCPY_HOST_TO_DEVICE, stream);

	GPUMemcpyAsync(
	    substream->pair_active_counts_d,
	    substream->pair_active_counts_h,
	    nslots * sizeof(int),
	    GPU_MEMCPY_HOST_TO_DEVICE, stream);

	GPUMemcpyAsync(
    		substream->pair_active_offsets_d,
    		substream->pair_active_offsets_h,
    		nslots * sizeof(int),
    		GPU_MEMCPY_HOST_TO_DEVICE, stream);

	GPUMemcpyAsync(
    		substream->pair_active_index_d,
    		substream->pair_active_index_h,
    		(size_t)substream->pair_total_active_count * sizeof(int),
    		GPU_MEMCPY_HOST_TO_DEVICE, stream);

	GPUMemcpyAsync(
	    gravity_gpu_values_send_pair_d,
	    gravity_gpu_values_send_pair,
	    total * sizeof(struct gravity_gpu_values_send),
	    GPU_MEMCPY_HOST_TO_DEVICE, stream);
	    
	GPUMemsetAsync(
	    substream->recv_pair_active_d,
	    0,
	    (size_t)substream->pair_total_active_count *
		sizeof(struct gravity_gpu_values_recv),
	    stream);

    GPUError err2 = GPUGetLastError();
    if (err2 != GPU_SUCCESS)
      printf("Error (flush H2D): %s\n", GPUGetErrorString(err2));

    /* ---- Kernel launch ---- */
    pair_pp_offload_new(
    periodic, min_trunc, &r_s_inv,
    substream->pair_counts_d,
    substream->pair_offsets_d,
    substream->pair_active_counts_d,
    substream->pair_active_offsets_d,
    substream->pair_active_index_d,
    dim_0, dim_1, dim_2,
    gravity_gpu_values_send_pair_d,
    substream->recv_pair_active_d,
    ncells_flush, max_cell_size,
    substream->pair_max_active_count, stream);

    /* ---- D2H copy ---- */
    GPUMemcpyAsync(
	    substream->recv_pair_active,
	    substream->recv_pair_active_d,
	    (size_t)substream->pair_total_active_count *
		sizeof(struct gravity_gpu_values_recv),
	    GPU_MEMCPY_DEVICE_TO_HOST,
	    stream);

    GPUEventRecord(substream->done, substream->stream);
    GPUEventSynchronize(substream->done);

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

      struct cell *ci_pair = grav_cells_pair[j];
      struct cell *cj_pair = grav_cells_pair[j + 1];
      struct cell *a_pair = ci_pair, *b_pair = cj_pair;

      if (a_pair > b_pair) {
        struct cell *tmp = a_pair;
        a_pair = b_pair;
        b_pair = tmp;
      }

      while (cell_glocktree(a_pair)) {
        ;
      }
      const int count_i = substream->pair_counts_h[j];
      const int off_i = substream->pair_offsets_h[j];

      const int active_count_i = substream->pair_active_counts_h[j];
	const int active_base_i = substream->pair_active_offsets_h[j];

	for (int a = 0; a < active_count_i; a++) {
	  const int local_pid = substream->pair_active_index_h[active_base_i + a];
	  const int k = active_base_i + a;

	  ci_pair->grav.parts[local_pid].a_grav[0] +=
	      substream->recv_pair_active[k].values_i.x;
	  ci_pair->grav.parts[local_pid].a_grav[1] +=
	      substream->recv_pair_active[k].values_i.y;
	  ci_pair->grav.parts[local_pid].a_grav[2] +=
	      substream->recv_pair_active[k].values_i.z;
	  ci_pair->grav.parts[local_pid].potential +=
	      substream->recv_pair_active[k].values_i.w;
	}
	
      cell_gunlocktree(a_pair);

      while (cell_glocktree(b_pair)) {
        ;
      }
     const int count_j = substream->pair_counts_h[j + 1];
     const int off_j = substream->pair_offsets_h[j + 1];

     const int active_count_j = substream->pair_active_counts_h[j + 1];
	const int active_base_j = substream->pair_active_offsets_h[j + 1];

	for (int a = 0; a < active_count_j; a++) {
	  const int local_pid = substream->pair_active_index_h[active_base_j + a];
	  const int k = active_base_j + a;

	  cj_pair->grav.parts[local_pid].a_grav[0] +=
	      substream->recv_pair_active[k].values_i.x;
	  cj_pair->grav.parts[local_pid].a_grav[1] +=
	      substream->recv_pair_active[k].values_i.y;
	  cj_pair->grav.parts[local_pid].a_grav[2] +=
	      substream->recv_pair_active[k].values_i.z;
	  cj_pair->grav.parts[local_pid].potential +=
	      substream->recv_pair_active[k].values_i.w;
	}
	
	#ifdef SWIFT_DEBUG_CHECKS
      for (int i = 0; i < count_j; i++) {
        for (int p = 0; p < count_i; p++) {
          accumulate_inc_ll(&cj_pair->grav.parts[i].num_interacted);
        	}
      	}
      	
      	for (int i = 0; i < count_i; i++) {
        for (int p = 0; p < count_j; p++) {
          accumulate_inc_ll(&ci_pair->grav.parts[i].num_interacted);
        	}
      	}
	#endif
      cell_gunlocktree(b_pair);
    }

    TIMER_TOC(timer_doself_grav_pp);
  }

  /* Complete any tasks from previous top-level calls that were packed
     into this batch, but skip current_task (still being walked). */
    struct scheduler *sched = &r->e->sched;
  struct task *prev_task = NULL;

  for (int j = 0; j < ncells_flush; j += 2) {
    const int pair_slot = j / 2;
    struct task *batch_task = grav_tasks_pair[pair_slot];
    const int internal = substream->grav_pair_internal_from_self[pair_slot];

    if (batch_task == NULL)
      error("Unexpected NULL task in pair GPU batch at slot %d", pair_slot);

    if (!internal &&
        batch_task != prev_task &&
        batch_task != current_task &&
        !batch_task->gpu_completed) {

      if (batch_task->type == task_type_self) {
        runner_gpu_complete_self_task(r, sched, batch_task);
      } else if (batch_task->type == task_type_pair) {
        runner_gpu_complete_pair_task(r, sched, batch_task);
      } else {
        error("Unexpected task type in pair GPU batch: %d", batch_task->type);
      }

      batch_task->gpu_completed = 1;
      prev_task = batch_task;
    }

    grav_cells_pair[j] = NULL;
    grav_cells_pair[j + 1] = NULL;
    grav_tasks_pair[pair_slot] = NULL;
    substream->grav_pair_internal_from_self[pair_slot] = 0;
  }

  substream->grav_batch_pair_count = 0;
  substream->busy = 0;

  for (int j = 0; j < ncells_flush; j++) {
    substream->pair_counts_h[j] = 0;
    substream->pair_offsets_h[j] = 0;
    substream->pair_active_counts_h[j] = 0;
  }
  substream->pair_total_count = 0;
  substream->pair_max_active_count = 0;
  substream->pair_total_active_count = 0;
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
    struct runner *r, struct gpu_runner_substream *substream, struct cell *ci,
    struct cell *cj, const int symmetric, const int allow_mpole,
    struct gravity_gpu_values_send *gravity_gpu_values_send_pair,
    struct gravity_gpu_values_send *gravity_gpu_values_send_pair_d,
    struct gravity_gpu_values_recv *gravity_gpu_values_recv_pair,
    struct gravity_gpu_values_recv *gravity_gpu_values_recv_pair_d,
    struct cell **grav_cells_pair, struct task **grav_tasks_pair,
    unsigned char *grav_pair_internal_from_self,
    struct task *t, int internal_from_self,
    int ncells, int max_cell_size, GPUStream stream) {

  /* Need 2 free slots for one pair */
  if (substream->grav_batch_pair_count + 2 > ncells) {
    runner_dopair_grav_pp_flush(
        r, substream,
        gravity_gpu_values_send_pair, gravity_gpu_values_send_pair_d,
        gravity_gpu_values_recv_pair, gravity_gpu_values_recv_pair_d,
        grav_cells_pair, grav_tasks_pair,
        t, ncells, max_cell_size, stream);
  }

  runner_dopair_grav_pp_pack(
      r, substream, ci, cj, symmetric, allow_mpole,
      gravity_gpu_values_send_pair,
      gravity_gpu_values_recv_pair,
      grav_cells_pair, grav_tasks_pair,
      grav_pair_internal_from_self,
      t, internal_from_self,
      max_cell_size, stream);

  if (substream->grav_batch_pair_count >= ncells) {
    runner_dopair_grav_pp_flush(
        r, substream,
        gravity_gpu_values_send_pair, gravity_gpu_values_send_pair_d,
        gravity_gpu_values_recv_pair, gravity_gpu_values_recv_pair_d,
        grav_cells_pair, grav_tasks_pair,
        t, ncells, max_cell_size, stream);
    return flushed_pair_task;
  }

  return packed_task;
}

static void runner_doself_grav_pp_flush(
    struct runner *r,
    struct gpu_runner_substream *substream,
    int nslots,
    int max_active_count,
    int max_cell_size,
    GPUStream stream) {

  const struct engine *e = r->e;
  const int periodic = e->mesh->periodic;
  const float r_s_inv = e->mesh->r_s_inv;
  const double min_trunc = e->mesh->r_cut_min;

  self_pp_offload_new(periodic, substream->self_rmax_d, min_trunc, &r_s_inv,
                    substream->self_counts_d,
                    substream->self_offsets_d,
                    substream->self_active_counts_d,
                    substream->self_active_offsets_d,
                    substream->self_active_index_d,
                    substream->send_self_d,
                    substream->recv_self_active_d,
                    nslots, max_cell_size, max_active_count, stream);
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
  enum runner_gpu_task_type runner_doself_grav_pp_task_new(
    struct runner *r,
    struct gpu_runner_substream *substream,
    struct cell *ci,
    struct task *t,
    int ncells,
    int max_cell_size) {

  const struct engine *e = r->e;
  struct gravity_cache *const ci_cache = &r->ci_gravity_cache;

  const int gcount = ci->grav.count;
  const int gcount_padded = gcount - (gcount % VEC_SIZE) + VEC_SIZE;

  if (gcount > max_cell_size)
    error("More particles than allocated memory!");

  const int slot = substream->grav_batch_self_count;

  const double loc[3] = {
      ci->loc[0] + 0.5 * ci->width[0],
      ci->loc[1] + 0.5 * ci->width[1],
      ci->loc[2] + 0.5 * ci->width[2]};

  gravity_cache_populate_no_mpole(
      e->max_active_bin, ci_cache,
      ci->grav.parts,
      gcount, gcount_padded,
      loc, ci,
      e->gravity_properties);

  while (cell_glocktree(ci)) {
    ;
  }

  /*record packed offset/count */
  append_packed_self_cell(substream, slot, gcount);
  substream->self_rmax_h[slot] = 2.f * ci->grav.multipole->r_max;
  const int offset = substream->self_offsets_h[slot];

  /* Build compact active-target list for this cell. */
  int active_count = 0;
  const int active_base = substream->self_total_active_count;
  substream->self_active_offsets_h[slot] = active_base;

  for (int i = 0; i < gcount; i++) {
  	if (ci_cache->active[i] > 0) {
    		substream->self_active_index_h[active_base + active_count] = i;
    		active_count++;
  	}
  }

  substream->self_total_active_count += active_count;
  
  substream->self_active_counts_h[slot] = active_count;
  if (active_count > substream->self_max_active_count)
    substream->self_max_active_count = active_count;

  /* Pack contiguously */
  for (int i = 0; i < gcount; i++) {
    const int k = offset + i;

    substream->send_self[k].values_i.w = ci_cache->epsilon[i];
    substream->send_self[k].values_j.w = ci_cache->epsilon[i];

    substream->send_self[k].mass.x = ci_cache->m[i];
    substream->send_self[k].mass.y = ci_cache->m[i];

    substream->send_self[k].values_i.x = ci_cache->x[i];
    substream->send_self[k].values_j.x = ci_cache->x[i];
    substream->send_self[k].values_i.y = ci_cache->y[i];
    substream->send_self[k].values_j.y = ci_cache->y[i];
    substream->send_self[k].values_i.z = ci_cache->z[i];
    substream->send_self[k].values_j.z = ci_cache->z[i];

    substream->send_self[k].flags0.x = ci_cache->active[i];
    substream->send_self[k].flags0.y = ci_cache->active[i];

    substream->send_self[k].flags0.w = cell_is_active_gravity(ci, e);
    substream->send_self[k].flags0.z = gcount;
  }

  /* Zero only the live recv span, not max_cell_size */
  /*for (int i = 0; i < gcount; i++) {
    const int k = offset + i;
    substream->recv_self[k].values_i.x = 0.0f;
    substream->recv_self[k].values_i.y = 0.0f;
    substream->recv_self[k].values_i.z = 0.0f;
    substream->recv_self[k].values_i.w = 0.0f;
  }*/
  
  //printf("PACKED:%i  \n", substream->grav_batch_self_count);

  substream->grav_cells_self[slot] = ci;
  substream->grav_tasks_self[slot] = t;
  substream->grav_batch_self_count++;
  substream->self_rmax_h[slot] = 2.f * ci->grav.multipole->r_max;

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

  /* ===================== FLUSH ===================== */

    if (substream->grav_batch_self_count >= ncells) {

    const int nslots = substream->grav_batch_self_count;
    const int total = substream->self_total_count;

    /* copy packed metadata */
    GPUMemcpyAsync(
        substream->self_counts_d,
        substream->self_counts_h,
        nslots * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        substream->stream);

    GPUMemcpyAsync(
        substream->self_offsets_d,
        substream->self_offsets_h,
        nslots * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        substream->stream);

    GPUMemcpyAsync(
        substream->self_active_counts_d,
        substream->self_active_counts_h,
        nslots * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        substream->stream);

    GPUMemcpyAsync(
    substream->self_active_offsets_d,
    substream->self_active_offsets_h,
    nslots * sizeof(int),
    GPU_MEMCPY_HOST_TO_DEVICE,
    substream->stream);

    GPUMemcpyAsync(
    substream->self_active_index_d,
    substream->self_active_index_h,
    (size_t)substream->self_total_active_count * sizeof(int),
    GPU_MEMCPY_HOST_TO_DEVICE,
    substream->stream);
        
    GPUMemcpyAsync(
    	substream->self_rmax_d,
    	substream->self_rmax_h,
    	nslots * sizeof(float),
    	GPU_MEMCPY_HOST_TO_DEVICE,
    	substream->stream);

    /* H2D: only live data */
    GPUMemcpyAsync(
        substream->send_self_d,
        substream->send_self,
        total * sizeof(struct gravity_gpu_values_send),
        GPU_MEMCPY_HOST_TO_DEVICE,
        substream->stream);

    GPUMemsetAsync(
	    substream->recv_self_active_d,
	    0,
	    (size_t)substream->self_total_active_count *
		sizeof(struct gravity_gpu_values_recv),
	    substream->stream);

    /* kernel */
    runner_doself_grav_pp_flush(
    r, substream, nslots, substream->self_max_active_count, max_cell_size, substream->stream);

    /* D2H: only live data */
    GPUMemcpyAsync(
	    substream->recv_self_active,
	    substream->recv_self_active_d,
	    (size_t)substream->self_total_active_count *
		sizeof(struct gravity_gpu_values_recv),
	    GPU_MEMCPY_DEVICE_TO_HOST,
	    substream->stream);

    GPUEventRecord(substream->done, substream->stream);
    GPUEventSynchronize(substream->done);

    /* ===================== UNPACK ===================== */

    for (int j = 0; j < nslots; j++) {

      struct cell *c_unpack = substream->grav_cells_self[j];
      const int count = substream->self_counts_h[j];
      const int offset = substream->self_offsets_h[j];

      while (cell_glocktree(c_unpack)) {
        ;
      }

      for (int i = 0; i < count; i++) {
        const int k = offset + i;

        c_unpack->grav.parts[i].a_grav[0] += substream->recv_self[k].values_i.x;
        c_unpack->grav.parts[i].a_grav[1] += substream->recv_self[k].values_i.y;
        c_unpack->grav.parts[i].a_grav[2] += substream->recv_self[k].values_i.z;
        c_unpack->grav.parts[i].potential += substream->recv_self[k].values_i.w;
      }

      cell_gunlocktree(c_unpack);
    }

    /* ===================== COMPLETE TASKS ===================== */

    runner_gpu_complete_self_batch(r, &r->e->sched, substream);

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
    struct runner *r, struct gpu_runner_substream *substream, struct cell *ci,
    struct cell *cj, const int gettimer,
    struct gravity_gpu_values_send *gravity_gpu_values_send_pair,
    struct gravity_gpu_values_send *gravity_gpu_values_send_pair_d,
    struct gravity_gpu_values_recv *gravity_gpu_values_recv_pair,
    struct gravity_gpu_values_recv *gravity_gpu_values_recv_pair_d,
    struct cell **grav_cells_pair, struct task **grav_tasks_pair,
    unsigned char *grav_pair_internal_from_self,
    struct task *t, int internal_from_self,
    int ncells, int max_cell_size, GPUStream stream) {

  if (ci == NULL || cj == NULL)
    error("runner_dopair_recursive_grav_new got NULL cell");

  const struct engine *e = r->e;

  if (!cell_are_gpart_drifted(ci, e))
    cell_drift_gpart(ci, e, /*force=*/1, /*init_particles=*/0, NULL);
  if (!cell_are_gpart_drifted(cj, e))
    cell_drift_gpart(cj, e, /*force=*/1, /*init_particles=*/0, NULL);

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
  struct gravity_tensors *const multi_i = ci->grav.multipole;
  struct gravity_tensors *const multi_j = cj->grav.multipole;

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
    r, substream, ci, cj, 1, 1,
    substream->send_pair,
    substream->send_pair_d,
    substream->recv_pair,
    substream->recv_pair_d,
    substream->grav_cells_pair,
    substream->grav_tasks_pair,
    substream->grav_pair_internal_from_self,
    t, internal_from_self,
    ncells, max_cell_size, substream->stream);

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
        		r, substream, ci->progeny[k], cj, 0,
        		substream->send_pair, substream->send_pair_d,
        		substream->recv_pair, substream->recv_pair_d,
        		substream->grav_cells_pair, substream->grav_tasks_pair,
        		substream->grav_pair_internal_from_self,
        		t, internal_from_self, ncells, max_cell_size, substream->stream);
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
                    r, substream, ci, cj->progeny[k], 0,
		    substream->send_pair, substream->send_pair_d,
		    substream->recv_pair, substream->recv_pair_d,
		    substream->grav_cells_pair, substream->grav_tasks_pair,
		    substream->grav_pair_internal_from_self,
		    t, internal_from_self, ncells, max_cell_size, substream->stream);
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
                    r, substream, ci, cj->progeny[k], 0,
		    substream->send_pair, substream->send_pair_d,
		    substream->recv_pair, substream->recv_pair_d,
		    substream->grav_cells_pair, substream->grav_tasks_pair,
		    substream->grav_pair_internal_from_self,
		    t, internal_from_self, ncells, max_cell_size, substream->stream);
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
                    r, substream, ci->progeny[k], cj, 0,
		    substream->send_pair, substream->send_pair_d,
		    substream->recv_pair, substream->recv_pair_d,
		    substream->grav_cells_pair, substream->grav_tasks_pair,
		    substream->grav_pair_internal_from_self,
		    t, internal_from_self, ncells, max_cell_size, substream->stream);
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
      if (substream->grav_batch_pair_count > 0) {
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
 * @brief Choose number of cells in a batch
 */
static int runner_gpu_choose_batch_ncells(const struct engine *e,
                                          int max_cell_size) {
                                          
  /* User override (<=0 means "not set") */
  const int user_ncells = e->ncells_per_gpu_grav_pack;

  size_t free_bytes = 0, total_bytes = 0;
  GPUMemGetInfo(&free_bytes, &total_bytes);

  /* Leave headroom for CUDA/HIP context/runtime overhead */
  const double usable_fraction = 0.70;
  const size_t usable_bytes = (size_t)(free_bytes * usable_fraction);
  
  const int nstreams = parser_get_opt_param_int(
      e->parameter_file, "GPU:nstreams", 2); //try changing this to 1 and see if it works better?

  const size_t bytes_per_cell_per_substream =
      (size_t)max_cell_size *
      (2 * sizeof(struct gravity_gpu_values_send) +
       2 * sizeof(struct gravity_gpu_values_recv));

  const size_t bytes_per_cell_per_runner =
      (size_t)nstreams * bytes_per_cell_per_substream;

  /* Replace with the actual number of runners that allocate GPU buffers */
  const int nr_gpu_runners = e->nr_threads > 0 ? e->nr_threads : 1;

  const size_t metadata = 64ULL * 1024ULL * 1024ULL; /* 64 MB */

  size_t budget = 0;
  if (usable_bytes > metadata)
    budget = usable_bytes - metadata;

  const size_t per_runner_budget = budget / (size_t)nr_gpu_runners;

  int ncells = (int)(per_runner_budget / bytes_per_cell_per_runner);

  if (ncells < 2) ncells = 2;
  if (ncells > 2048) ncells = 2048;

  if (user_ncells > 0) {

    if (user_ncells > ncells) {
      if (e->verbose) {
        message("GPU:ncells_per_gpu_grav_pack=%d too large, limiting to %d",
                user_ncells, ncells);
      }
      return ncells;
    }

    return user_ncells;
  }

  return ncells;
}


/**
 * @brief Choose number of streams per runner
 */
static int runner_gpu_choose_nstreams(const struct engine *e,
                                      int max_cell_size,
                                      int ncells) {

  size_t free_bytes = 0, total_bytes = 0;
  GPUMemGetInfo(&free_bytes, &total_bytes);

  /* Leave headroom for other allocations */
  const double usable_fraction = 0.70;
  const size_t usable_bytes = (size_t)(free_bytes * usable_fraction);

  const int nr_gpu_runners = e->nr_threads > 0 ? e->nr_threads : 1;
  const size_t metadata = 64ULL * 1024ULL * 1024ULL; /* 64 MB */

  size_t budget = 0;
  if (usable_bytes > metadata)
    budget = usable_bytes - metadata;

  const size_t per_runner_budget = budget / (size_t)nr_gpu_runners;

  const size_t bytes_per_substream =
      (size_t)ncells * (size_t)max_cell_size *
      (2 * sizeof(struct gravity_gpu_values_send) +
       2 * sizeof(struct gravity_gpu_values_recv));

  if (bytes_per_substream == 0)
    return 1;

  int max_safe_nstreams = (int)(per_runner_budget / bytes_per_substream);

  if (max_safe_nstreams < 1) max_safe_nstreams = 1;

  /* Profiling/tuning usually stops helping after a few streams */
  //if (max_safe_nstreams > 4) max_safe_nstreams = 4;

  return max_safe_nstreams;
}

/**
 * @brief Initialise the GPU-specific state attached to a runner.
 *
 * @param r The runner whose GPU state to initialise.
 */
void runner_gpu_init(struct runner *r) {

  struct gpu_runner *gpu = &r->gpu;
  struct engine *e = r->e;

  GPUSetDevice(0);

  GPUDeviceProp prop;
  GPUGetDeviceProperties(&prop, 0);

  const int max_cell_size = space_subsize_self_grav + 100;

  gpu->grav_max_cell_size = max_cell_size;
  gpu->grav_batch_ncells =
      runner_gpu_choose_batch_ncells(e, gpu->grav_max_cell_size);

  /* Pair batches consume 2 slots at a time */
  if (gpu->grav_batch_ncells % 2 != 0)
    gpu->grav_batch_ncells--;

  if (gpu->grav_batch_ncells < 2)
    gpu->grav_batch_ncells = 2;

  /* User may request nstreams, otherwise auto-pick based on ncells/max_cell_size */
  const int user_nstreams = parser_get_opt_param_int(
      e->parameter_file, "GPU:nstreams", 1);

  const int auto_nstreams =
      runner_gpu_choose_nstreams(e,
                                 gpu->grav_max_cell_size,
                                 gpu->grav_batch_ncells);

  if (user_nstreams > 0) {
    gpu->nstreams = user_nstreams;
    if (gpu->nstreams > auto_nstreams) {
      if (r->id == 0) {
        message("GPU:nstreams=%d too large for current ncells/max_cell_size, "
                "limiting to %d",
                user_nstreams, auto_nstreams);
      }
      gpu->nstreams = auto_nstreams;
    }
  } else {
    gpu->nstreams = auto_nstreams;
  }

  if (gpu->nstreams < 1) gpu->nstreams = 1;
  if (gpu->nstreams > 8) gpu->nstreams = 8;

  gpu->substreams = malloc((size_t)gpu->nstreams *
                           sizeof(struct gpu_runner_substream));
  if (gpu->substreams == NULL)
    error("Failed to allocate GPU substreams");

  const size_t send_bytes =
      (size_t)gpu->grav_batch_ncells *
      (size_t)gpu->grav_max_cell_size *
      sizeof(struct gravity_gpu_values_send);

  const size_t recv_bytes =
      (size_t)gpu->grav_batch_ncells *
      (size_t)gpu->grav_max_cell_size *
      sizeof(struct gravity_gpu_values_recv);

  const size_t bytes_per_substream =
      2 * send_bytes + 2 * recv_bytes; /* self + pair */

  size_t free_bytes = 0, total_bytes = 0;
  GPUMemGetInfo(&free_bytes, &total_bytes);

  if (r->id == 0) {
    message("GPU device: %s", prop.name);
    message("GPU free memory: %.2f GB",
            free_bytes / (1024.0 * 1024.0 * 1024.0));
    message("GPU total memory: %.2f GB",
            total_bytes / (1024.0 * 1024.0 * 1024.0));
    message("Max cell size: %i", gpu->grav_max_cell_size);
    message("ncells per pack: %i", gpu->grav_batch_ncells);
    message("Streams per runner: %i", gpu->nstreams);
    message("Per-substream buffer bytes: %zu", bytes_per_substream);
  }

  gpu->next_substream = 0;

  for (int i = 0; i < gpu->nstreams; i++) {

    struct gpu_runner_substream *substream = &gpu->substreams[i];

    GPUStreamCreateWithFlags(&substream->stream, GPUStreamNonBlocking);
    GPUEventCreate(&substream->done);
    substream->busy = 0;

    /* ---------- Pair state ---------- */

    substream->grav_batch_pair_count = 0;

    GPUMalloc((void **)&substream->send_pair_d, send_bytes);
    GPUHostMalloc((void **)&substream->send_pair, send_bytes);

    GPUMalloc((void **)&substream->recv_pair_d, recv_bytes);
    GPUHostMalloc((void **)&substream->recv_pair, recv_bytes);
    
    const size_t active_recv_bytes =
    (size_t)gpu->grav_batch_ncells *
    (size_t)gpu->grav_max_cell_size *
    sizeof(struct gravity_gpu_values_recv);

    GPUMalloc((void **)&substream->recv_pair_active_d, active_recv_bytes);
    GPUHostMalloc((void **)&substream->recv_pair_active, active_recv_bytes);

    substream->grav_cells_pair =
        malloc((size_t)gpu->grav_batch_ncells * sizeof(struct cell *));
    substream->grav_tasks_pair =
        malloc((size_t)gpu->grav_batch_ncells * sizeof(struct task *));
    substream->grav_pair_internal_from_self =
        malloc((size_t)gpu->grav_batch_ncells * sizeof(unsigned char));

    if (substream->grav_pair_internal_from_self != NULL) {
      memset(substream->grav_pair_internal_from_self, 0,
             (size_t)gpu->grav_batch_ncells * sizeof(unsigned char));
    }

    substream->pair_total_count = 0;
    substream->pair_max_active_count = 0;
    substream->pair_total_active_count = 0;

    substream->pair_counts_h =
        malloc((size_t)gpu->grav_batch_ncells * sizeof(int));
    substream->pair_offsets_h =
        malloc((size_t)gpu->grav_batch_ncells * sizeof(int));
    substream->pair_active_counts_h =
    	malloc((size_t)gpu->grav_batch_ncells * sizeof(int));
    substream->pair_active_offsets_h =
    	malloc((size_t)gpu->grav_batch_ncells * sizeof(int));
    substream->pair_active_index_h =
    	malloc((size_t)gpu->grav_batch_ncells *
           (size_t)gpu->grav_max_cell_size * sizeof(int));

    GPUMalloc((void **)&substream->pair_counts_d,
              (size_t)gpu->grav_batch_ncells * sizeof(int));
    GPUMalloc((void **)&substream->pair_offsets_d,
              (size_t)gpu->grav_batch_ncells * sizeof(int));
    GPUMalloc((void **)&substream->pair_active_counts_d,
          (size_t)gpu->grav_batch_ncells * sizeof(int));
    GPUMalloc((void **)&substream->pair_active_offsets_d,
          (size_t)gpu->grav_batch_ncells * sizeof(int));
    GPUMalloc((void **)&substream->pair_active_index_d,
          (size_t)gpu->grav_batch_ncells *
          (size_t)gpu->grav_max_cell_size * sizeof(int));

    for (int j = 0; j < gpu->grav_batch_ncells; j++) {
      substream->pair_counts_h[j] = 0;
      substream->pair_offsets_h[j] = 0;
      substream->pair_active_counts_h[j] = 0;
      substream->pair_active_offsets_h[j] = 0;
    }

    /* ---------- Self state ---------- */

    substream->grav_batch_self_count = 0;

    GPUMalloc((void **)&substream->send_self_d, send_bytes);
    GPUHostMalloc((void **)&substream->send_self, send_bytes);

    GPUMalloc((void **)&substream->recv_self_d, recv_bytes);
    GPUHostMalloc((void **)&substream->recv_self, recv_bytes);
    
    GPUMalloc((void **)&substream->recv_self_active_d, active_recv_bytes);
    GPUHostMalloc((void **)&substream->recv_self_active, active_recv_bytes);

    substream->grav_cells_self =
        malloc((size_t)gpu->grav_batch_ncells * sizeof(struct cell *));
    substream->grav_tasks_self =
        malloc((size_t)gpu->grav_batch_ncells * sizeof(struct task *));

    substream->self_total_count = 0;
    substream->self_max_active_count = 0;
    substream->self_total_active_count = 0;

    substream->self_counts_h =
        malloc((size_t)gpu->grav_batch_ncells * sizeof(int));
    substream->self_offsets_h =
        malloc((size_t)gpu->grav_batch_ncells * sizeof(int));
    substream->self_active_counts_h =
        malloc((size_t)gpu->grav_batch_ncells * sizeof(int));
    substream->self_active_offsets_h =
    	malloc((size_t)gpu->grav_batch_ncells * sizeof(int));
    substream->self_active_index_h =
    	malloc((size_t)gpu->grav_batch_ncells *
           (size_t)gpu->grav_max_cell_size * sizeof(int));
    substream->self_rmax_h =
        malloc((size_t)gpu->grav_batch_ncells * sizeof(float));

    GPUMalloc((void **)&substream->self_rmax_d,
              (size_t)gpu->grav_batch_ncells * sizeof(float));
    GPUMalloc((void **)&substream->self_counts_d,
              (size_t)gpu->grav_batch_ncells * sizeof(int));
    GPUMalloc((void **)&substream->self_offsets_d,
              (size_t)gpu->grav_batch_ncells * sizeof(int));
    GPUMalloc((void **)&substream->self_active_counts_d,
          (size_t)gpu->grav_batch_ncells * sizeof(int));
    GPUMalloc((void **)&substream->self_active_offsets_d,
          (size_t)gpu->grav_batch_ncells * sizeof(int));
    GPUMalloc((void **)&substream->self_active_index_d,
          (size_t)gpu->grav_batch_ncells *
          (size_t)gpu->grav_max_cell_size * sizeof(int));

    for (int j = 0; j < gpu->grav_batch_ncells; j++) {
      substream->self_counts_h[j] = 0;
      substream->self_offsets_h[j] = 0;
      substream->self_active_counts_h[j] = 0;
      substream->self_active_offsets_h[j] = 0;
      substream->self_rmax_h[j] = 0.f;
    }

    /* ---------- checks ---------- */

    if (substream->grav_cells_pair == NULL ||
        substream->grav_tasks_pair == NULL ||
        substream->grav_pair_internal_from_self == NULL ||
        substream->pair_counts_h == NULL ||
        substream->pair_offsets_h == NULL ||
        substream->pair_active_counts_h == NULL ||
        substream->pair_active_index_h == NULL)
      error("Failed to allocate runner GPU pair substream metadata arrays.");

    if (substream->grav_cells_self == NULL ||
        substream->grav_tasks_self == NULL ||
        substream->self_counts_h == NULL ||
        substream->self_offsets_h == NULL ||
        substream->self_active_counts_h == NULL ||
        substream->self_active_index_h == NULL ||
        substream->self_rmax_h == NULL)
      error("Failed to allocate runner GPU self substream metadata arrays.");
  }

  const GPUError err = GPUGetLastError();
  if (err != GPU_SUCCESS)
    error("runner_gpu_init failed: %s", GPUGetErrorString(err));
}

/**
 * @brief Acquire the substream for the GPU work to be launched to
 *
 * @param r The runner
 */
struct gpu_runner_substream *
runner_gpu_acquire_substream(struct runner *r) {

  struct gpu_runner *gpu = &r->gpu;

  for (int i = 0; i < gpu->nstreams; i++) {
    int idx = (gpu->next_substream + i) % gpu->nstreams;
    struct gpu_runner_substream *substream = &gpu->substreams[idx];

    if (!substream->busy ||
        GPUEventQuery(substream->done) == GPU_SUCCESS) {

      substream->busy = 1;
      gpu->next_substream = (idx + 1) % gpu->nstreams;
      return substream;
    }
  }

  /* fallback: wait */
  struct gpu_runner_substream *substream = &gpu->substreams[gpu->next_substream];
  GPUEventSynchronize(substream->done);
  substream->busy = 1;

  gpu->next_substream = (gpu->next_substream + 1) % gpu->nstreams;
  return substream;
}

/**
 * @brief Flush all the substreams for the pair tasks
 *
 * @param r The runner whose GPU state to clean.
 */
static inline void runner_gpu_flush_all_pair_substreams(struct runner *r) {
  for (int i = 0; i < r->gpu.nstreams; i++) {
    struct gpu_runner_substream *substream = &r->gpu.substreams[i];

    if (substream->grav_batch_pair_count > 0) {
      runner_dopair_grav_pp_flush(
          r, substream,
          substream->send_pair, substream->send_pair_d,
          substream->recv_pair, substream->recv_pair_d,
          substream->grav_cells_pair, substream->grav_tasks_pair,
          NULL,
          r->gpu.grav_batch_ncells,
          r->gpu.grav_max_cell_size,
          substream->stream);
    }
  }
}

/**
 * @brief Clean the GPU-specific state attached to a runner.
 *
 * @param r The runner whose GPU state to clean.
 */
void runner_gpu_clean(struct runner *r) {

  struct gpu_runner *gpu = &r->gpu;

  for (int i = 0; i < gpu->nstreams; i++) {

    struct gpu_runner_substream *substream = &gpu->substreams[i];

    /* Pair buffers */
    GPUFreeHost(substream->send_pair);
    GPUFree(substream->send_pair_d);
    GPUFreeHost(substream->recv_pair);
    GPUFree(substream->recv_pair_d);

    free(substream->grav_cells_pair);
    free(substream->grav_tasks_pair);
    free(substream->grav_pair_internal_from_self);
    
    free(substream->pair_counts_h);
    free(substream->pair_offsets_h);
    free(substream->pair_active_counts_h);
    free(substream->pair_active_index_h);
    GPUFree(substream->pair_counts_d);
    GPUFree(substream->pair_offsets_d);
    GPUFree(substream->pair_active_counts_d);
    GPUFree(substream->pair_active_index_d);
    
    GPUFreeHost(substream->recv_pair_active);
    GPUFree(substream->recv_pair_active_d);

    /* Self buffers */
    GPUFreeHost(substream->send_self);
    GPUFree(substream->send_self_d);
    GPUFreeHost(substream->recv_self);
    GPUFree(substream->recv_self_d);

    free(substream->grav_cells_self);
    free(substream->grav_tasks_self);
    
    free(substream->self_counts_h);
    free(substream->self_offsets_h);
    GPUFree(substream->self_counts_d);
    GPUFree(substream->self_offsets_d);
    
    GPUFreeHost(substream->recv_self_active);
    GPUFree(substream->recv_self_active_d);
    
    free(substream->self_rmax_h);
    GPUFree(substream->self_rmax_d);

    /* Stream/event */
    GPUEventDestroy(substream->done);
    GPUStreamDestroy(substream->stream);

    /* Reset */
    substream->send_pair = NULL;
    substream->send_pair_d = NULL;
    substream->recv_pair = NULL;
    substream->recv_pair_d = NULL;
    substream->grav_cells_pair = NULL;
    substream->grav_tasks_pair = NULL;
    substream->grav_pair_internal_from_self = NULL;
    substream->grav_batch_pair_count = 0;
    substream->pair_counts_h = NULL;
    substream->pair_offsets_h = NULL;
    substream->pair_active_counts_h = NULL;
    substream->pair_active_index_h = NULL;
    substream->pair_counts_d = NULL;
    substream->pair_offsets_d = NULL;
    substream->pair_total_count = 0;

    substream->send_self = NULL;
    substream->send_self_d = NULL;
    substream->recv_self = NULL;
    substream->recv_self_d = NULL;
    substream->grav_cells_self = NULL;
    substream->grav_tasks_self = NULL;
    substream->grav_batch_self_count = 0;
    substream->self_counts_h = NULL;
    substream->self_offsets_h = NULL;
    substream->self_counts_d = NULL;
    substream->self_offsets_d = NULL;
    substream->self_total_count = 0;
    substream->self_rmax_h = NULL;
    substream->self_rmax_d = NULL;

    substream->busy = 0;
  }

  gpu->next_substream = 0;
  gpu->grav_batch_ncells = 0;
  gpu->grav_max_cell_size = 0;
  
  free(gpu->substreams);
  gpu->substreams = NULL;
  gpu->nstreams = 0;
}

/**
 * @brief Flush any leftover packed self-gravity work owned by a runner.
 *
 * @param r The runner whose GPU batch should be flushed.
 * @return The outcome of the leftover flush attempt.
 */
enum runner_gpu_task_type runner_gpu_flush_leftover_self(struct runner *r) {

  enum runner_gpu_task_type result = regular_task;

  for (int l = 0; l < r->gpu.nstreams; l++) {
    struct gpu_runner_substream *substream = &r->gpu.substreams[l];
    const int nslots = substream->grav_batch_self_count;
    const int total = substream->self_total_count;
    const int max_cell_size = r->gpu.grav_max_cell_size;

    if (nslots == 0) continue;

    GPUMemcpyAsync(substream->self_counts_d, substream->self_counts_h,
                   nslots * sizeof(int),
                   GPU_MEMCPY_HOST_TO_DEVICE, substream->stream);

    GPUMemcpyAsync(substream->self_offsets_d, substream->self_offsets_h,
                   nslots * sizeof(int),
                   GPU_MEMCPY_HOST_TO_DEVICE, substream->stream);

    GPUMemcpyAsync(substream->self_active_counts_d, substream->self_active_counts_h,
                   nslots * sizeof(int),
                   GPU_MEMCPY_HOST_TO_DEVICE, substream->stream);

    GPUMemcpyAsync(substream->self_active_offsets_d,
               substream->self_active_offsets_h,
               nslots * sizeof(int),
               GPU_MEMCPY_HOST_TO_DEVICE, substream->stream);

    GPUMemcpyAsync(substream->self_active_index_d,
               substream->self_active_index_h,
               (size_t)substream->self_total_active_count * sizeof(int),
               GPU_MEMCPY_HOST_TO_DEVICE, substream->stream);
                   
    GPUMemcpyAsync(substream->self_rmax_d, substream->self_rmax_h,
               nslots * sizeof(float),
               GPU_MEMCPY_HOST_TO_DEVICE, substream->stream);

    GPUMemcpyAsync(substream->send_self_d, substream->send_self,
                   total * sizeof(struct gravity_gpu_values_send),
                   GPU_MEMCPY_HOST_TO_DEVICE, substream->stream);

    GPUMemsetAsync(
	    substream->recv_self_active_d,
	    0,
	    (size_t)substream->self_total_active_count *
		sizeof(struct gravity_gpu_values_recv),
	    substream->stream);
    
    runner_doself_grav_pp_flush(
    r, substream, nslots, substream->self_max_active_count, max_cell_size, substream->stream);

    GPUMemcpyAsync(
	    substream->recv_self_active,
	    substream->recv_self_active_d,
	    (size_t)substream->self_total_active_count *
		sizeof(struct gravity_gpu_values_recv),
	    GPU_MEMCPY_DEVICE_TO_HOST,
	    substream->stream);

    GPUEventRecord(substream->done, substream->stream);
    GPUEventSynchronize(substream->done);

    for (int j = 0; j < nslots; j++) {
      struct cell *c_unpack = substream->grav_cells_self[j];
      const int count = substream->self_counts_h[j];
      const int offset = substream->self_offsets_h[j];

      while (cell_glocktree(c_unpack)) {
        ;
      }

      const int active_count = substream->self_active_counts_h[j];
	const int active_base = substream->self_active_offsets_h[j];

	for (int a = 0; a < active_count; a++) {
  		const int local_pid = substream->self_active_index_h[active_base + a];
  		const int k = active_base + a;

	  c_unpack->grav.parts[local_pid].a_grav[0] +=
	      substream->recv_self_active[k].values_i.x;
	  c_unpack->grav.parts[local_pid].a_grav[1] +=
	      substream->recv_self_active[k].values_i.y;
	  c_unpack->grav.parts[local_pid].a_grav[2] +=
	      substream->recv_self_active[k].values_i.z;
	  c_unpack->grav.parts[local_pid].potential +=
	      substream->recv_self_active[k].values_i.w;
	}

      cell_gunlocktree(c_unpack);
    }

    runner_gpu_complete_self_batch(r, &r->e->sched, substream);

    result = flushed_self_task;
  }

  return result;
}

/**
 * @brief Flush any leftover packed pair-gravity work owned by a runner.
 *
 * @param r The runner whose GPU batch should be flushed.
 * @return The outcome of the leftover flush attempt.
 */
enum runner_gpu_task_type runner_gpu_flush_leftover_pair(struct runner *r) {

  enum runner_gpu_task_type result = regular_task;

  for (int l = 0; l < r->gpu.nstreams; l++) {
    struct gpu_runner_substream *substream = &r->gpu.substreams[l];

    if (substream->grav_batch_pair_count == 0) continue;

    runner_dopair_grav_pp_flush(
        r, substream,
        substream->send_pair,
        substream->send_pair_d,
        substream->recv_pair,
        substream->recv_pair_d,
        substream->grav_cells_pair,
        substream->grav_tasks_pair,
        NULL,
        r->gpu.grav_batch_ncells,
        r->gpu.grav_max_cell_size,
        substream->stream);

    result = flushed_pair_task;
  }

  return result;
}
