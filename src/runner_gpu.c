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
#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef WITH_MPI
#include <mpi.h>
#endif

static int runner_gpu_local_ranks_on_device_for_budget = 1;

static void runner_gpu_check_queue_counters(struct runner *r,
                                            struct scheduler *sched,
                                            const char *where) {

  const int self_left = sched->queues[r->qid].gpu_self_tasks_left;
  const int pair_left = sched->queues[r->qid].gpu_pair_tasks_left;

  if (self_left < 0 || pair_left < 0 ||
      self_left > 1000000 || pair_left > 1000000) {
    error("%s: GPU queue counter corrupted: qid=%d "
          "self_left=%d pair_left=%d",
          where,
          r->qid,
          self_left,
          pair_left);
  }
}

static void runner_gpu_count_self_task(struct runner *r,
                                       struct scheduler *sched,
                                       struct task *t) {

  if (t == NULL)
    error("runner_gpu_count_self_task got NULL task.");

  if (t->gpu_completed) {
    error("Packing already GPU-completed self task: task=%p type=%s subtype=%s "
          "gpu_counted=%d done_count=%d qid=%d self_left=%d",
          (void *)t,
          taskID_names[t->type],
          subtaskID_names[t->subtype],
          t->gpu_counted,
          t->done_count,
          r->qid,
          sched->queues[r->qid].gpu_self_tasks_left);
  }

  if (t->gpu_counted)
    return;

  t->gpu_counted = 1;

  lock_lock(&sched->queues[r->qid].lock);
  sched->queues[r->qid].gpu_self_tasks_left++;
  runner_gpu_check_queue_counters(r, sched, "runner_gpu_count_self_task");
  /*message("GPU_SELF_COUNTER_INC packed: task=%p qid=%d new=%d",
          (void *)t, r->qid,
          sched->queues[r->qid].gpu_self_tasks_left);*/
  (void)lock_unlock(&sched->queues[r->qid].lock);
}

static void runner_gpu_count_pair_task(struct runner *r,
                                       struct scheduler *sched,
                                       struct task *t) {

  if (t == NULL)
    error("runner_gpu_count_pair_task got NULL task.");

  if (t->gpu_completed) {
    error("Packing already GPU-completed pair task: task=%p type=%s subtype=%s "
          "gpu_counted=%d done_count=%d qid=%d",
          (void *)t,
          taskID_names[t->type],
          subtaskID_names[t->subtype],
          t->gpu_counted,
          t->done_count,
          r->qid);
  }

  if (t->gpu_counted)
    return;

  t->gpu_counted = 1;

  lock_lock(&sched->queues[r->qid].lock);
  sched->queues[r->qid].gpu_pair_tasks_left++;
  runner_gpu_check_queue_counters(r, sched, "runner_gpu_count_pair_task");
  /*message("GPU_PAIR_COUNTER_INC packed: task=%p qid=%d new=%d",
          (void *)t, r->qid,
          sched->queues[r->qid].gpu_pair_tasks_left);*/
  (void)lock_unlock(&sched->queues[r->qid].lock);
}


void runner_gpu_bind_device(struct runner *r) {

#if defined(WITH_CUDA) || defined(WITH_HIP)
  const int device_id = r->gpu.device_id;

  if (device_id < 0)
    error("runner_gpu_bind_device: invalid GPU device_id=%d", device_id);

  const GPUError err = GPUSetDevice(device_id);

  if (err != GPU_SUCCESS)
    error("runner_gpu_bind_device: GPUSetDevice(%d) failed: %s",
          device_id, GPUGetErrorString(err));
#else
  (void)r;
#endif
}

static void runner_gpu_mark_done_debug(
    struct runner *r,
    struct task *t,
    const char *where) {

  if (t == NULL)
    error("%s: NULL task passed to GPU completion marker.", where);

#ifdef WITH_MPI
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  int rank = 0;
#endif

  const int old_count = t->gpu_done_count;

  t->gpu_done_count++;

  if (old_count > 0) {
    error("GPU task completed more than once. "
          "task=%p type=%s subtype=%s old_count=%d new_where=%s "
          "old_where=%s old_runner=%d old_qid=%d old_rank=%d "
          "new_runner=%d new_qid=%d new_rank=%d gpu_completed=%d",
          (void *)t,
          taskID_names[t->type],
          subtaskID_names[t->subtype],
          old_count,
          where,
          t->gpu_done_where != NULL ? t->gpu_done_where : "(null)",
          t->gpu_done_runner,
          t->gpu_done_qid,
          t->gpu_done_rank,
          r->id,
          r->qid,
          rank,
          t->gpu_completed);
  }

  t->gpu_done_where = where;
  t->gpu_done_runner = r->id;
  t->gpu_done_qid = r->qid;
  t->gpu_done_rank = rank;
}

static inline void runner_gpu_check_error(const char *where) {
  const GPUError err = GPUGetLastError();

  if (err != GPU_SUCCESS)
    error("%s: %s", where, GPUGetErrorString(err));
}

/* ------------------------------------------------------------------------- */
/* GPU timing helpers: thread-local state only, no struct layout changes.     */
/* This avoids changing runner_gpu.h and avoids shared locks in runner threads.*/
/* ------------------------------------------------------------------------- */

static __thread double runner_gpu_self_pack_time_s = 0.0;
static __thread double runner_gpu_pair_pack_time_s = 0.0;

static inline double runner_gpu_walltime_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

/*
 * GPU event timing for asynchronous work.
 * pack_s/unpack_s use CPU wall-clock time because they run on the CPU.
 * h2d_s/kernel_s/d2h_s use GPU events because copies and kernel launches are async.
 */
static double runner_gpu_event_elapsed_s(GPUEvent start, GPUEvent stop) {
  float ms = 0.0f;

  GPUEventSynchronize(stop);

#if defined(HAVE_HIP) || defined(SWIFT_HIP)
  hipEventElapsedTime(&ms, start, stop);
#else
  cudaEventElapsedTime(&ms, start, stop);
#endif

  return 1.0e-3 * (double)ms;
}

static void runner_gpu_write_timing_row(
    const char *kind,
    long long step,
    int runner_id,
    int stream_id,
    int nslots,
    int nparts,
    double pack_s,
    double h2d_s,
    double kernel_s,
    double d2h_s,
    double unpack_s) {

  const char *path = getenv("SWIFT_GPU_TIMING_FILE");

  if (path == NULL || path[0] == '\0')
    path = "gpu_timing.csv";

  FILE *fp = fopen(path, "a");

  if (fp == NULL)
    return;

  fseek(fp, 0, SEEK_END);

  if (ftell(fp) == 0) {
    fprintf(fp,
            "kind,step,runner,stream,nslots,nparticles,"
            "pack_s,h2d_s,kernel_s,d2h_s,unpack_s\n");
  }

  fprintf(fp,
          "%s,%lld,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e\n",
          kind,
          step,
          runner_id,
          stream_id,
          nslots,
          nparts,
          pack_s,
          h2d_s,
          kernel_s,
          d2h_s,
          unpack_s);

  fclose(fp);
}

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
    int periodic, double min_trunc, const float *r_s_inv,
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


static int runner_gpu_find_or_pack_pair_cell(
    struct runner *r,
    struct gpu_runner_substream *substream,
    struct cell *c,
    struct gravity_cache *cache,
    int max_cell_size) {

  for (int s = 0; s < substream->pair_unique_cell_count; s++) {
    if (substream->pair_unique_cells[s] == c)
      return s;
  }

  const int slot = substream->pair_unique_cell_count++;
  const int gcount = c->grav.count;

  if (slot >= r->gpu.grav_batch_ncells)
    error("Too many unique pair cells in GPU batch");

  if (gcount > max_cell_size)
    error("Pair unique-cell pack overflow: gcount=%d > max_cell_size=%d",
          gcount, max_cell_size);

  append_packed_pair_cell(substream, slot, gcount);

  const int off = substream->pair_offsets_h[slot];

  for (int i = 0; i < gcount; i++) {
    const int k = off + i;

    substream->send_pair_pos_mass[k].x = cache->x[i];
    substream->send_pair_pos_mass[k].y = cache->y[i];
    substream->send_pair_pos_mass[k].z = cache->z[i];
    substream->send_pair_pos_mass[k].w = cache->m[i];

    substream->send_pair_h[k] = cache->epsilon[i];
  }

  int active_count = 0;
  const int active_base = substream->pair_total_active_count;
  substream->pair_active_offsets_h[slot] = active_base;

  const int local_active_cell =
    (c->nodeID == r->e->nodeID) && cell_is_active_gravity(c, r->e);

  for (int i = 0; i < gcount; i++) {
    if (local_active_cell && cache->active[i] > 0) {
      substream->pair_active_index_h[active_base + active_count] = i;
      active_count++;
    }
  }

  substream->pair_total_active_count += active_count;
  substream->pair_active_counts_h[slot] = active_count;

  if (active_count > substream->pair_max_active_count)
    substream->pair_max_active_count = active_count;

  substream->pair_unique_cells[slot] = c;

  return slot;
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

  if (t == NULL)
    error("runner_gpu_complete_self_task got NULL task.");

  if (t->gpu_completed) {
	  error("runner_gpu_complete_self_task called for already-completed task: "
		"task=%p type=%s subtype=%s gpu_counted=%d done_count=%d "
		"self_left=%d qid=%d",
		(void *)t,
		taskID_names[t->type],
		subtaskID_names[t->subtype],
		t->gpu_counted,
		t->done_count,
		sched->queues[r->qid].gpu_self_tasks_left,
		r->qid);
	}

  t->gpu_completed = 1;

  if (!t->gpu_counted) {
	  error("Completing uncounted GPU self task: task=%p type=%s subtype=%s "
		"gpu_completed=%d done_count=%d qid=%d",
		(void *)t,
		taskID_names[t->type],
		subtaskID_names[t->subtype],
		t->gpu_completed,
		t->done_count,
		r->qid);
	}

	lock_lock(&sched->queues[r->qid].lock);
	
	const int before = sched->queues[r->qid].gpu_self_tasks_left;

	if (sched->queues[r->qid].gpu_self_tasks_left <= 0)
	  error("gpu_self_tasks_left underflow: task=%p type=%s subtype=%s qid=%d",
		(void *)t,
		taskID_names[t->type],
		subtaskID_names[t->subtype],
		r->qid);

	sched->queues[r->qid].gpu_self_tasks_left--;
	runner_gpu_check_queue_counters(r, sched, "runner_gpu_complete_self_task");
	
	/*message("GPU_SELF_COUNTER_DEC complete: task=%p qid=%d old=%d new=%d "
        "gpu_completed=%d done_count=%d",
        (void *)t,
        r->qid,
        before,
        before - 1,
        t->gpu_completed,
        t->done_count);*/

	(void)lock_unlock(&sched->queues[r->qid].lock);

	t->gpu_counted = 0;

  runner_gpu_mark_done_debug(r, t, "runner_gpu_complete_self_task");

  scheduler_done(sched, t);
}

void runner_gpu_complete_current_self_task(struct runner *r,
                                           struct scheduler *sched,
                                           struct task *t) {
  runner_gpu_complete_self_task(r, sched, t);
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

  if (t == NULL)
    error("runner_gpu_complete_pair_task got NULL task.");

  if (t->gpu_completed) {
    error("runner_gpu_complete_pair_task called for already-completed task: "
          "task=%p type=%s subtype=%s gpu_counted=%d done_count=%d "
          "pair_left=%d qid=%d",
          (void *)t,
          taskID_names[t->type],
          subtaskID_names[t->subtype],
          t->gpu_counted,
          t->done_count,
          sched->queues[r->qid].gpu_pair_tasks_left,
          r->qid);
  }

  t->gpu_completed = 1;

  if (!t->gpu_counted) {
    error("Completing uncounted GPU pair task: task=%p type=%s subtype=%s "
          "gpu_completed=%d done_count=%d qid=%d pair_left=%d",
          (void *)t,
          taskID_names[t->type],
          subtaskID_names[t->subtype],
          t->gpu_completed,
          t->done_count,
          r->qid,
          sched->queues[r->qid].gpu_pair_tasks_left);
  }

  lock_lock(&sched->queues[r->qid].lock);

  const int before = sched->queues[r->qid].gpu_pair_tasks_left;

  if (before <= 0)
    error("gpu_pair_tasks_left underflow before completing task=%p "
          "type=%s subtype=%s qid=%d pair_left=%d",
          (void *)t,
          taskID_names[t->type],
          subtaskID_names[t->subtype],
          r->qid,
          before);

  sched->queues[r->qid].gpu_pair_tasks_left--;
  runner_gpu_check_queue_counters(r, sched, "runner_gpu_complete_pair_task");

  /*message("GPU_PAIR_COUNTER_DEC complete: task=%p qid=%d old=%d new=%d "
          "gpu_completed=%d done_count=%d",
          (void *)t,
          r->qid,
          before,
          before - 1,
          t->gpu_completed,
          t->done_count);*/

  (void)lock_unlock(&sched->queues[r->qid].lock);

  t->gpu_counted = 0;

  runner_gpu_mark_done_debug(r, t, "runner_gpu_complete_pair_task");

  scheduler_done(sched, t);
}

/**
 * @brief Complete all self-gravity tasks in the current GPU batch.
 *
 * @param r The #runner owning the batch.
 * @param sched The scheduler tracking the tasks.
 */
void runner_gpu_complete_self_batch(struct runner *r, struct scheduler *sched,
                                    struct gpu_runner_substream *substream,
                                    struct task *current_task) {

  const int count = substream->grav_batch_self_count;

  /* Complete each top-level self task at most once. Recursive self walks can
   * pack many leaf cells for the same scheduler task, so duplicate task
   * pointers in grav_tasks_self[] are normal. */
  for (int i = 0; i < count; i++) {

    struct task *task = substream->grav_tasks_self[i];

    if (task == NULL)
      continue;

    /* The currently-walking task is completed by runner_main when
     * flushed_self_task is returned. */
    if (task == current_task)
      continue;

    /* Duplicate slot for a task already completed earlier in this batch. */
    if (task->gpu_completed)
      continue;

    /* Only counted top-level GPU tasks own a queue counter. */
    if (!task->gpu_counted)
      continue;

    runner_gpu_complete_self_task(r, sched, task);
  }

  /* Clear self batch cell/task entries. */
  for (int i = 0; i < count; i++) {
    substream->grav_cells_self[i] = NULL;
    substream->grav_tasks_self[i] = NULL;
  }

  substream->grav_batch_self_count = 0;
  substream->busy = 0;

  for (int i = 0; i < count; i++) {
    substream->self_counts_h[i] = 0;
    substream->self_offsets_h[i] = 0;
    substream->self_active_counts_h[i] = 0;
    substream->self_active_offsets_h[i] = 0;
    substream->self_rmax_h[i] = 0.f;
  }

  for (int a = 0; a < substream->self_total_active_count; a++)
    substream->self_active_index_h[a] = 0;

  substream->self_total_count = 0;
  substream->self_max_active_count = 0;
  substream->self_total_active_count = 0;

  runner_gpu_check_queue_counters(r, sched,
                                  "runner_gpu_complete_self_batch:end");
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

  for (int pair_id = 0; pair_id < count; pair_id++) {
    struct task *task = substream->grav_tasks_pair[pair_id];
    const int internal = substream->grav_pair_internal_from_self[pair_id];

    if (!internal && task != NULL)
      runner_gpu_complete_pair_task(r, sched, task);

    substream->grav_cells_pair[2 * pair_id] = NULL;
    substream->grav_cells_pair[2 * pair_id + 1] = NULL;
    substream->grav_tasks_pair[pair_id] = NULL;
    substream->grav_pair_internal_from_self[pair_id] = 0;
  }

  substream->grav_batch_pair_count = 0;
  substream->busy = 0;

  for (int i = 0; i < count; i++) {
    substream->pair_counts_h[i] = 0;
    substream->pair_offsets_h[i] = 0;
    substream->pair_active_counts_h[i] = 0;
    substream->pair_active_offsets_h[i] = 0;
    substream->pair_cell_flags_h[i] = 0;
  }

  substream->pair_total_count = 0;
  substream->pair_unique_cell_count = 0;
  substream->pair_total_active_count = 0;
  substream->pair_max_active_count = 0;
  substream->pair_total_pair_active_count = 0;
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
    struct cell **grav_cells_pair, struct task **grav_tasks_pair,
    unsigned char *grav_pair_internal_from_self,
    struct task *t, int internal_from_self,
    int max_cell_size, GPUStream stream) {

  (void)symmetric;
  (void)stream;

  const struct engine *e = r->e;
  const int periodic = e->mesh->periodic;
  const float dim[3] = {(float)e->mesh->dim[0], (float)e->mesh->dim[1],
                        (float)e->mesh->dim[2]};
  const double min_trunc = e->mesh->r_cut_min;

  const double pack_time_start = runner_gpu_walltime_s();

  const int ci_active =
      cell_is_active_gravity(ci, e) && (ci->nodeID == e->nodeID);
  const int cj_active =
      cell_is_active_gravity(cj, e) && (cj->nodeID == e->nodeID);

#ifdef SWIFT_DEBUG_CHECKS
  if (ci->split || cj->split) error("Running P-P on splitable cells");
  if (!cell_are_gpart_drifted(ci, e)) error("Un-drifted gparts");
  if (!cell_are_gpart_drifted(cj, e)) error("Un-drifted gparts");

  if (cj_active && ci->grav.ti_old_multipole != e->ti_current)
    error("Un-drifted multipole");
  if (ci_active && cj->grav.ti_old_multipole != e->ti_current)
    error("Un-drifted multipole");
#endif

  struct gravity_cache *const ci_cache = &r->ci_gravity_cache;
  struct gravity_cache *const cj_cache = &r->cj_gravity_cache;

  const double shift_i[3] = {0., 0., 0.};
  const double shift_j[3] = {0., 0., 0.};

  const float rmax_i = ci->grav.multipole->r_max;
  const float rmax_j = cj->grav.multipole->r_max;

  const float CoM_i[3] = {
      (float)(ci->grav.multipole->CoM[0] - shift_i[0]),
      (float)(ci->grav.multipole->CoM[1] - shift_i[1]),
      (float)(ci->grav.multipole->CoM[2] - shift_i[2])};

  const float CoM_j[3] = {
      (float)(cj->grav.multipole->CoM[0] - shift_j[0]),
      (float)(cj->grav.multipole->CoM[1] - shift_j[1]),
      (float)(cj->grav.multipole->CoM[2] - shift_j[2])};

  const int gcount_i = ci->grav.count;
  const int gcount_j = cj->grav.count;

  const int gcount_padded_i = gcount_i - (gcount_i % VEC_SIZE) + VEC_SIZE;
  const int gcount_padded_j = gcount_j - (gcount_j % VEC_SIZE) + VEC_SIZE;

  const int allow_multipole_i = allow_mpole && ci->grav.count > 1;
  const int allow_multipole_j = allow_mpole && cj->grav.count > 1;

  if (gcount_i > max_cell_size)
    error("Pair pack overflow: gcount_i=%d > max_cell_size=%d",
          gcount_i, max_cell_size);

  if (gcount_j > max_cell_size)
    error("Pair pack overflow: gcount_j=%d > max_cell_size=%d",
          gcount_j, max_cell_size);

  if (ci->nodeID == e->nodeID) {
    gravity_cache_populate(e->max_active_bin, allow_multipole_j, periodic, dim,
                           ci_cache, ci->grav.parts, gcount_i, gcount_padded_i,
                           shift_i, CoM_j, cj->grav.multipole, ci,
                           e->gravity_properties);
  } else {
    gravity_cache_populate_foreign(periodic, dim, ci_cache,
                                   ci->grav.parts_foreign, gcount_i,
                                   gcount_padded_i, shift_i, ci,
                                   e->gravity_properties);
  }

  if (cj->nodeID == e->nodeID) {
    gravity_cache_populate(e->max_active_bin, allow_multipole_i, periodic, dim,
                           cj_cache, cj->grav.parts, gcount_j, gcount_padded_j,
                           shift_j, CoM_i, ci->grav.multipole, cj,
                           e->gravity_properties);
  } else {
    gravity_cache_populate_foreign(periodic, dim, cj_cache,
                                   cj->grav.parts_foreign, gcount_j,
                                   gcount_padded_j, shift_j, cj,
                                   e->gravity_properties);
  }

  struct cell *a = ci;
  struct cell *b = cj;

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
  
  const int pair_capacity = r->gpu.grav_batch_ncells / 2;
	const int cell_capacity = r->gpu.grav_batch_ncells;

	if (substream->grav_batch_pair_count < 0 ||
	    substream->grav_batch_pair_count >= pair_capacity) {
	  error("Pair batch overflow before pack: pair_count=%d pair_capacity=%d",
		substream->grav_batch_pair_count,
		pair_capacity);
	}

	if (substream->pair_unique_cell_count < 0 ||
	    substream->pair_unique_cell_count + 2 > cell_capacity) {
	  error("Pair unique-cell overflow before pack: unique=%d cell_capacity=%d",
		substream->pair_unique_cell_count,
		cell_capacity);
	}

  const int slot_i =
    runner_gpu_find_or_pack_pair_cell(r, substream, ci, ci_cache, max_cell_size);

  const int slot_j =
    runner_gpu_find_or_pack_pair_cell(r, substream, cj, cj_cache, max_cell_size);

  const int pair_id = substream->grav_batch_pair_count;
  
  if (pair_id < 0 || pair_id >= pair_capacity)
  	error("Bad pair_id=%d pair_capacity=%d", pair_id, pair_capacity);

  substream->pair_pair_i_h[pair_id] = slot_i;
  substream->pair_pair_j_h[pair_id] = slot_j;

  grav_cells_pair[2 * pair_id] = ci;
  grav_cells_pair[2 * pair_id + 1] = cj;
  grav_tasks_pair[pair_id] = t;
  grav_pair_internal_from_self[pair_id] =
      (unsigned char)internal_from_self;
      
  if (!internal_from_self)
  	runner_gpu_count_pair_task(r, &r->e->sched, t);

  int use_full = 1;

  if (periodic) {
    double d0 = CoM_j[0] - CoM_i[0];
    double d1 = CoM_j[1] - CoM_i[1];
    double d2 = CoM_j[2] - CoM_i[2];

    d0 = nearest(d0, e->mesh->dim[0]);
    d1 = nearest(d1, e->mesh->dim[1]);
    d2 = nearest(d2, e->mesh->dim[2]);

    const double r2 = d0 * d0 + d1 * d1 + d2 * d2;
    const double max_r = sqrt(r2) + rmax_i + rmax_j;

    use_full = (max_r <= min_trunc);
  }
  
  substream->pair_use_full_h[pair_id] = use_full;

  substream->pair_cell_flags_h[slot_i] =
    (ci->nodeID == e->nodeID && cell_is_active_gravity(ci, e)) ? 1 : 0;

  substream->pair_cell_flags_h[slot_j] =
    (cj->nodeID == e->nodeID && cell_is_active_gravity(cj, e)) ? 1 : 0;
    
  const int side_i = 2 * pair_id;
  const int side_j = side_i + 1;

  substream->pair_side_active_offsets_h[side_i] =
    substream->pair_total_pair_active_count;

  substream->pair_total_pair_active_count +=
    substream->pair_active_counts_h[slot_i];
    
  if (2 * pair_id + 1 >= 2 * pair_capacity)
	  error("Bad pair side offset index: pair_id=%d pair_capacity=%d",
		pair_id, pair_capacity);

  substream->pair_side_active_offsets_h[side_j] =
    substream->pair_total_pair_active_count;

  substream->pair_total_pair_active_count +=
    substream->pair_active_counts_h[slot_j];

  substream->grav_batch_pair_count++;

  gravity_cache_zero_output(ci_cache, gcount_padded_i);
  gravity_cache_zero_output(cj_cache, gcount_padded_j);

  cell_gunlocktree(b);
  cell_gunlocktree(a);

  runner_gpu_pair_pack_time_s += runner_gpu_walltime_s() - pack_time_start;
}

static inline void runner_gpu_unpack_pair_side(
    struct runner *r,
    struct gpu_runner_substream *substream,
    struct cell *c,
    int slot,
    int recv_base) {

  const struct engine *e = r->e;

  if (c == NULL)
    error("GPU pair unpack received NULL cell.");

  /* MPI safety: never write to a foreign cell. */
  if (c->nodeID != e->nodeID)
    return;

  if (!cell_is_active_gravity(c, e))
    return;

  const int active_count = substream->pair_active_counts_h[slot];
  const int active_base = substream->pair_active_offsets_h[slot];

  if (active_count == 0)
    return;

  while (cell_glocktree(c)) {
    ;
  }

  for (int a = 0; a < active_count; a++) {
    const int local_pid =
        substream->pair_active_index_h[active_base + a];

    const int k = recv_base + a;

#ifdef SWIFT_DEBUG_CHECKS
    if (local_pid < 0 || local_pid >= c->grav.count)
      error("GPU pair unpack local_pid=%d out of range [0,%d).",
            local_pid, c->grav.count);
#endif

    c->grav.parts[local_pid].a_grav[0] +=
        substream->recv_pair_active[k].values_i.x;
    c->grav.parts[local_pid].a_grav[1] +=
        substream->recv_pair_active[k].values_i.y;
    c->grav.parts[local_pid].a_grav[2] +=
        substream->recv_pair_active[k].values_i.z;
    c->grav.parts[local_pid].potential +=
        substream->recv_pair_active[k].values_i.w;
  }

  cell_gunlocktree(c);
}

/**
 * @brief Flush a full pair-gravity GPU batch: H2D copy, kernel launch, D2H
 *        copy, stream synchronisation, result unpacking, scheduler completion,
 *        and metadata reset.
 *
 * current_task is skipped because it may still be in the recursive walk.
 * If this flush happened while walking current_task, runner_dopair_grav_pp_new()
 * must return flushed_pair_task so runner_main() completes current_task.
 * If this is a leftover flush, current_task should be NULL and all non-internal
 * tasks in the batch are completed here.
 */
static void runner_dopair_grav_pp_flush(
    struct runner *r,
    struct gpu_runner_substream *substream,
    struct cell **grav_cells_pair,
    struct task **grav_tasks_pair,
    struct task *current_task,
    int ncells,
    int max_cell_size,
    GPUStream stream) {

  runner_gpu_bind_device(r);

  const int npairs = substream->grav_batch_pair_count;
  const int nslots = substream->pair_unique_cell_count;
  const int ncells_flush = nslots;

  if (npairs == 0 || ncells_flush == 0)
    return;

  if (npairs < 0 || npairs > ncells / 2)
    error("Bad pair flush npairs=%d capacity=%d", npairs, ncells / 2);

  if (nslots < 0 || nslots > ncells)
    error("Bad pair flush nslots=%d ncells=%d", nslots, ncells);

  if (substream->pair_total_count < 0)
    error("Bad pair_total_count=%d", substream->pair_total_count);

  if (substream->pair_total_active_count < 0)
    error("Bad pair_total_active_count=%d",
          substream->pair_total_active_count);

  if (substream->pair_total_pair_active_count < 0)
    error("Bad pair_total_pair_active_count=%d",
          substream->pair_total_pair_active_count);

  double h2d_s = 0.0;
  double kernel_s = 0.0;
  double d2h_s = 0.0;
  double unpack_s = 0.0;

  const struct engine *e = r->e;
  const int periodic = e->mesh->periodic;
  const float r_s_inv = e->mesh->r_s_inv;
  const double min_trunc = e->mesh->r_cut_min;

  const float dim_0 = (float)e->mesh->dim[0];
  const float dim_1 = (float)e->mesh->dim[1];
  const float dim_2 = (float)e->mesh->dim[2];

  const double pack_s = runner_gpu_pair_pack_time_s;
  runner_gpu_pair_pack_time_s = 0.0;

  /*message("PAIR FLUSH begin: qid=%d npairs=%d nslots=%d "
          "pair_total_count=%d pair_total_active=%d "
          "pair_total_pair_active=%d pair_left=%d current_task=%p",
          r->qid,
          npairs,
          nslots,
          substream->pair_total_count,
          substream->pair_total_active_count,
          substream->pair_total_pair_active_count,
          r->e->sched.queues[r->qid].gpu_pair_tasks_left,
          (void *)current_task);*/

  /* ---- H2D copies ---- */
  {
    const double h2d_t0 = runner_gpu_walltime_s();

    TIMER_TIC;

    GPUMemcpyAsync(
        substream->send_pair_pos_mass_d,
        substream->send_pair_pos_mass,
        (size_t)substream->pair_total_count * sizeof(float4),
        GPU_MEMCPY_HOST_TO_DEVICE,
        stream);

    GPUMemcpyAsync(
        substream->send_pair_h_d,
        substream->send_pair_h,
        (size_t)substream->pair_total_count * sizeof(float),
        GPU_MEMCPY_HOST_TO_DEVICE,
        stream);

    GPUMemcpyAsync(
        substream->pair_counts_d,
        substream->pair_counts_h,
        (size_t)nslots * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        stream);

    GPUMemcpyAsync(
        substream->pair_offsets_d,
        substream->pair_offsets_h,
        (size_t)nslots * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        stream);

    GPUMemcpyAsync(
        substream->pair_active_counts_d,
        substream->pair_active_counts_h,
        (size_t)nslots * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        stream);

    GPUMemcpyAsync(
        substream->pair_active_offsets_d,
        substream->pair_active_offsets_h,
        (size_t)nslots * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        stream);

    GPUMemcpyAsync(
        substream->pair_active_index_d,
        substream->pair_active_index_h,
        (size_t)substream->pair_total_active_count * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        stream);

    GPUMemcpyAsync(
        substream->pair_pair_i_d,
        substream->pair_pair_i_h,
        (size_t)npairs * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        stream);

    GPUMemcpyAsync(
        substream->pair_pair_j_d,
        substream->pair_pair_j_h,
        (size_t)npairs * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        stream);

    GPUMemcpyAsync(
        substream->pair_use_full_d,
        substream->pair_use_full_h,
        (size_t)npairs * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        stream);

    GPUMemcpyAsync(
        substream->pair_side_active_offsets_d,
        substream->pair_side_active_offsets_h,
        (size_t)(2 * npairs) * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        stream);

    GPUMemcpyAsync(
        substream->pair_cell_flags_d,
        substream->pair_cell_flags_h,
        (size_t)nslots * sizeof(int),
        GPU_MEMCPY_HOST_TO_DEVICE,
        stream);

    TIMER_TOC(timer_doself_grav_pp);

    h2d_s = runner_gpu_walltime_s() - h2d_t0;
  }

  runner_gpu_check_error("runner_dopair_grav_pp_flush H2D");

  /* ---- Kernel launch ---- */
  {
    const double kernel_t0 = runner_gpu_walltime_s();

    TIMER_TIC;

    pair_pp_offload_new(
        periodic,
        min_trunc,
        &r_s_inv,
        substream->pair_use_full_d,
        substream->pair_side_active_offsets_d,
        substream->pair_counts_d,
        substream->pair_offsets_d,
        substream->pair_active_counts_d,
        substream->pair_active_offsets_d,
        substream->pair_active_index_d,
        substream->pair_pair_i_d,
        substream->pair_pair_j_d,
        npairs,
        nslots,
        dim_0,
        dim_1,
        dim_2,
        substream->pair_cell_flags_d,
        substream->send_pair_pos_mass_d,
        substream->send_pair_h_d,
        substream->recv_pair_active_d,
        ncells_flush,
        max_cell_size,
        substream->pair_max_active_count,
        stream);

    TIMER_TOC(timer_doself_grav_pp);

    kernel_s = runner_gpu_walltime_s() - kernel_t0;
  }

  runner_gpu_check_error("runner_dopair_grav_pp_flush kernel");

  /* ---- D2H copy ---- */
  {
    const double d2h_t0 = runner_gpu_walltime_s();

    TIMER_TIC;

    const size_t recv_pair_active_capacity =
        (size_t)(2 * (r->gpu.grav_batch_ncells / 2)) *
        (size_t)r->gpu.grav_max_cell_size;

    if ((size_t)substream->pair_total_pair_active_count >
        recv_pair_active_capacity) {
      error("GPU pair recv overflow: pair_total_pair_active_count=%d "
            "capacity=%zu",
            substream->pair_total_pair_active_count,
            recv_pair_active_capacity);
    }

    GPUMemcpyAsync(
        substream->recv_pair_active,
        substream->recv_pair_active_d,
        (size_t)substream->pair_total_pair_active_count *
            sizeof(struct gravity_gpu_values_recv),
        GPU_MEMCPY_DEVICE_TO_HOST,
        stream);

    GPUEventRecord(substream->done, stream);
    GPUEventSynchronize(substream->done);

    TIMER_TOC(timer_doself_grav_pp);

    d2h_s = runner_gpu_walltime_s() - d2h_t0;
  }

  runner_gpu_check_error("runner_dopair_grav_pp_flush D2H");

  /* ---- Unpack results back to particles ---- */
  {
    const double unpack_t0 = runner_gpu_walltime_s();

    TIMER_TIC;

    for (int pair_id = 0; pair_id < npairs; pair_id++) {

      const int slot_i = substream->pair_pair_i_h[pair_id];
      const int slot_j = substream->pair_pair_j_h[pair_id];

      if (slot_i < 0 || slot_i >= nslots)
        error("Bad pair slot_i=%d nslots=%d pair_id=%d",
              slot_i,
              nslots,
              pair_id);

      if (slot_j < 0 || slot_j >= nslots)
        error("Bad pair slot_j=%d nslots=%d pair_id=%d",
              slot_j,
              nslots,
              pair_id);

      const int recv_i = substream->pair_side_active_offsets_h[2 * pair_id];
      const int recv_j =
          substream->pair_side_active_offsets_h[2 * pair_id + 1];

      if (recv_i < 0 ||
          recv_i + substream->pair_active_counts_h[slot_i] >
              substream->pair_total_pair_active_count) {
        error("Bad pair recv_i=%d active_i=%d total_pair_active=%d "
              "pair_id=%d",
              recv_i,
              substream->pair_active_counts_h[slot_i],
              substream->pair_total_pair_active_count,
              pair_id);
      }

      if (recv_j < 0 ||
          recv_j + substream->pair_active_counts_h[slot_j] >
              substream->pair_total_pair_active_count) {
        error("Bad pair recv_j=%d active_j=%d total_pair_active=%d "
              "pair_id=%d",
              recv_j,
              substream->pair_active_counts_h[slot_j],
              substream->pair_total_pair_active_count,
              pair_id);
      }

      runner_gpu_unpack_pair_side(
          r,
          substream,
          grav_cells_pair[2 * pair_id],
          slot_i,
          recv_i);

      runner_gpu_unpack_pair_side(
          r,
          substream,
          grav_cells_pair[2 * pair_id + 1],
          slot_j,
          recv_j);
    }

    TIMER_TOC(timer_doself_grav_pp);

    unpack_s = runner_gpu_walltime_s() - unpack_t0;
  }

  /* ---- Complete scheduler tasks before clearing task pointers ---- */
  {
    struct scheduler *sched = &r->e->sched;

    int printed_current_skip = 0;

	for (int pair_id = 0; pair_id < npairs; pair_id++) {

	  struct task *batch_task = grav_tasks_pair[pair_id];
	  const int internal = substream->grav_pair_internal_from_self[pair_id];

	  if (batch_task == NULL)
	    error("NULL task in GPU pair batch: pair_id=%d npairs=%d",
		  pair_id, npairs);

	  if (!internal && batch_task == current_task) {
	    if (!printed_current_skip) {
	      /*message("PAIR FLUSH skipping current_task: task=%p "
		      "gpu_counted=%d gpu_completed=%d done_count=%d "
		      "pair_left=%d npairs=%d",
		      (void *)batch_task,
		      batch_task->gpu_counted,
		      batch_task->gpu_completed,
		      batch_task->done_count,
		      sched->queues[r->qid].gpu_pair_tasks_left,
		      npairs);*/
	      printed_current_skip = 1;
	    }
	    continue;
	  }

	  if (!internal && !batch_task->gpu_completed && batch_task->gpu_counted)
	    runner_gpu_complete_pair_task(r, sched, batch_task);
	
    }
  }

  /* ---- Timing output ---- */
  runner_gpu_write_timing_row(
      "pair",
      (long long)e->ti_current,
      r->id,
      0,
      npairs,
      substream->pair_total_count,
      pack_s,
      h2d_s,
      kernel_s,
      d2h_s,
      unpack_s);

  /* ---- Reset pair entries: arrays indexed by pair_id ---- */
  for (int pair_id = 0; pair_id < npairs; pair_id++) {
    grav_cells_pair[2 * pair_id] = NULL;
    grav_cells_pair[2 * pair_id + 1] = NULL;
    grav_tasks_pair[pair_id] = NULL;

    substream->grav_pair_internal_from_self[pair_id] = 0;
    substream->pair_pair_i_h[pair_id] = 0;
    substream->pair_pair_j_h[pair_id] = 0;
    substream->pair_use_full_h[pair_id] = 0;

    substream->pair_side_active_offsets_h[2 * pair_id] = 0;
    substream->pair_side_active_offsets_h[2 * pair_id + 1] = 0;
  }

  /* ---- Reset unique-cell entries: arrays indexed by unique cell slot ---- */
  for (int slot = 0; slot < nslots; slot++) {
    substream->pair_counts_h[slot] = 0;
    substream->pair_offsets_h[slot] = 0;
    substream->pair_active_counts_h[slot] = 0;
    substream->pair_active_offsets_h[slot] = 0;
    substream->pair_cell_flags_h[slot] = 0;
    substream->pair_unique_cells[slot] = NULL;
  }

  /* ---- Reset active particle index buffer only for the span used ---- */
  for (int a = 0; a < substream->pair_total_active_count; a++) {
    substream->pair_active_index_h[a] = 0;
  }

  substream->grav_batch_pair_count = 0;
  substream->busy = 0;

  substream->pair_total_count = 0;
  substream->pair_unique_cell_count = 0;
  substream->pair_total_active_count = 0;
  substream->pair_max_active_count = 0;
  substream->pair_total_pair_active_count = 0;

  runner_gpu_check_queue_counters(
      r,
      &r->e->sched,
      "runner_dopair_grav_pp_flush:end");

  /*message("PAIR FLUSH end: qid=%d pair_left=%d",
          r->qid,
          r->e->sched.queues[r->qid].gpu_pair_tasks_left);*/
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
    struct cell **grav_cells_pair, struct task **grav_tasks_pair,
    unsigned char *grav_pair_internal_from_self,
    struct task *t, int internal_from_self,
    int ncells, int max_cell_size, GPUStream stream) {

  const int pair_capacity = ncells / 2;
  enum runner_gpu_task_type result = packed_task;

  /* If the existing batch cannot accept this pair, flush it first.
     This flush skips current_task=t, so runner_main must complete t. */
  if (substream->grav_batch_pair_count + 1 > pair_capacity ||
      substream->pair_unique_cell_count + 2 > ncells) {

    runner_dopair_grav_pp_flush(
        r, substream,
        grav_cells_pair, grav_tasks_pair,
        t, ncells, max_cell_size, stream);

    result = flushed_pair_task;
  }

  runner_dopair_grav_pp_pack(
      r, substream, ci, cj, symmetric, allow_mpole,
      grav_cells_pair, grav_tasks_pair,
      grav_pair_internal_from_self,
      t, internal_from_self,
      max_cell_size, stream);

  /* If packing this pair filled the batch, flush it too.
     Again, current_task=t is skipped in the flush, so return flushed_pair_task. */
  if (substream->grav_batch_pair_count >= pair_capacity ||
      substream->pair_unique_cell_count >= ncells) {

    runner_dopair_grav_pp_flush(
        r, substream,
        grav_cells_pair, grav_tasks_pair,
        t, ncells, max_cell_size, stream);

    return flushed_pair_task;
  }

  return result;
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
    
    runner_gpu_bind_device(r);

  if (ci->nodeID != r->e->nodeID)
    error("GPU self task attempted to pack a foreign cell.");

  const double pack_time_start = runner_gpu_walltime_s();

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
  
  runner_gpu_count_self_task(r, &r->e->sched, t);
  
  substream->grav_batch_self_count++;
  substream->self_rmax_h[slot] = 2.f * ci->grav.multipole->r_max;

  gravity_cache_zero_output(ci_cache, gcount_padded);
  cell_gunlocktree(ci);

  runner_gpu_self_pack_time_s += runner_gpu_walltime_s() - pack_time_start;

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

    double h2d_s = 0.0;
    double kernel_s = 0.0;
    double d2h_s = 0.0;
    double unpack_s = 0.0;

    /*GPUEvent h2d_start, h2d_stop;
    GPUEvent kernel_start, kernel_stop;
    GPUEvent d2h_start, d2h_stop;

    GPUEventCreate(&h2d_start);
    GPUEventCreate(&h2d_stop);
    GPUEventCreate(&kernel_start);
    GPUEventCreate(&kernel_stop);
    GPUEventCreate(&d2h_start);
    GPUEventCreate(&d2h_stop);*/

    /* copy packed metadata */
    //GPUEventRecord(h2d_start, substream->stream);
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

    //GPUEventRecord(h2d_stop, substream->stream);

    /*GPUMemsetAsync(
	    substream->recv_self_active_d,
	    0,
	    (size_t)substream->self_total_active_count *
		sizeof(struct gravity_gpu_values_recv),
	    substream->stream);*/

    /* kernel */
    //GPUEventRecord(kernel_start, substream->stream);

    runner_doself_grav_pp_flush(
    r, substream, nslots, substream->self_max_active_count, max_cell_size, substream->stream);

    //GPUEventRecord(kernel_stop, substream->stream);

    /* D2H: only live data */
    //GPUEventRecord(d2h_start, substream->stream);

    GPUMemcpyAsync(
	    substream->recv_self_active,
	    substream->recv_self_active_d,
	    (size_t)substream->self_total_active_count *
		sizeof(struct gravity_gpu_values_recv),
	    GPU_MEMCPY_DEVICE_TO_HOST,
	    substream->stream);

    //GPUEventRecord(d2h_stop, substream->stream);
    GPUEventRecord(substream->done, substream->stream);
    GPUEventSynchronize(substream->done);

    /*h2d_s = runner_gpu_event_elapsed_s(h2d_start, h2d_stop);
    kernel_s = runner_gpu_event_elapsed_s(kernel_start, kernel_stop);
    d2h_s = runner_gpu_event_elapsed_s(d2h_start, d2h_stop);*/

    /* ===================== UNPACK ===================== */

    const double unpack_t0 = runner_gpu_walltime_s();

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
	  if (c_unpack == NULL)
  		error("GPU self unpack received NULL cell.");

	  if (c_unpack->nodeID != r->e->nodeID)
  		error("GPU self unpack attempted to write a foreign cell.");

	  if (!cell_is_active_gravity(c_unpack, r->e))
  		continue;
  
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

    unpack_s = runner_gpu_walltime_s() - unpack_t0;

    const double self_pack_s = runner_gpu_self_pack_time_s;
    runner_gpu_self_pack_time_s = 0.0;
    /*runner_gpu_write_timing_row("self", (long long)r->e->step, r->id, r->qid, nslots, total,
                                self_pack_s, h2d_s, kernel_s, d2h_s,
                                unpack_s);*/

    /*GPUEventDestroy(h2d_start);
    GPUEventDestroy(h2d_stop);
    GPUEventDestroy(kernel_start);
    GPUEventDestroy(kernel_stop);
    GPUEventDestroy(d2h_start);
    GPUEventDestroy(d2h_stop);*/

    /* ===================== COMPLETE TASKS ===================== */

    runner_gpu_complete_self_batch(r, &r->e->sched, substream, t);

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
  const double usable_fraction = 0.80;
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
  const int nr_threads = e->nr_threads > 0 ? e->nr_threads : 1;
  const int nr_gpu_runners =
    nr_threads * runner_gpu_local_ranks_on_device_for_budget;

  const size_t metadata = 64ULL * 1024ULL * 1024ULL; /* 64 MB */

  size_t budget = 0;
  if (usable_bytes > metadata)
    budget = usable_bytes - metadata;

  const size_t per_runner_budget = budget / (size_t)nr_gpu_runners;

  int ncells = (int)(per_runner_budget / bytes_per_cell_per_runner);

  if (ncells < 2) ncells = 2;
  if (ncells > 10000) ncells = 10000;

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

  const int nr_threads = e->nr_threads > 0 ? e->nr_threads : 1;
  const int nr_gpu_runners =
    nr_threads * runner_gpu_local_ranks_on_device_for_budget;
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

static int runner_gpu_select_device(struct runner *r) {

  struct engine *e = r->e;
  struct gpu_runner *gpu = &r->gpu;

  int ngpu = 0;
  GPUGetDeviceCount(&ngpu);

  if (ngpu <= 0)
    error("No CUDA/HIP GPU visible to this MPI rank.");

  int local_rank = 0;
  int local_size = 1;

#ifdef WITH_MPI
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &local_comm);

  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);

  MPI_Comm_free(&local_comm);
#endif

  const int user_device = parser_get_opt_param_int(
      e->parameter_file, "GPU:device_id", -1);

  int device_id = 0;

  if (user_device >= 0) {
    if (user_device >= ngpu)
      error("GPU:device_id=%d requested but only %d GPU(s) visible.",
            user_device, ngpu);

    device_id = user_device;

  } else {
    device_id = local_rank % ngpu;
  }

  int local_ranks_on_device = 1;

#ifdef WITH_MPI
  /*
   * Count ranks on this node that will map to the same GPU.
   * This is used only for conservative memory budgeting.
   */
  MPI_Comm local_comm2;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &local_comm2);

  const int my_device_id = device_id;
  int *all_devices = malloc((size_t)local_size * sizeof(int));

  if (all_devices == NULL)
    error("Failed to allocate local GPU mapping array.");

  MPI_Allgather(&my_device_id, 1, MPI_INT,
                all_devices, 1, MPI_INT, local_comm2);

  local_ranks_on_device = 0;

  for (int i = 0; i < local_size; i++) {
    if (all_devices[i] == my_device_id)
      local_ranks_on_device++;
  }

  free(all_devices);
  MPI_Comm_free(&local_comm2);
#endif

  gpu->device_id = device_id;
  gpu->local_mpi_rank = local_rank;
  gpu->local_mpi_size = local_size;
  gpu->local_ranks_on_device = local_ranks_on_device;

  GPUSetDevice(device_id);
  runner_gpu_check_error("GPUSetDevice");

  return device_id;
}

/**
 * @brief Initialise the GPU-specific state attached to a runner.
 *
 * @param r The runner whose GPU state to initialise.
 */
void runner_gpu_init(struct runner *r) {

  struct gpu_runner *gpu = &r->gpu;
  struct engine *e = r->e;
  
  #ifdef WITH_MPI
  MPI_Comm local_comm;
  int local_rank = 0, local_size = 1;

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                    MPI_INFO_NULL, &local_comm);
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);
  MPI_Comm_free(&local_comm);
  #else
  int local_rank = 0;
  #endif

  int ngpu = 0;
  GPUGetDeviceCount(&ngpu);

  if (ngpu <= 0)
    error("No GPU visible to MPI rank");

  const int gpu_id = local_rank % ngpu;
  GPUSetDevice(gpu_id);

  const int device_id = runner_gpu_select_device(r);
  
  runner_gpu_local_ranks_on_device_for_budget =
    gpu->local_ranks_on_device > 0 ? gpu->local_ranks_on_device : 1;

  GPUDeviceProp prop;
  GPUGetDeviceProperties(&prop, device_id);
  runner_gpu_check_error("GPUGetDeviceProperties");

  if (e->verbose && r->id == 0) {
    message("MPI rank %d local_rank=%d using GPU device %d "
          "(%d visible GPU(s), %d local rank(s) sharing this device)",
          e->nodeID,
          gpu->local_mpi_rank,
          gpu->device_id,
          gpu->local_mpi_size,
          gpu->local_ranks_on_device);
  }

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
    substream->pair_unique_cell_count = 0;

    substream->pair_unique_cells =
    malloc((size_t)gpu->grav_batch_ncells * sizeof(struct cell *));

    substream->pair_pair_i_h =
    malloc((size_t)gpu->grav_batch_ncells * sizeof(int));

    substream->pair_pair_j_h =
    malloc((size_t)gpu->grav_batch_ncells * sizeof(int));

    GPUMalloc((void **)&substream->pair_pair_i_d,
          (size_t)gpu->grav_batch_ncells * sizeof(int));

    GPUMalloc((void **)&substream->pair_pair_j_d,
          (size_t)gpu->grav_batch_ncells * sizeof(int));
          
    const int pair_capacity = gpu->grav_batch_ncells / 2;

    substream->pair_total_pair_active_count = 0;

    substream->pair_use_full_h =
    	malloc((size_t)pair_capacity * sizeof(int));

    GPUMalloc((void **)&substream->pair_use_full_d,
          (size_t)pair_capacity * sizeof(int));

    substream->pair_side_active_offsets_h =
        malloc((size_t)(2 * pair_capacity) * sizeof(int));

    GPUMalloc((void **)&substream->pair_side_active_offsets_d,
          (size_t)(2 * pair_capacity) * sizeof(int));
    
    const size_t pos_mass_bytes =
    (size_t)gpu->grav_batch_ncells *
    (size_t)gpu->grav_max_cell_size *
    sizeof(float4);

	const size_t h_bytes =
	    (size_t)gpu->grav_batch_ncells *
	    (size_t)gpu->grav_max_cell_size *
	    sizeof(float);

	GPUMalloc((void **)&substream->send_pair_pos_mass_d, pos_mass_bytes);
	GPUHostMalloc((void **)&substream->send_pair_pos_mass, pos_mass_bytes);

	GPUMalloc((void **)&substream->send_pair_h_d, h_bytes);
	GPUHostMalloc((void **)&substream->send_pair_h, h_bytes);
    
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
    substream->pair_total_pair_active_count = 0;

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
          
    substream->pair_cell_flags_h =
    malloc((size_t)gpu->grav_batch_ncells * sizeof(int));

    GPUMalloc((void **)&substream->pair_cell_flags_d,
          (size_t)gpu->grav_batch_ncells * sizeof(int));

    for (int j = 0; j < gpu->grav_batch_ncells; j++) {
      substream->pair_counts_h[j] = 0;
      substream->pair_offsets_h[j] = 0;
      substream->pair_active_counts_h[j] = 0;
      substream->pair_active_offsets_h[j] = 0;
      substream->pair_unique_cells[j] = NULL;
      substream->pair_pair_i_h[j] = 0;
      substream->pair_pair_j_h[j] = 0;
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
        substream->pair_cell_flags_h == NULL ||
        substream->pair_active_counts_h == NULL ||
        substream->pair_unique_cells == NULL ||
        substream->pair_pair_i_h == NULL ||
        substream->pair_pair_j_h == NULL ||
        substream->pair_use_full_h == NULL ||
	substream->pair_side_active_offsets_h == NULL ||
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

  runner_gpu_bind_device(r);

  struct gpu_runner *gpu = &r->gpu;

  for (int i = 0; i < gpu->nstreams; i++) {
    const int idx = (gpu->next_substream + i) % gpu->nstreams;
    struct gpu_runner_substream *substream = &gpu->substreams[idx];

    if (!substream->busy) {
      substream->busy = 1;
      gpu->next_substream = (idx + 1) % gpu->nstreams;
      return substream;
    }

    const GPUError qerr = GPUEventQuery(substream->done);

    if (qerr == GPU_SUCCESS) {
      substream->busy = 1;
      gpu->next_substream = (idx + 1) % gpu->nstreams;
      return substream;
    }

#if defined(WITH_CUDA)
    if (qerr != cudaErrorNotReady)
      error("runner_gpu_acquire_substream: GPUEventQuery failed: %s",
            GPUGetErrorString(qerr));
#elif defined(WITH_HIP)
    if (qerr != hipErrorNotReady)
      error("runner_gpu_acquire_substream: GPUEventQuery failed: %s",
            GPUGetErrorString(qerr));
#endif
  }

  struct gpu_runner_substream *substream =
      &gpu->substreams[gpu->next_substream];

  const GPUError serr = GPUEventSynchronize(substream->done);
  if (serr != GPU_SUCCESS)
    error("runner_gpu_acquire_substream: GPUEventSynchronize failed: %s",
          GPUGetErrorString(serr));

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

  runner_gpu_bind_device(r);

  struct gpu_runner *gpu = &r->gpu;

  for (int i = 0; i < gpu->nstreams; i++) {

    struct gpu_runner_substream *substream = &gpu->substreams[i];

    /* Pair buffers */
    GPUFreeHost(substream->send_pair_pos_mass);
    GPUFree(substream->send_pair_pos_mass_d);

    GPUFreeHost(substream->send_pair_h);
    GPUFree(substream->send_pair_h_d);;

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
    
    free(substream->pair_cell_flags_h);
    GPUFree(substream->pair_cell_flags_d);
    
    free(substream->pair_unique_cells);
    free(substream->pair_pair_i_h);
    free(substream->pair_pair_j_h);

    GPUFree(substream->pair_pair_i_d);
    GPUFree(substream->pair_pair_j_d);
    
    free(substream->pair_use_full_h);
    GPUFree(substream->pair_use_full_d);

    free(substream->pair_side_active_offsets_h);
    GPUFree(substream->pair_side_active_offsets_d);

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

  runner_gpu_bind_device(r);

  enum runner_gpu_task_type result = regular_task;

  for (int l = 0; l < r->gpu.nstreams; l++) {
    struct gpu_runner_substream *substream = &r->gpu.substreams[l];
    const int nslots = substream->grav_batch_self_count;
    const int total = substream->self_total_count;
    const int max_cell_size = r->gpu.grav_max_cell_size;

    if (nslots == 0) continue;

    double h2d_s = 0.0;
    double kernel_s = 0.0;
    double d2h_s = 0.0;
    double unpack_s = 0.0;

    /*GPUEvent h2d_start, h2d_stop;
    GPUEvent kernel_start, kernel_stop;
    GPUEvent d2h_start, d2h_stop;

    GPUEventCreate(&h2d_start);
    GPUEventCreate(&h2d_stop);
    GPUEventCreate(&kernel_start);
    GPUEventCreate(&kernel_stop);
    GPUEventCreate(&d2h_start);
    GPUEventCreate(&d2h_stop);*/

    /* ===================== H2D ===================== */

    //GPUEventRecord(h2d_start, substream->stream);

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

    //GPUEventRecord(h2d_stop, substream->stream);

    /* ===================== KERNEL ===================== */

    //GPUEventRecord(kernel_start, substream->stream);

    runner_doself_grav_pp_flush(
        r, substream, nslots, substream->self_max_active_count, max_cell_size,
        substream->stream);

    //GPUEventRecord(kernel_stop, substream->stream);

    /* ===================== D2H ===================== */

    //GPUEventRecord(d2h_start, substream->stream);

    GPUMemcpyAsync(
        substream->recv_self_active,
        substream->recv_self_active_d,
        (size_t)substream->self_total_active_count *
            sizeof(struct gravity_gpu_values_recv),
        GPU_MEMCPY_DEVICE_TO_HOST,
        substream->stream);

    //GPUEventRecord(d2h_stop, substream->stream);

    GPUEventRecord(substream->done, substream->stream);
    GPUEventSynchronize(substream->done);

    /*h2d_s = runner_gpu_event_elapsed_s(h2d_start, h2d_stop);
    kernel_s = runner_gpu_event_elapsed_s(kernel_start, kernel_stop);
    d2h_s = runner_gpu_event_elapsed_s(d2h_start, d2h_stop);*/

    /* ===================== UNPACK ===================== */

    const double unpack_t0 = runner_gpu_walltime_s();

    for (int j = 0; j < nslots; j++) {
      struct cell *c_unpack = substream->grav_cells_self[j];
      
      if (c_unpack == NULL)
        error("GPU self unpack received NULL cell.");

      if (c_unpack->nodeID != r->e->nodeID)
        error("GPU self unpack attempted to write a foreign cell.");

      if (!cell_is_active_gravity(c_unpack, r->e))
        continue;

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

    unpack_s = runner_gpu_walltime_s() - unpack_t0;

    const double self_pack_s = runner_gpu_self_pack_time_s;
    runner_gpu_self_pack_time_s = 0.0;

    /*runner_gpu_write_timing_row("self", (long long)r->e->step, r->id, r->qid, nslots, total,
                                self_pack_s, h2d_s, kernel_s, d2h_s,
                                unpack_s);*/

    /*GPUEventDestroy(h2d_start);
    GPUEventDestroy(h2d_stop);
    GPUEventDestroy(kernel_start);
    GPUEventDestroy(kernel_stop);
    GPUEventDestroy(d2h_start);
    GPUEventDestroy(d2h_stop);*/

    runner_gpu_complete_self_batch(r, &r->e->sched, substream, NULL);

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

  runner_gpu_bind_device(r);

  enum runner_gpu_task_type result = regular_task;

  /*message("runner_gpu_flush_leftover_pair entry: qid=%d "
        "pair_left=%d",
        r->qid,
        r->e->sched.queues[r->qid].gpu_pair_tasks_left);*/

for (int l = 0; l < r->gpu.nstreams; l++) {
  struct gpu_runner_substream *ss = &r->gpu.substreams[l];

  /*message("runner_gpu_flush_leftover_pair stream=%d before: "
          "pair_batch=%d pair_unique_cells=%d pair_total_count=%d "
          "pair_total_active=%d pair_total_pair_active=%d",
          l,
          ss->grav_batch_pair_count,
          ss->pair_unique_cell_count,
          ss->pair_total_count,
          ss->pair_total_active_count,
          ss->pair_total_pair_active_count);*/

    if (ss->grav_batch_pair_count == 0) continue;

    runner_dopair_grav_pp_flush(
        r, ss,
        ss->grav_cells_pair,
        ss->grav_tasks_pair,
        NULL,
        r->gpu.grav_batch_ncells,
        r->gpu.grav_max_cell_size,
        ss->stream);

    result = flushed_pair_task;
    
    /*message("runner_gpu_flush_leftover_pair exit: qid=%d "
        "pair_left=%d",
        r->qid,
        r->e->sched.queues[r->qid].gpu_pair_tasks_left);*/
  }

  return result;
}
