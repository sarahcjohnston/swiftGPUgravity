/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 *                    Matthieu Schaller (schaller@strw.leidenuniv.nl)
 *               2015 Peter W. Draper (p.w.draper@durham.ac.uk)
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

/* Config parameters. */
#include <config.h>

/* MPI headers. */
#ifdef WITH_MPI
#include <mpi.h>
#endif

/* This object's header. */
#include "runner.h"

/* GPU headers */
#include <hip/hip_runtime_api.h>

/* Local headers. */
#include "engine.h"
#include "feedback.h"
#include "scheduler.h"
#include "space_getsid.h"
#include "timers.h"

/* Import the gravity loop functions. */
#include "runner_doiact_grav.h"

/* Import the density loop functions. */
#define FUNCTION density
#define FUNCTION_TASK_LOOP TASK_LOOP_DENSITY
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"

/* Import the gradient loop functions (if required). */
#ifdef EXTRA_HYDRO_LOOP
#define FUNCTION gradient
#define FUNCTION_TASK_LOOP TASK_LOOP_GRADIENT
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"
#endif

/* Import the force loop functions. */
#define FUNCTION force
#define FUNCTION_TASK_LOOP TASK_LOOP_FORCE
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"

/* Import the limiter loop functions. */
#define FUNCTION limiter
#define FUNCTION_TASK_LOOP TASK_LOOP_LIMITER
#include "runner_doiact_limiter.h"
#include "runner_doiact_undef.h"

/* Import the stars density loop functions. */
#define FUNCTION density
#define FUNCTION_TASK_LOOP TASK_LOOP_DENSITY
#include "runner_doiact_stars.h"
#include "runner_doiact_undef.h"

#ifdef EXTRA_STAR_LOOPS

/* Import the stars prepare1 loop functions. */
#define FUNCTION prep1
#define FUNCTION_TASK_LOOP TASK_LOOP_STARS_PREP1
#include "runner_doiact_stars.h"
#include "runner_doiact_undef.h"

/* Import the stars prepare2 loop functions. */
#define FUNCTION prep2
#define FUNCTION_TASK_LOOP TASK_LOOP_STARS_PREP2
#include "runner_doiact_stars.h"
#include "runner_doiact_undef.h"

#endif /* EXTRA_STAR_LOOPS */

/* Import the stars feedback loop functions. */
#define FUNCTION feedback
#define FUNCTION_TASK_LOOP TASK_LOOP_FEEDBACK
#include "runner_doiact_stars.h"
#include "runner_doiact_undef.h"

/* Import the black hole density loop functions. */
#define FUNCTION density
#define FUNCTION_TASK_LOOP TASK_LOOP_DENSITY
#include "runner_doiact_black_holes.h"
#include "runner_doiact_undef.h"

/* Import the black hole feedback loop functions. */
#define FUNCTION swallow
#define FUNCTION_TASK_LOOP TASK_LOOP_SWALLOW
#include "runner_doiact_black_holes.h"
#include "runner_doiact_undef.h"

/* Import the black hole feedback loop functions. */
#define FUNCTION feedback
#define FUNCTION_TASK_LOOP TASK_LOOP_FEEDBACK
#include "runner_doiact_black_holes.h"
#include "runner_doiact_undef.h"

/* Import the sink density loop functions. */
#define FUNCTION density
#define FUNCTION_TASK_LOOP TASK_LOOP_DENSITY
#include "runner_doiact_sinks.h"
#include "runner_doiact_undef.h"

/* Import the sink swallow loop functions. */
#define FUNCTION swallow
#define FUNCTION_TASK_LOOP TASK_LOOP_SWALLOW
#include "runner_doiact_sinks.h"
#include "runner_doiact_undef.h"

/* Import the RT gradient loop functions */
#define FUNCTION rt_gradient
#define FUNCTION_TASK_LOOP TASK_LOOP_RT_GRADIENT
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"

/* Import the RT transport (force) loop functions. */
#define FUNCTION rt_transport
#define FUNCTION_TASK_LOOP TASK_LOOP_RT_TRANSPORT
#include "runner_doiact_hydro.h"
#include "runner_doiact_undef.h"

/* Import the GPU functions needed. */
#include "active.h"
#include "gpu_functions.h"

#include <stdatomic.h>

struct task* enqueue_dependencies(struct scheduler* s, struct task* t) {
  /* Loop through the dependencies and add them to a queue if
         they are ready. */
  for (int k = 0; k < t->nr_unlock_tasks; k++) {
    struct task* t2 = t->unlock_tasks[k];
    if (t2->skip) continue;
    const int res = atomic_dec(&t2->wait);
    if (res < 1) {
      error("Negative wait!");
    } else if (res == 1) {
      scheduler_enqueue(s, t2);
    }
  }
  return NULL;
}

extern void self_pp_offload_new(
    int periodic, float rmax_i, double min_trunc, const float* r_s_inv,
    const int* gcount_i, const int* gcount_padded_i, int ci_active,
    struct gravity_gpu_values_send* gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_d, int ncells,
    int max_cell_size, hipStream_t stream);
extern void pair_pp_offload_new(
    int periodic, float rmax_i, float rmax_j, double min_trunc,
    const float* r_s_inv, const int* gcount_i, const int* gcount_padded_i,
    const int* gcount_j, const int* gcount_padded_j, int ci_active,
    int cj_active, float dim_0, float dim_1, float dim_2, int symmetric,
    struct gravity_gpu_values_send* gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_d, int ncells,
    int max_cell_size, hipStream_t stream);
;
/**
 * @brief The #runner main thread routine.
 *
 * @param data A pointer to this thread's data.
 */
void* runner_main(void* data) {

  struct runner* r = (struct runner*)data;
  struct engine* e = r->e;
  struct scheduler* sched = &e->sched;

  const int max_cell_size = r->gpu.grav_max_cell_size;
  const int ncells = r->gpu.grav_batch_ncells;

  /* Main loop. */
  while (1) {

    /* Wait at the barrier. */
    engine_barrier(e);

    /* Can we go home yet? */
    if (e->step_props & engine_step_prop_done) break;

    /* Re-set the pointer to the previous task, as there is none. */
    struct task* t = NULL;
    struct task* prev = NULL;

    /* Reset the batch counts. */
    r->gpu.grav_batch_self_count = 0;
    r->gpu.grav_batch_pair_count = 0;

    int ntasks;

    int packed = 0;

    /* Loop while there are tasks... */
    while (1) {

      /* If there's no old task, try to get a new one. */
      if (t == NULL) {

        /* Get the task. */
        TIMER_TIC
        t = scheduler_gettask(sched, r->qid, prev);
        TIMER_TOC(timer_gettask);

        /* Did I get anything? */
        if (t == NULL) {

          if (r->gpu.grav_batch_self_count != 0)
            error("qid=%d going idle with %d packed self tasks", r->qid,
                  r->gpu.grav_batch_self_count);

          if (r->gpu.grav_batch_pair_count != 0)
            error("qid=%d going idle with %d packed pair tasks", r->qid,
                  r->gpu.grav_batch_pair_count);

          break;
        }
      }

      /* Get the cells. */
      struct cell* ci = t->ci;
      struct cell* cj = t->cj;

#ifdef SWIFT_DEBUG_TASKS
      /* Mark the thread we run on */
      t->rid = r->cpuid;

      /* And recover the pair direction */
      if (t->type == task_type_pair) {
        struct cell* ci_temp = ci;
        struct cell* cj_temp = cj;
        double shift[3];
        t->sid = space_getsid_and_swap_cells(e->s, &ci_temp, &cj_temp, shift);
      } else {
        t->sid = -1;
      }
#endif

#ifdef SWIFT_DEBUG_CHECKS
      /* Check that we haven't scheduled an inactive task */
      t->ti_run = e->ti_current;
      /* Store the task that will be running (for debugging only) */
      r->t = t;
#endif

      const ticks task_beg = getticks();

      /* Different types of tasks... */
      switch (t->type) {

        case task_type_self:
          if (t->subtype == task_subtype_grav) {

            // make long arrays with all the values
            struct gravity_cache* const ci_cache = &r->ci_gravity_cache;

            const int gcount = ci->grav.count;
            const int gcount_padded = gcount - (gcount % VEC_SIZE) + VEC_SIZE;

            if (gcount > max_cell_size)
              error(
                  "More particles than allocated memory! %i particles in cell "
                  "and only %i slots in memory available. Increase the number "
                  "of top level cells!",
                  gcount, max_cell_size);

            const double loc[3] = {ci->loc[0] + 0.5 * ci->width[0],
                                   ci->loc[1] + 0.5 * ci->width[1],
                                   ci->loc[2] + 0.5 * ci->width[2]};

            gravity_cache_populate_no_mpole(
                e->max_active_bin, ci_cache, ci->grav.parts, gcount,
                gcount_padded, loc, ci, e->gravity_properties);

            while (cell_glocktree(ci)) {
              ; /* spin until we acquire the lock */
            }

            hipEvent_t startpack, stoppack;
            hipEventCreate(&startpack);
            hipEventCreate(&stoppack);

            hipEventRecord(startpack, r->gpu.stream);

            {
              TIMER_TIC;
              for (int i = 0; i < gcount; i++) {
                r->gpu
                    .gravity_gpu_values_send_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .h_i = ci_cache->epsilon[i];
                r->gpu
                    .gravity_gpu_values_send_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .h_j = ci_cache->epsilon[i];
                r->gpu
                    .gravity_gpu_values_send_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .mass_i = ci_cache->m[i];
                r->gpu
                    .gravity_gpu_values_send_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .mass_j = ci_cache->m[i];
                r->gpu
                    .gravity_gpu_values_send_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .x_i = ci_cache->x[i];
                r->gpu
                    .gravity_gpu_values_send_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .x_j = ci_cache->x[i];
                r->gpu
                    .gravity_gpu_values_send_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .y_i = ci_cache->y[i];
                r->gpu
                    .gravity_gpu_values_send_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .y_j = ci_cache->y[i];
                r->gpu
                    .gravity_gpu_values_send_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .z_i = ci_cache->z[i];
                r->gpu
                    .gravity_gpu_values_send_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .z_j = ci_cache->z[i];
                r->gpu
                    .gravity_gpu_values_send_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .active_i = ci_cache->active[i];
                r->gpu
                    .gravity_gpu_values_send_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .active_j = ci_cache->active[i];
              }

              for (int i = 0; i < max_cell_size; i++) {
                r->gpu
                    .gravity_gpu_values_recv_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .a_x_i = 0;
                r->gpu
                    .gravity_gpu_values_recv_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .a_y_i = 0;
                r->gpu
                    .gravity_gpu_values_recv_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .a_z_i = 0;
                r->gpu
                    .gravity_gpu_values_recv_self[i +
                                                  r->gpu.grav_batch_self_count *
                                                      max_cell_size]
                    .pot_i = 0;
              }

              TIMER_TOC(timer_doself_grav_pp);
            }  // TIMER_TOC(timer_gpu_pack);

            // store the address of the cells and tasks we are working on
            r->gpu.grav_cells_self[r->gpu.grav_batch_self_count] = ci;
            r->gpu.grav_tasks_self[r->gpu.grav_batch_self_count] = t;

            for (int i = 0; i < gcount; i++) {
              r->gpu
                  .gravity_gpu_values_send_self[i +
                                                r->gpu.grav_batch_self_count *
                                                    max_cell_size]
                  .cell_active = cell_is_active_gravity(ci, e);
              r->gpu
                  .gravity_gpu_values_send_self[i +
                                                r->gpu.grav_batch_self_count *
                                                    max_cell_size]
                  .gcounts = gcount;
            }

            // update that we packed a cell into our array
            r->gpu.grav_batch_self_count += 1;

            gravity_cache_zero_output(ci_cache, gcount_padded);  // ADDED HERE?

            cell_gunlocktree(ci);

            lock_lock(&sched->queues[r->qid].lock);
            sched->queues[r->qid].gpu_self_tasks_left--;
            (void)lock_unlock(&sched->queues[r->qid].lock);

            int acc = 0;
#ifdef SWIFT_DEBUG_CHECKS
            /* Update the interaction counter if it's not a padded gpart */
            for (int j = 0; j < gcount; j++) {
              for (int i = 0; i < gcount; i++) {
                if (i == j) continue;
                // if (!gpart_is_inhibited(&grav_cells[j]->grav.parts[i], e))
                acc++;
                accumulate_inc_ll(&ci->grav.parts[j].num_interacted);
              }
            }
#endif

            // set what happens when the pack count is reached
            if (r->gpu.grav_batch_self_count >= ncells) {
              hipEvent_t startcopyH2D, stopcopyH2D;
              hipEventCreate(&startcopyH2D);
              hipEventCreate(&stopcopyH2D);

              hipEventRecord(startcopyH2D, r->gpu.stream);

              {
                TIMER_TIC;

                // now copy all the arrays to the device
                // gravity_gpu_H2D(gravity_gpu_values_h, gravity_gpu_values_d,
                // ncells, max_cell_size, r->gpu.stream);
                hipMemcpyAsync(r->gpu.gravity_gpu_values_send_self_d,
                               r->gpu.gravity_gpu_values_send_self,
                               ncells * max_cell_size *
                                   sizeof(struct gravity_gpu_values_send),
                               hipMemcpyHostToDevice, r->gpu.stream);
                hipMemcpyAsync(r->gpu.gravity_gpu_values_recv_self_d,
                               r->gpu.gravity_gpu_values_recv_self,
                               ncells * max_cell_size *
                                   sizeof(struct gravity_gpu_values_recv),
                               hipMemcpyHostToDevice, r->gpu.stream);

                hipEventRecord(stopcopyH2D, r->gpu.stream);

                hipError_t err2 = hipGetLastError();
                if (err2 != hipSuccess)
                  printf("Error2: %s\n", hipGetErrorString(err2));

                hipEvent_t startker, stopker;
                hipEventCreate(&startker);
                hipEventCreate(&stopker);

                hipEventRecord(startker, r->gpu.stream);

                runner_doself_recursive_grav_new(
                    r, ci, 1, r->gpu.gravity_gpu_values_send_self_d,
                    r->gpu.gravity_gpu_values_recv_self_d, ncells,
                    max_cell_size, r->gpu.stream);

                hipEventRecord(stopker, r->gpu.stream);

                // hipDeviceSynchronize();

                hipEvent_t startcopyD2H, stopcopyD2H;
                hipEventCreate(&startcopyD2H);
                hipEventCreate(&stopcopyD2H);

                hipEventRecord(startcopyD2H, r->gpu.stream);

                // copy the arrays from device to host
                // gravity_gpu_D2H(gravity_gpu_values_h, gravity_gpu_values_d,
                // ncells, max_cell_size, r->gpu.stream);
                hipMemcpyAsync(r->gpu.gravity_gpu_values_recv_self,
                               r->gpu.gravity_gpu_values_recv_self_d,
                               ncells * max_cell_size *
                                   sizeof(struct gravity_gpu_values_recv),
                               hipMemcpyDeviceToHost, r->gpu.stream);

                hipEventRecord(stopcopyD2H, r->gpu.stream);

                hipStreamSynchronize(r->gpu.stream);  // THIS ONE IS NEEDED!

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
              if (err3 != hipSuccess)
                printf("Error3: %s\n", hipGetErrorString(err3));

              {
                TIMER_TIC;

                /*send results back to relevant cell structs*/
                for (int j = 0; j < ncells; j++) {
                  while (cell_glocktree(r->gpu.grav_cells_self[j])) {
                    ; /* spin until we acquire the lock */
                  }
                  for (int i = 0;
                       i <
                       r->gpu.gravity_gpu_values_send_self[j * max_cell_size]
                           .gcounts;
                       i++) {
                    r->gpu.grav_cells_self[j]->grav.parts[i].a_grav[0] +=
                        r->gpu
                            .gravity_gpu_values_recv_self[i + j * max_cell_size]
                            .a_x_i;
                    r->gpu.grav_cells_self[j]->grav.parts[i].a_grav[1] +=
                        r->gpu
                            .gravity_gpu_values_recv_self[i + j * max_cell_size]
                            .a_y_i;
                    r->gpu.grav_cells_self[j]->grav.parts[i].a_grav[2] +=
                        r->gpu
                            .gravity_gpu_values_recv_self[i + j * max_cell_size]
                            .a_z_i;
                    r->gpu.grav_cells_self[j]->grav.parts[i].potential +=
                        r->gpu
                            .gravity_gpu_values_recv_self[i + j * max_cell_size]
                            .pot_i;
                  }
                  cell_gunlocktree(r->gpu.grav_cells_self[j]);
                }

                TIMER_TOC(timer_doself_grav_pp);
              }  // TIMER_TOC(timer_gpu_unpack);

              // hipDeviceSynchronize();

              for (int i = 0; i < ncells; i++) {
                scheduler_done(sched, r->gpu.grav_tasks_self[i]);
              }

              // reset counter for next pack
              for (int i = 0; i < ncells; i++) {
                r->gpu.grav_cells_self[i] = NULL;
                r->gpu.grav_tasks_self[i] = NULL;
              }
              r->gpu.grav_batch_self_count = 0;
            }

          }

          else if (t->subtype == task_subtype_external_grav)
            runner_do_grav_external(r, ci, 1);
          else if (t->subtype == task_subtype_density)
            runner_dosub_self1_density(r, ci, /*below_h_max=*/0, 1);
#ifdef EXTRA_HYDRO_LOOP
          else if (t->subtype == task_subtype_gradient)
#ifdef EXTRA_HYDRO_LOOP_TYPE2
            runner_dosub_self2_gradient(r, ci, /*below_h_max=*/0, 1);
#else
            runner_dosub_self1_gradient(r, ci, /*below_h_max=*/0, 1);
#endif
#endif
          else if (t->subtype == task_subtype_force)
            runner_dosub_self2_force(r, ci, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_limiter)
            runner_dosub_self1_limiter(r, ci, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_stars_density)
            runner_dosub_self_stars_density(r, ci, /*below_h_max=*/0, 1);
#ifdef EXTRA_STAR_LOOPS
          else if (t->subtype == task_subtype_stars_prep1)
            runner_dosub_self_stars_prep1(r, ci, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_stars_prep2)
            runner_dosub_self_stars_prep2(r, ci, /*below_h_max=*/0, 1);
#endif
          else if (t->subtype == task_subtype_stars_feedback)
            runner_dosub_self_stars_feedback(r, ci, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_bh_density)
            runner_dosub_self_bh_density(r, ci, 1);
          else if (t->subtype == task_subtype_bh_swallow)
            runner_dosub_self_bh_swallow(r, ci, 1);
          else if (t->subtype == task_subtype_do_gas_swallow)
            runner_do_gas_swallow_self(r, ci, 1);
          else if (t->subtype == task_subtype_do_bh_swallow)
            runner_do_bh_swallow_self(r, ci, 1);
          else if (t->subtype == task_subtype_bh_feedback)
            runner_dosub_self_bh_feedback(r, ci, 1);
          else if (t->subtype == task_subtype_rt_gradient)
            runner_dosub_self1_rt_gradient(r, ci, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_rt_transport)
            runner_dosub_self2_rt_transport(r, ci, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_sink_density)
            runner_dosub_self_sinks_density(r, ci, 1);
          else if (t->subtype == task_subtype_sink_swallow)
            runner_dosub_self_sinks_swallow(r, ci, 1);
          else if (t->subtype == task_subtype_sink_do_gas_swallow)
            runner_do_sinks_gas_swallow_self(r, ci, 1);
          else if (t->subtype == task_subtype_sink_do_sink_swallow)
            runner_do_sinks_sink_swallow_self(r, ci, 1);
          else
            error("Unknown/invalid task subtype (%s/%s).",
                  taskID_names[t->type], subtaskID_names[t->subtype]);
          break;

        case task_type_pair:
          if (t->subtype == task_subtype_grav) {

            runner_dopair_recursive_grav_new(
                r, ci, cj, 1, r->gpu.gravity_gpu_values_send_pair,
                r->gpu.gravity_gpu_values_send_pair_d,
                r->gpu.gravity_gpu_values_recv_pair,
                r->gpu.gravity_gpu_values_recv_pair_d, r->gpu.grav_cells_pair,
                r->gpu.grav_tasks_pair, t, sched, ncells, max_cell_size,
                &packed, r->gpu.stream);

          } else if (t->subtype == task_subtype_density)
            runner_dosub_pair1_density(r, ci, cj, /*below_h_max=*/0, 1);
#ifdef EXTRA_HYDRO_LOOP
          else if (t->subtype == task_subtype_gradient)
#ifdef EXTRA_HYDRO_LOOP_TYPE2
            runner_dosub_pair2_gradient(r, ci, cj, /*below_h_max=*/0, 1);
#else
            runner_dosub_pair1_gradient(r, ci, cj, /*below_h_max=*/0, 1);
#endif
#endif
          else if (t->subtype == task_subtype_force)
            runner_dosub_pair2_force(r, ci, cj, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_limiter)
            runner_dosub_pair1_limiter(r, ci, cj, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_stars_density)
            runner_dosub_pair_stars_density(r, ci, cj, /*below_h_max=*/0, 1);
#ifdef EXTRA_STAR_LOOPS
          else if (t->subtype == task_subtype_stars_prep1)
            runner_dosub_pair_stars_prep1(r, ci, cj, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_stars_prep2)
            runner_dosub_pair_stars_prep2(r, ci, cj, /*below_h_max=*/0, 1);
#endif
          else if (t->subtype == task_subtype_stars_feedback)
            runner_dosub_pair_stars_feedback(r, ci, cj, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_bh_density)
            runner_dosub_pair_bh_density(r, ci, cj, 1);
          else if (t->subtype == task_subtype_bh_swallow)
            runner_dosub_pair_bh_swallow(r, ci, cj, 1);
          else if (t->subtype == task_subtype_do_gas_swallow)
            runner_do_gas_swallow_pair(r, ci, cj, 1);
          else if (t->subtype == task_subtype_do_bh_swallow)
            runner_do_bh_swallow_pair(r, ci, cj, 1);
          else if (t->subtype == task_subtype_bh_feedback)
            runner_dosub_pair_bh_feedback(r, ci, cj, 1);
          else if (t->subtype == task_subtype_rt_gradient)
            runner_dosub_pair1_rt_gradient(r, ci, cj, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_rt_transport)
            runner_dosub_pair2_rt_transport(r, ci, cj, /*below_h_max=*/0, 1);
          else if (t->subtype == task_subtype_sink_density)
            runner_dosub_pair_sinks_density(r, ci, cj, 1);
          else if (t->subtype == task_subtype_sink_swallow)
            runner_dosub_pair_sinks_swallow(r, ci, cj, 1);
          else if (t->subtype == task_subtype_sink_do_gas_swallow)
            runner_do_sinks_gas_swallow_pair(r, ci, cj, 1);
          else if (t->subtype == task_subtype_sink_do_sink_swallow)
            runner_do_sinks_sink_swallow_pair(r, ci, cj, 1);
          else
            error("Unknown/invalid task subtype (%s/%s).",
                  taskID_names[t->type], subtaskID_names[t->subtype]);
          break;

        case task_type_sort:
          /* Cleanup only if any of the indices went stale. */
          runner_do_hydro_sort(
              r, ci, t->flags,
              ci->hydro.dx_max_sort_old > space_maxreldx * ci->dmin,
              /*lock=*/0, cell_get_flag(ci, cell_flag_rt_requests_sort),
              /*clock=*/1);
          /* Reset the sort flags as our work here is done. */
          t->flags = 0;
          break;
        case task_type_rt_sort:
          /* Cleanup only if any of the indices went stale.
           * NOTE: we check whether we reset the sort flags when the
           * recv tasks are running. Cells without an RT recv task
           * don't have rt_sort tasks. */
          runner_do_hydro_sort(
              r, ci, t->flags,
              ci->hydro.dx_max_sort_old > space_maxreldx * ci->dmin,
              /*lock=*/0, /*rt_requests_sorts=*/1, /*clock=*/1);
          /* Reset the sort flags as our work here is done. */
          t->flags = 0;
          break;
        case task_type_stars_sort:
          /* Cleanup only if any of the indices went stale. */
          runner_do_stars_sort(
              r, ci, t->flags,
              ci->stars.dx_max_sort_old > space_maxreldx * ci->dmin, 1);
          /* Reset the sort flags as our work here is done. */
          t->flags = 0;
          break;
        case task_type_init_grav:
          runner_do_init_grav(r, ci, 1);
          break;
        case task_type_ghost:
          runner_do_ghost(r, ci, 1);
          break;
#ifdef EXTRA_HYDRO_LOOP
        case task_type_extra_ghost:
          runner_do_extra_ghost(r, ci, 1);
          break;
#endif
        case task_type_stars_ghost:
          runner_do_stars_ghost(r, ci, 1);
          break;
        case task_type_bh_density_ghost:
          runner_do_black_holes_density_ghost(r, ci, 1);
          break;
        case task_type_bh_swallow_ghost3:
          runner_do_black_holes_swallow_ghost(r, ci, 1);
          break;
        case task_type_sink_density_ghost:
          runner_do_sinks_density_ghost(r, ci, 1);
          break;
        case task_type_drift_part:
          runner_do_drift_part(r, ci, 1);
          break;
        case task_type_drift_spart:
          runner_do_drift_spart(r, ci, 1);
          break;
        case task_type_drift_sink:
          runner_do_drift_sink(r, ci, 1);
          break;
        case task_type_drift_bpart:
          runner_do_drift_bpart(r, ci, 1);
          break;
        case task_type_drift_gpart:
          runner_do_drift_gpart(r, ci, 1);
          break;
        case task_type_kick1:
          runner_do_kick1(r, ci, 1);
          break;
        case task_type_kick2:
          runner_do_kick2(r, ci, 1);
          break;
        case task_type_end_hydro_force:
          runner_do_end_hydro_force(r, ci, 1);
          break;
        case task_type_end_grav_force:
          runner_do_end_grav_force(r, ci, 1);
          break;
        case task_type_csds:
          runner_do_csds(r, ci, 1);
          break;
        case task_type_timestep:
          runner_do_timestep(r, ci, 1);
          break;
        case task_type_timestep_limiter:
          runner_do_limiter(r, ci, 0, 1);
          break;
        case task_type_timestep_sync:
          runner_do_sync(r, ci, 0, 1);
          break;
        case task_type_collect:
          runner_do_timestep_collect(r, ci, 1);
          break;
        case task_type_rt_collect_times:
          runner_do_collect_rt_times(r, ci, 1);
          break;
#ifdef WITH_MPI
        case task_type_send:
          if (t->subtype == task_subtype_tend) {
            free(t->buff);
          } else if (t->subtype == task_subtype_sf_counts) {
            free(t->buff);
          } else if (t->subtype == task_subtype_grav_counts) {
            free(t->buff);
          } else if (t->subtype == task_subtype_part_swallow) {
            free(t->buff);
          } else if (t->subtype == task_subtype_bpart_merger) {
            free(t->buff);
          } else if (t->subtype == task_subtype_limiter) {
            free(t->buff);
          } else if (t->subtype == task_subtype_gpart) {
            free(t->buff);
          } else if (t->subtype == task_subtype_fof) {
            free(t->buff);
          }
          break;
        case task_type_recv:
          if (t->subtype == task_subtype_tend) {
            cell_unpack_end_step(ci, (struct pcell_step*)t->buff);
            free(t->buff);
          } else if (t->subtype == task_subtype_sf_counts) {
            cell_unpack_sf_counts(ci, (struct pcell_sf_stars*)t->buff);
            cell_clear_stars_sort_flags(ci, /*clear_unused_flags=*/0);
            free(t->buff);
          } else if (t->subtype == task_subtype_grav_counts) {
            cell_unpack_grav_counts(ci, (struct pcell_sf_grav*)t->buff);
            free(t->buff);
          } else if (t->subtype == task_subtype_xv) {
            runner_do_recv_part(r, ci, 1, 1);
          } else if (t->subtype == task_subtype_rho) {
            runner_do_recv_part(r, ci, 0, 1);
          } else if (t->subtype == task_subtype_gradient) {
            runner_do_recv_part(r, ci, 0, 1);
          } else if (t->subtype == task_subtype_rt_gradient) {
            runner_do_recv_part(r, ci, 2, 1);
          } else if (t->subtype == task_subtype_rt_transport) {
            runner_do_recv_part(r, ci, -1, 1);
          } else if (t->subtype == task_subtype_part_swallow) {
            cell_unpack_part_swallow(ci,
                                     (struct black_holes_part_data*)t->buff);
            free(t->buff);
          } else if (t->subtype == task_subtype_bpart_merger) {
            cell_unpack_bpart_swallow(ci,
                                      (struct black_holes_bpart_data*)t->buff);
            free(t->buff);
          } else if (t->subtype == task_subtype_limiter) {
            /* Nothing to do here. Unpacking done in a separate task */
          } else if (t->subtype == task_subtype_gpart) {
            runner_do_recv_gpart(r, ci, 1);
          } else if (t->subtype == task_subtype_fof) {
            /* Nothing to do here. */
          } else if (t->subtype == task_subtype_spart_density) {
            runner_do_recv_spart(r, ci, 1, 1);
          } else if (t->subtype == task_subtype_part_prep1) {
            runner_do_recv_part(r, ci, 0, 1);
          } else if (t->subtype == task_subtype_spart_prep2) {
            runner_do_recv_spart(r, ci, 0, 1);
          } else if (t->subtype == task_subtype_bpart_rho) {
            runner_do_recv_bpart(r, ci, 1, 1);
          } else if (t->subtype == task_subtype_bpart_feedback) {
            runner_do_recv_bpart(r, ci, 0, 1);
          } else {
            error("Unknown/invalid task subtype (%d).", t->subtype);
          }
          break;

        case task_type_pack:
          if (t->subtype == task_subtype_limiter) {
            runner_do_pack_limiter(r, ci, &t->buff, 1);
            task_get_unique_dependent(t)->buff = t->buff;
          } else if (t->subtype == task_subtype_gpart) {
            runner_do_pack_gpart(r, ci, &t->buff, 1);
            task_get_unique_dependent(t)->buff = t->buff;
          } else if (t->subtype == task_subtype_fof) {
            runner_do_pack_fof(r, ci, &t->buff, 1);
            task_get_unique_dependent(t)->buff = t->buff;
          } else {
            error("Unknown/invalid task subtype (%d).", t->subtype);
          }
          break;
        case task_type_unpack:
          if (t->subtype == task_subtype_limiter) {
            runner_do_unpack_limiter(r, ci, t->buff, 1);
          } else {
            error("Unknown/invalid task subtype (%d).", t->subtype);
          }
          break;
#endif
        case task_type_grav_down:
          runner_do_grav_down(r, t->ci, 1);
          break;
        case task_type_grav_long_range:
          runner_do_grav_long_range(r, t->ci, 1);
          break;
        case task_type_grav_mm:
          runner_dopair_grav_mm_progenies(r, t->flags, t->ci, t->cj);
          break;
        case task_type_cooling:
          runner_do_cooling(r, t->ci, 1);
          break;
        case task_type_star_formation:
          runner_do_star_formation(r, t->ci, 1);
          break;
        case task_type_star_formation_sink:
          runner_do_star_formation_sink(r, t->ci, 1);
          break;
        case task_type_stars_resort:
          runner_do_stars_resort(r, t->ci, 1);
          break;
        case task_type_sink_formation:
          runner_do_sink_formation(r, t->ci);
          break;
        case task_type_fof_self:
          runner_do_fof_search_self(r, t->ci, 1);
          break;
        case task_type_fof_pair:
          runner_do_fof_search_pair(r, t->ci, t->cj, 1);
          break;
        case task_type_fof_attach_self:
          runner_do_fof_attach_self(r, t->ci, 1);
          break;
        case task_type_fof_attach_pair:
          runner_do_fof_attach_pair(r, t->ci, t->cj, 1);
          break;
        case task_type_neutrino_weight:
          runner_do_neutrino_weighting(r, ci, 1);
          break;
        case task_type_rt_ghost1:
          runner_do_rt_ghost1(r, t->ci, 1);
          break;
        case task_type_rt_ghost2:
          runner_do_rt_ghost2(r, t->ci, 1);
          break;
        case task_type_rt_tchem:
          runner_do_rt_tchem(r, t->ci, 1);
          break;
        case task_type_rt_advance_cell_time:
          runner_do_rt_advance_cell_time(r, t->ci, 1);
          break;
        default:
          error("Unknown/invalid task type (%d).", t->type);
      }

      /* Check to see if this is the last task in the queue. If so,
       * setlaunch_leftovers to 1 and pack and launch on GPU */
      int self_launch = 0;
      lock_lock(&sched->queues[r->qid].lock);
      if (sched->queues[r->qid].gpu_self_tasks_left < 1) self_launch = 1;
      (void)lock_unlock(&sched->queues[r->qid].lock);

      if (self_launch == 1 && r->gpu.grav_batch_self_count != 0) {
        int ncells_flush_self = r->gpu.grav_batch_self_count;

        {
          TIMER_TIC;

          // now copy all the arrays to the device
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
          if (err4 != hipSuccess)
            printf("Error4: %s\n", hipGetErrorString(err4));

          // run the GPU function
          runner_doself_recursive_grav_new(
              r, ci, 1, r->gpu.gravity_gpu_values_send_self_d,
              r->gpu.gravity_gpu_values_recv_self_d, ncells_flush_self,
              max_cell_size, r->gpu.stream);

          // hipDeviceSynchronize();

          // copy the arrays from device to host
          hipMemcpyAsync(r->gpu.gravity_gpu_values_recv_self,
                         r->gpu.gravity_gpu_values_recv_self_d,
                         ncells_flush_self * max_cell_size *
                             sizeof(struct gravity_gpu_values_recv),
                         hipMemcpyDeviceToHost, r->gpu.stream);

          hipStreamSynchronize(r->gpu.stream);  // THIS ONE IS NEEDED!

          TIMER_TOC(timer_doself_grav_pp);
        }  // TIMER_TOC(timer_gpu_copycalc);
        hipError_t err5 = hipGetLastError();
        if (err5 != hipSuccess) printf("Error5: %s\n", hipGetErrorString(err5));

        {
          TIMER_TIC;

          /*send results back to relevant cell structs*/
          for (int j = 0; j < ncells_flush_self; j++) {
            while (cell_glocktree(r->gpu.grav_cells_self[j])) {
              ; /* spin until we acquire the lock */
            }
            for (int i = 0;
                 i <
                 r->gpu.gravity_gpu_values_send_self[j * max_cell_size].gcounts;
                 i++) {
              r->gpu.grav_cells_self[j]->grav.parts[i].a_grav[0] +=
                  r->gpu.gravity_gpu_values_recv_self[i + j * max_cell_size]
                      .a_x_i;
              r->gpu.grav_cells_self[j]->grav.parts[i].a_grav[1] +=
                  r->gpu.gravity_gpu_values_recv_self[i + j * max_cell_size]
                      .a_y_i;
              r->gpu.grav_cells_self[j]->grav.parts[i].a_grav[2] +=
                  r->gpu.gravity_gpu_values_recv_self[i + j * max_cell_size]
                      .a_z_i;
              r->gpu.grav_cells_self[j]->grav.parts[i].potential +=
                  r->gpu.gravity_gpu_values_recv_self[i + j * max_cell_size]
                      .pot_i;
            }
            cell_gunlocktree(r->gpu.grav_cells_self[j]);
          }

          TIMER_TOC(timer_doself_grav_pp);
        }  // TIMER_TOC(timer_gpu_unpack);

        for (int i = 0; i < ncells_flush_self; i++) {
          scheduler_done(sched, r->gpu.grav_tasks_self[i]);
        }

        // reset counter for next pack
        for (int i = 0; i < ncells_flush_self; i++) {
          r->gpu.grav_cells_self[i] = NULL;
          r->gpu.grav_tasks_self[i] = NULL;
        }
        r->gpu.grav_batch_self_count = 0;
      }

      int pair_launch = 0;
      lock_lock(&sched->queues[r->qid].lock);
      // printf("qid:%i tasks left %i\n", r->qid,
      // sched->queues[r->qid].gpu_pair_tasks_left); fflush(stdout);
      if (sched->queues[r->qid].gpu_pair_tasks_left < 1) pair_launch = 1;
      (void)lock_unlock(&sched->queues[r->qid].lock);

      if (pair_launch == 1 && r->gpu.grav_batch_pair_count != 0) {
        int ncells_flush_pair = r->gpu.grav_batch_pair_count;

        {
          TIMER_TIC;

          // now copy all the arrays to the device
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
          if (err4 != hipSuccess)
            printf("Error4: %s\n", hipGetErrorString(err4));

          // run the GPU function
          struct cell* ci_flush = r->gpu.grav_cells_pair[0];
          struct cell* cj_flush = r->gpu.grav_cells_pair[1];

          if (ci_flush == NULL || cj_flush == NULL)
            error("pair flush: NULL packed cells");

          const struct engine* engine_local = r->e;
          const int periodic = engine_local->mesh->periodic;
          const float dim[3] = {(float)engine_local->mesh->dim[0],
                                (float)engine_local->mesh->dim[1],
                                (float)engine_local->mesh->dim[2]};
          const float r_s_inv = engine_local->mesh->r_s_inv;
          const double min_trunc = engine_local->mesh->r_cut_min;

          float dim_0 = dim[0];
          float dim_1 = dim[1];
          float dim_2 = dim[2];

          TIMER_TIC;

          /* Record activity status */
          const int ci_active =
              cell_is_active_gravity(ci_flush, engine_local) &&
              (ci_flush->nodeID == engine_local->nodeID);
          const int cj_active =
              cell_is_active_gravity(cj_flush, engine_local) &&
              (cj_flush->nodeID == engine_local->nodeID);

          /* Recover the multipole info and shift the CoM locations */
          const float rmax_i = ci_flush->grav.multipole->r_max;
          const float rmax_j = cj_flush->grav.multipole->r_max;

          /* Start by constructing particle caches */

          /* Computed the padded counts */
          const int gcount_i = ci_flush->grav.count;
          const int gcount_j = cj_flush->grav.count;
          const int gcount_padded_i =
              gcount_i - (gcount_i % VEC_SIZE) + VEC_SIZE;
          const int gcount_padded_j =
              gcount_j - (gcount_j % VEC_SIZE) + VEC_SIZE;

          pair_pp_offload_new(periodic, rmax_i, rmax_j, min_trunc, &r_s_inv,
                              &gcount_i, &gcount_padded_i, &gcount_j,
                              &gcount_padded_j, ci_active, cj_active, dim_0,
                              dim_1, dim_2, /*symmetric =*/1,
                              r->gpu.gravity_gpu_values_send_pair_d,
                              r->gpu.gravity_gpu_values_recv_pair_d,
                              ncells_flush_pair, max_cell_size, r->gpu.stream);

          // copy the arrays from device to host
          hipMemcpyAsync(r->gpu.gravity_gpu_values_recv_pair,
                         r->gpu.gravity_gpu_values_recv_pair_d,
                         ncells_flush_pair * max_cell_size *
                             sizeof(struct gravity_gpu_values_recv),
                         hipMemcpyDeviceToHost, r->gpu.stream);

          hipStreamSynchronize(r->gpu.stream);  // THIS ONE IS NEEDED!

          TIMER_TOC(timer_doself_grav_pp);
        }  // TIMER_TOC(timer_gpu_copycalc);
        hipError_t err5 = hipGetLastError();
        if (err5 != hipSuccess) printf("Error5: %s\n", hipGetErrorString(err5));

        {
          TIMER_TIC;

          /*send results back to relevant cell structs*/
          for (int j = 0; j < ncells_flush_pair; j += 2) {
            if (r->gpu.grav_cells_pair[j] == NULL ||
                r->gpu.grav_cells_pair[j + 1] == NULL)
              error("PAIR UNPACK: NULL cell j=%d packed=%d qid=%d", j,
                    ncells_flush_pair, r->qid);

            if (r->gpu.grav_tasks_pair[j / 2] == NULL)
              error("PAIR UNPACK: NULL task k=%d (j=%d) packed=%d qid=%d",
                    j / 2, j, ncells_flush_pair, r->qid);
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
            }  //{printf("hunting for lock for cell %p\n", (void*)a); }
            /*while (cell_glocktree(r->gpu.grav_cells_pair[j])) {
            ; //spin until we acquire the lock
            }*/
            for (int i = 0;
                 i <
                 r->gpu.gravity_gpu_values_send_pair[j * max_cell_size].gcounts;
                 i++) {
              ci0->grav.parts[i].a_grav[0] +=
                  r->gpu.gravity_gpu_values_recv_pair[i + j * max_cell_size]
                      .a_x_i;
              ci0->grav.parts[i].a_grav[1] +=
                  r->gpu.gravity_gpu_values_recv_pair[i + j * max_cell_size]
                      .a_y_i;
              ci0->grav.parts[i].a_grav[2] +=
                  r->gpu.gravity_gpu_values_recv_pair[i + j * max_cell_size]
                      .a_z_i;
              ci0->grav.parts[i].potential +=
                  r->gpu.gravity_gpu_values_recv_pair[i + j * max_cell_size]
                      .pot_i;
            }
            cell_gunlocktree(a);

            while (cell_glocktree(b)) {
              ;
            }  // {printf("hunting for lock for cell %p\n", (void*)b);}
            for (int i = 0;
                 i <
                 r->gpu.gravity_gpu_values_send_pair[(j + 1) * max_cell_size]
                     .gcounts;
                 i++) {
              cj0->grav.parts[i].a_grav[0] +=
                  r->gpu
                      .gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size]
                      .a_x_i;
              cj0->grav.parts[i].a_grav[1] +=
                  r->gpu
                      .gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size]
                      .a_y_i;
              cj0->grav.parts[i].a_grav[2] +=
                  r->gpu
                      .gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size]
                      .a_z_i;
              cj0->grav.parts[i].potential +=
                  r->gpu
                      .gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size]
                      .pot_i;
            }
            cell_gunlocktree(b);

            scheduler_done(sched, r->gpu.grav_tasks_pair[j / 2]);
          }

          TIMER_TOC(timer_doself_grav_pp);
        }  // TIMER_TOC(timer_gpu_unpack);

        // reset counter for next pack
        for (int i = 0; i < ncells_flush_pair; i += 2) {
          r->gpu.grav_cells_pair[i] = NULL;
          r->gpu.grav_cells_pair[i + 1] = NULL;
          r->gpu.grav_tasks_pair[i / 2] = NULL;
        }

        lock_lock(&sched->queues[r->qid].lock);
        r->gpu.grav_batch_pair_count = 0;
        (void)lock_unlock(&sched->queues[r->qid].lock);
      }

      r->active_time += (getticks() - task_beg);

/* Mark that we have run this task on these cells */
#ifdef SWIFT_DEBUG_CHECKS
      if (ci != NULL) {
        ci->tasks_executed[t->type]++;
        ci->subtasks_executed[t->subtype]++;
      }
      if (cj != NULL) {
        cj->tasks_executed[t->type]++;
        cj->subtasks_executed[t->subtype]++;
      }

      /* This runner is not doing a task anymore */
      r->t = NULL;
#endif

      /* We're done with this task, see if we get a next one. */
      prev = t;
      // printf("pack_count= %i \n", pack_count);
      // Here we need an if statement that checks if I am a self gravity task
      // that is not finished packing
      if (t->subtype == task_subtype_grav && t->type == task_type_self) {
        // t->skip = 1;

        /*fprintf(stderr,
        "[DEFER] task=%p type=%d subtype=%d qid=%d pack_self=%d pack_pair=%d
        waiting=%i\n", (void*)t, t->type, t->subtype, r->qid,
        r->gpu.grav_batch_self_count, r->gpu.grav_batch_pair_count,
        sched->waiting);*/

        t->toc = getticks();
        t->total_ticks += t->toc - t->tic;
        t = NULL;

      } else if (t->subtype == task_subtype_grav && t->type == task_type_pair &&
                 packed == 1) {  // pass a bool into here to set if this applies
                                 // to cell - i.e. not just top level cell
        // t->skip = 1;
        /*fprintf(stderr,
        "[DEFER] task=%p type=%d subtype=%d qid=%d pack_self=%d pack_pair=%d
        waiting=%i\n", (void*)t, t->type, t->subtype, r->qid,
        r->gpu.grav_batch_self_count, r->gpu.grav_batch_pair_count,
        sched->waiting);*/

        t->toc = getticks();
        t->total_ticks += t->toc - t->tic;
        t = NULL;
        packed = 0;
        // printf("qid:%i packed:%i \n", r->qid, packed);
        // fflush(stdout);

      } else {
        t = scheduler_done(sched,
                           t);  // copy and replace with gpu, use if statement
      }

    } /* main loop. */
  }
  /* Be kind, rewind. */
  return NULL;
}

ticks runner_get_active_time(const struct runner* restrict r) {
  return r->active_time;
}

void runner_reset_active_time(struct runner* restrict r) { r->active_time = 0; }
