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

  hipSetDevice(0);

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);

  // NEED TO UPDATE THIS SECTION FOR AUTOMATED SPLITTING WITH NEW STRUCT MEMORY
  // SIZES!
  float totGPUmem = (float)prop.totalGlobalMem;  // total memory available
  float availGPUmem =
      0.8 * totGPUmem;  // use up to 80% of the GPU memory to leave some free
  int max_cell_size =
      space_subsize_self_grav +
      100;  // pull max cell size from max interactions set by user
  float allarray =
      4 * max_cell_size *
      24;  // 24 arrays of length max_cell_size containing 4 byte values
  int ncells_tot = availGPUmem / allarray;
  int n_threads = e->nr_threads;
  int ncells_queue = ncells_tot / n_threads;  // dividing by threads
  int ncells = ncells_queue / 50;             // choose to pack e.g. 50 times

  ncells = 4;

  if (ncells == 0) {
    // if too few to pack well then pick arbitrary pack number
    ncells = 8;
  }

  if (r->qid == 0) {
    printf("Total GPU memory: %.2f B\n", (float)prop.totalGlobalMem);
    printf("Max cell size: %i\n", max_cell_size);
    printf("Array size: %f B\n", allarray);
    printf("ncells total: %i \n", ncells_tot);
    printf("nthreads total: %i \n", n_threads);
    printf("ncells per queue: %i \n", ncells_queue);
    printf("ncells per pack: %i \n", ncells);
  }

  // define number of cells to transfer
  // int ncells = 10;

  hipStream_t stream;
  hipStreamCreate(&stream);

  // int max_cell_size = 10000;

  int selfgravs = 0;

  /* Main loop. */
  while (1) {

    /* Wait at the barrier. */
    engine_barrier(e);

    /* Can we go home yet? */
    if (e->step_props & engine_step_prop_done) break;

    /* Re-set the pointer to the previous task, as there is none. */
    struct task* t = NULL;
    struct task* prev = NULL;

    struct gravity_gpu_values_send* gravity_gpu_values_send_self;
    struct gravity_gpu_values_send* gravity_gpu_values_send_self_d;
    hipMalloc((void**)&gravity_gpu_values_send_self_d,
              ncells * max_cell_size * sizeof(struct gravity_gpu_values_send));
    hipMallocHost(
        (void**)&gravity_gpu_values_send_self,
        ncells * max_cell_size * sizeof(struct gravity_gpu_values_send));

    struct gravity_gpu_values_send* gravity_gpu_values_send_pair;
    struct gravity_gpu_values_send* gravity_gpu_values_send_pair_d;
    hipMalloc((void**)&gravity_gpu_values_send_pair_d,
              ncells * max_cell_size * sizeof(struct gravity_gpu_values_send));
    hipMallocHost(
        (void**)&gravity_gpu_values_send_pair,
        ncells * max_cell_size * sizeof(struct gravity_gpu_values_send));

    struct gravity_gpu_values_recv* gravity_gpu_values_recv_self;
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_self_d;
    hipMalloc((void**)&gravity_gpu_values_recv_self_d,
              ncells * max_cell_size * sizeof(struct gravity_gpu_values_recv));
    hipMallocHost(
        (void**)&gravity_gpu_values_recv_self,
        ncells * max_cell_size * sizeof(struct gravity_gpu_values_recv));

    struct gravity_gpu_values_recv* gravity_gpu_values_recv_pair;
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_pair_d;
    hipMalloc((void**)&gravity_gpu_values_recv_pair_d,
              ncells * max_cell_size * sizeof(struct gravity_gpu_values_recv));
    hipMallocHost(
        (void**)&gravity_gpu_values_recv_pair,
        ncells * max_cell_size * sizeof(struct gravity_gpu_values_recv));

    // start counting packing operations
    r->gpu.grav_batch_self_count = 0;  // how many packed in each operation

    // struct cell** grav_cells;
    // grav_cells = malloc(ncells * sizeof(struct cell*));

    struct cell** grav_cells_self;
    grav_cells_self = malloc(ncells * sizeof(struct cell*));

    struct cell** grav_cells_pair;
    grav_cells_pair = malloc(ncells * sizeof(struct cell*));

    struct task** grav_tasks_self;
    grav_tasks_self = malloc(ncells * sizeof(struct task*));

    struct task** grav_tasks_pair;
    grav_tasks_pair = malloc(ncells * sizeof(struct task*));

    int* cell_active;
    cell_active = malloc(ncells * sizeof(int));

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) printf("Error1: %s\n", hipGetErrorString(err));

    int ntasks;

    r->gpu.grav_batch_pair_count = 0;
    int packed = 0;

    /* Loop while there are tasks... */
    while (1) {

      /* If there's no old task, try to get a new one. */
      if (t == NULL) {

        /*printf("hunting for a task\n");*/

        /* Get the task. */
        TIMER_TIC
        t = scheduler_gettask(sched, r->qid, prev);
        TIMER_TOC(timer_gettask);

        /*fprintf(stderr,
        "[TASK GRABBED] task=%p qid=%d waiting=%i\n",
        (void*)t, r->qid, sched->waiting);*/

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

      // int ntasks_g=0;
      // int ndrifts_g=0;
      /*ntasks = sched->queues[r->qid].count; //how many tasks there are to do
      //printf("ntasks %i\n", ntasks);
      printf("qid:%i ntasks: %i  ", r->qid, ntasks);
      for (int i = 0; i < ntasks; i++){
        struct task t1 = sched->queues[r->qid].tasks[i];
        printf(" %d %d;  ", t1.type, t1.subtype);*/
      /*if(t1.subtype == task_subtype_grav && t1.type == task_type_self){
              ntasks_g++;
              }
      if(t1.type == task_type_drift_gpart){
              ndrifts_g++;
              }*/
      //}
      // printf("qid:%i ntasks:%i ntasks_g %i\n", r->qid, ntasks, ntasks_g);

      const ticks task_beg = getticks();
      /* Different types of tasks... */
      switch (t->type) {

        case task_type_self:
          if (t->subtype == task_subtype_grav) {
            // printf("self grav \n");
            selfgravs++;

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

            hipEventRecord(startpack, stream);

            // fill the GPU arrays
            // gravity_gpu_fill_arrays(gravity_gpu_values_h, ci_cache,
            // pack_count, max_cell_size, gcount);
            // gravity_gpu_fill_arrays_send(gravity_gpu_values_send, ci_cache,
            // pack_count, max_cell_size, gcount);
            {
              TIMER_TIC;
              for (int i = 0; i < gcount; i++) {
                gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .h_i = ci_cache->epsilon[i];
                gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .h_j = ci_cache->epsilon[i];
                gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .mass_i = ci_cache->m[i];
                gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .mass_j = ci_cache->m[i];
                gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .x_i = ci_cache->x[i];
                gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .x_j = ci_cache->x[i];
                gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .y_i = ci_cache->y[i];
                gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .y_j = ci_cache->y[i];
                gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .z_i = ci_cache->z[i];
                gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .z_j = ci_cache->z[i];
                gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .active_i = ci_cache->active[i];
                gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .active_j = ci_cache->active[i];
              }

              for (int i = 0; i < max_cell_size; i++) {
                gravity_gpu_values_recv_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .a_x_i = 0;
                gravity_gpu_values_recv_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .a_y_i = 0;
                gravity_gpu_values_recv_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .a_z_i = 0;
                gravity_gpu_values_recv_self[i + r->gpu.grav_batch_self_count *
                                                     max_cell_size]
                    .pot_i = 0;
              }

              TIMER_TOC(timer_doself_grav_pp);
            }  // TIMER_TOC(timer_gpu_pack);

            // store the address of the cells and tasks we are working on
            grav_cells_self[r->gpu.grav_batch_self_count] = ci;
            grav_tasks_self[r->gpu.grav_batch_self_count] = t;

            /*gravity_gpu_values_h->cell_active[pack_count] =
            cell_is_active_gravity(ci, e); if
            (gravity_gpu_values_h->cell_active[pack_count] == 0) printf("active:
            %i\n", gravity_gpu_values_h->cell_active[pack_count]);
            gravity_gpu_values_h->gcounts[pack_count] = gcount;*/

            for (int i = 0; i < gcount; i++) {
              gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
                                                   max_cell_size]
                  .cell_active = cell_is_active_gravity(ci, e);
              gravity_gpu_values_send_self[i + r->gpu.grav_batch_self_count *
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

            // printf("qid: %i pack count: %i packed: %i\n", r->qid, pack_count,
            // packed);

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
              // printf("qid: %i send to GPU \n", r->qid);
              // printf("qid: %i Cells packed. GPU time\n", r->qid);
              int ncells_orig = ncells;

              hipEvent_t startcopyH2D, stopcopyH2D;
              hipEventCreate(&startcopyH2D);
              hipEventCreate(&stopcopyH2D);

              hipEventRecord(startcopyH2D, stream);

              {
                TIMER_TIC;

                // now copy all the arrays to the device
                // gravity_gpu_H2D(gravity_gpu_values_h, gravity_gpu_values_d,
                // ncells, max_cell_size, stream);
                hipMemcpyAsync(gravity_gpu_values_send_self_d,
                               gravity_gpu_values_send_self,
                               ncells * max_cell_size *
                                   sizeof(struct gravity_gpu_values_send),
                               hipMemcpyHostToDevice, stream);
                hipMemcpyAsync(gravity_gpu_values_recv_self_d,
                               gravity_gpu_values_recv_self,
                               ncells * max_cell_size *
                                   sizeof(struct gravity_gpu_values_recv),
                               hipMemcpyHostToDevice, stream);

                hipEventRecord(stopcopyH2D, stream);

                hipError_t err2 = hipGetLastError();
                if (err2 != hipSuccess)
                  printf("Error2: %s\n", hipGetErrorString(err2));

                hipEvent_t startker, stopker;
                hipEventCreate(&startker);
                hipEventCreate(&stopker);

                hipEventRecord(startker, stream);

                // run the GPU function
                // runner_doself_recursive_grav(r, ci, 1,
                // gravity_gpu_values_d->d_h_i, gravity_gpu_values_d->d_h_j,
                // gravity_gpu_values_d->d_mass_i,
                // gravity_gpu_values_d->d_mass_j, gravity_gpu_values_d->d_x_i,
                // gravity_gpu_values_d->d_x_j, gravity_gpu_values_d->d_y_i,
                // gravity_gpu_values_d->d_y_j, gravity_gpu_values_d->d_z_i,
                // gravity_gpu_values_d->d_z_j, gravity_gpu_values_d->d_a_x_i,
                // gravity_gpu_values_d->d_a_y_i, gravity_gpu_values_d->d_a_z_i,
                // gravity_gpu_values_d->d_a_x_j, gravity_gpu_values_d->d_a_y_j,
                // gravity_gpu_values_d->d_a_z_j, gravity_gpu_values_d->d_pot_i,
                // gravity_gpu_values_d->d_pot_j,
                // gravity_gpu_values_d->d_active_i,
                // gravity_gpu_values_d->d_active_j,
                // gravity_gpu_values_d->d_CoM_i, gravity_gpu_values_d->d_CoM_j,
                // ncells, max_cell_size, gravity_gpu_values_d->d_gcounts,
                // gravity_gpu_values_d->d_cell_active, stream);

                runner_doself_recursive_grav_new(r, ci, 1,
                                                 gravity_gpu_values_send_self_d,
                                                 gravity_gpu_values_recv_self_d,
                                                 ncells, max_cell_size, stream);

                hipEventRecord(stopker, stream);

                // hipDeviceSynchronize();

                hipEvent_t startcopyD2H, stopcopyD2H;
                hipEventCreate(&startcopyD2H);
                hipEventCreate(&stopcopyD2H);

                hipEventRecord(startcopyD2H, stream);

                // copy the arrays from device to host
                // gravity_gpu_D2H(gravity_gpu_values_h, gravity_gpu_values_d,
                // ncells, max_cell_size, stream);
                hipMemcpyAsync(gravity_gpu_values_recv_self,
                               gravity_gpu_values_recv_self_d,
                               ncells * max_cell_size *
                                   sizeof(struct gravity_gpu_values_recv),
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
              if (err3 != hipSuccess)
                printf("Error3: %s\n", hipGetErrorString(err3));

              {
                TIMER_TIC;

                /*send results back to relevant cell structs*/
                for (int j = 0; j < ncells; j++) {
                  // printf("[SELF-UNPACK] qid=%d j=%d cell=%p gcount=%d\n",
                  // r->qid, j, (void*)grav_cells_self[j],
                  // gravity_gpu_values_send_self[j*max_cell_size].gcounts);
                  while (cell_glocktree(grav_cells_self[j])) {
                    ; /* spin until we acquire the lock */
                  }
                  for (int i = 0;
                       i <
                       gravity_gpu_values_send_self[j * max_cell_size].gcounts;
                       i++) {
                    grav_cells_self[j]->grav.parts[i].a_grav[0] +=
                        gravity_gpu_values_recv_self[i + j * max_cell_size]
                            .a_x_i;
                    grav_cells_self[j]->grav.parts[i].a_grav[1] +=
                        gravity_gpu_values_recv_self[i + j * max_cell_size]
                            .a_y_i;
                    grav_cells_self[j]->grav.parts[i].a_grav[2] +=
                        gravity_gpu_values_recv_self[i + j * max_cell_size]
                            .a_z_i;
                    grav_cells_self[j]->grav.parts[i].potential +=
                        gravity_gpu_values_recv_self[i + j * max_cell_size]
                            .pot_i;
                    // printf("cell:%i part:%i gcount:%i acceleration: [%f %f
                    // %f]\n", j, i,
                    // gravity_gpu_values_send[j*max_cell_size].gcounts,
                    // grav_cells[j]->grav.parts[i].a_grav[0],
                    // grav_cells[j]->grav.parts[i].a_grav[1],
                    // grav_cells[j]->grav.parts[i].a_grav[2]);
                  }
                  cell_gunlocktree(grav_cells_self[j]);
                }

                TIMER_TOC(timer_doself_grav_pp);
              }  // TIMER_TOC(timer_gpu_unpack);

              // hipDeviceSynchronize();

              for (int i = 0; i < ncells; i++) {
                scheduler_done(sched, grav_tasks_self[i]);
                /*enqueue_dependencies(sched, grav_tasks_self[i]);
                pthread_mutex_lock(&sched->sleep_mutex);
                atomic_dec(&sched->waiting);
                pthread_cond_broadcast(&sched->sleep_cond);
                pthread_mutex_unlock(&sched->sleep_mutex);*/
              }

              // reset counter for next pack
              for (int i = 0; i < ncells; i++) {
                grav_cells_self[i] = NULL;
                grav_tasks_self[i] = NULL;
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

            // printf("PAIR GRAV pack count = %i \n",
            // sched->queues[r->qid].r->gpu.grav_batch_pair_count);
            runner_dopair_recursive_grav_new(
                r, ci, cj, 1, gravity_gpu_values_send_pair,
                gravity_gpu_values_send_pair_d, gravity_gpu_values_recv_pair,
                gravity_gpu_values_recv_pair_d, grav_cells_pair,
                grav_tasks_pair, t, sched, ncells, max_cell_size, &packed,
                stream);

            /*const struct engine *e = r->e;
            const int periodic = e->mesh->periodic;
            const float dim[3] = {(float)e->mesh->dim[0],
            (float)e->mesh->dim[1], (float)e->mesh->dim[2]}; const double
            min_trunc = e->mesh->r_cut_min;

            struct gravity_cache *const ci_cache = &r->ci_gravity_cache;
            struct gravity_cache *const cj_cache = &r->cj_gravity_cache;

            const double shift_i[3] = {0., 0., 0.};
            const double shift_j[3] = {0., 0., 0.};

            const int gcount_i = ci->grav.count;
            const int gcount_j = cj->grav.count;
            const int gcount_padded_i = ((gcount_i + VEC_SIZE - 1) / VEC_SIZE) *
            VEC_SIZE; const int gcount_padded_j = ((gcount_j + VEC_SIZE - 1) /
            VEC_SIZE) * VEC_SIZE;
            //const int gcount_padded_i = gcount_i - (gcount_i % VEC_SIZE) +
            VEC_SIZE; const int gcount_padded_j = gcount_j - (gcount_j %
            VEC_SIZE) + VEC_SIZE; const int allow_mpole = 1; const int
            allow_multipole_i = allow_mpole && ci->grav.count > 1; const int
            allow_multipole_j = allow_mpole && cj->grav.count > 1;

            const float rmax_i = ci->grav.multipole->r_max;
            const float rmax_j = cj->grav.multipole->r_max;

            const float CoM_i[3] = {(float)(ci->grav.multipole->CoM[0] -
            shift_i[0]), (float)(ci->grav.multipole->CoM[1] - shift_i[1]),
                          (float)(ci->grav.multipole->CoM[2] - shift_i[2])};
            const float CoM_j[3] = {(float)(cj->grav.multipole->CoM[0] -
            shift_j[0]), (float)(cj->grav.multipole->CoM[1] - shift_j[1]),
                          (float)(cj->grav.multipole->CoM[2] - shift_j[2])};

            if (gcount_i > max_cell_size)
                error("More particles than allocated memory! %i particles in
            cell and only %i slots in memory available. Increase the number of
            top level cells!", gcount_i, max_cell_size); if (gcount_j >
            max_cell_size) error("More particles than allocated memory! %i
            particles in cell and only %i slots in memory available. Increase
            the number of top level cells!", gcount_j, max_cell_size);*/

            /* Fill the caches */
            /*if (ci->nodeID == e->nodeID) {
            gravity_cache_populate(e->max_active_bin, allow_multipole_j,
            periodic, dim, ci_cache, ci->grav.parts, gcount_i, gcount_padded_i,
                       shift_i, CoM_j, cj->grav.multipole, ci,
                       e->gravity_properties);
            } else {
            gravity_cache_populate_foreign(
                    periodic, dim, ci_cache, ci->grav.parts_foreign, gcount_i,
                    gcount_padded_i, shift_i, ci, e->gravity_properties);
            }

            if (cj->nodeID == e->nodeID) {
            gravity_cache_populate(e->max_active_bin, allow_multipole_i,
            periodic, dim, cj_cache, cj->grav.parts, gcount_j, gcount_padded_j,
                       shift_j, CoM_i, ci->grav.multipole, cj,
                       e->gravity_properties);
            } else {
            gravity_cache_populate_foreign(
                    periodic, dim, cj_cache, cj->grav.parts_foreign, gcount_j,
                    gcount_padded_j, shift_j, cj, e->gravity_properties);
            }*/

            /*struct cell *ci0 = ci;
            struct cell *cj0 = cj;
            struct cell *a = ci0, *b = cj0;

            if (a > b) { struct cell *tmp = a; a = b; b = tmp; }

            while (cell_glocktree(a)) { ; }
            while (cell_glocktree(b)) { ; }*/

            /*hipEvent_t startpack, stoppack;
            hipEventCreate(&startpack);
            hipEventCreate(&stoppack);

            hipEventRecord(startpack, stream);*/

            // fill the GPU arrays
            // gravity_gpu_fill_arrays(gravity_gpu_values_h, ci_cache,
            // pack_count, max_cell_size, gcount);
            // gravity_gpu_fill_arrays_send(gravity_gpu_values_send, ci_cache,
            // pack_count, max_cell_size, gcount);
            /*{TIMER_TIC;
            for (int i = 0; i < gcount_i; i++){
                gravity_gpu_values_send_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].h_i =
            ci_cache->epsilon[i]; gravity_gpu_values_send_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].mass_i = ci_cache->m[i];
                gravity_gpu_values_send_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].x_i = ci_cache->x[i];
                gravity_gpu_values_send_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].y_i = ci_cache->y[i];
                gravity_gpu_values_send_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].z_i = ci_cache->z[i];
                gravity_gpu_values_send_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].active_i =
            ci_cache->active[i];

                gravity_gpu_values_send_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].h_j =
            ci_cache->epsilon[i]; gravity_gpu_values_send_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].mass_j = ci_cache->m[i];
                gravity_gpu_values_send_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].x_j = ci_cache->x[i];
                gravity_gpu_values_send_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].y_j = ci_cache->y[i];
                gravity_gpu_values_send_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].z_j = ci_cache->z[i];
                gravity_gpu_values_send_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].active_j =
            ci_cache->active[i];
            }

            for (int i = 0; i < gcount_j; i++){
                gravity_gpu_values_send_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].h_j =
            cj_cache->epsilon[i]; gravity_gpu_values_send_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].mass_j =
            cj_cache->m[i]; gravity_gpu_values_send_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].x_j =
            cj_cache->x[i]; gravity_gpu_values_send_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].y_j =
            cj_cache->y[i]; gravity_gpu_values_send_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].z_j =
            cj_cache->z[i]; gravity_gpu_values_send_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].active_j =
            cj_cache->active[i];

                gravity_gpu_values_send_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].h_i      =
            cj_cache->epsilon[i]; gravity_gpu_values_send_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].mass_i   =
            cj_cache->m[i]; gravity_gpu_values_send_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].x_i      =
            cj_cache->x[i]; gravity_gpu_values_send_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].y_i      =
            cj_cache->y[i]; gravity_gpu_values_send_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].z_i      =
            cj_cache->z[i]; gravity_gpu_values_send_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].active_i =
            cj_cache->active[i];
            }

            for (int i = 0; i < max_cell_size; i++){
                gravity_gpu_values_recv_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].a_x_i = 0;
                gravity_gpu_values_recv_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].a_y_i = 0;
                gravity_gpu_values_recv_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].a_z_i = 0;
                gravity_gpu_values_recv_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].pot_i = 0;
                gravity_gpu_values_recv_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].a_x_j = 0;
                gravity_gpu_values_recv_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].a_y_j = 0;
                gravity_gpu_values_recv_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].a_z_j = 0;
                gravity_gpu_values_recv_pair[i +
            r->gpu.grav_batch_pair_count*max_cell_size].pot_j = 0;
                }

            for (int i = 0; i < max_cell_size; i++){
                gravity_gpu_values_recv_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].a_x_i = 0;
                gravity_gpu_values_recv_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].a_y_i = 0;
                gravity_gpu_values_recv_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].a_z_i = 0;
                gravity_gpu_values_recv_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].pot_i = 0;
                gravity_gpu_values_recv_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].a_x_j = 0;
                gravity_gpu_values_recv_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].a_y_j = 0;
                gravity_gpu_values_recv_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].a_z_j = 0;
                gravity_gpu_values_recv_pair[i +
            (r->gpu.grav_batch_pair_count+1)*max_cell_size].pot_j = 0;
                }*/

            /*memset(&gravity_gpu_values_recv_pair[r->gpu.grav_batch_pair_count*max_cell_size],
            0, max_cell_size*sizeof(struct gravity_gpu_values_recv));
            memset(&gravity_gpu_values_recv_pair[(r->gpu.grav_batch_pair_count+1)*max_cell_size],
            0,max_cell_size*sizeof(struct gravity_gpu_values_recv));*/

            // TIMER_TOC(timer_dopair_grav_pp);}//TIMER_TOC(timer_gpu_pack);

            // store the address of the cells and tasks we are working on
            /*grav_cells_pair[r->gpu.grav_batch_pair_count] = ci;
            grav_cells_pair[r->gpu.grav_batch_pair_count + 1] = cj;
            grav_tasks_pair[r->gpu.grav_batch_pair_count/2] = t;*/

            /*gravity_gpu_values_h->cell_active[pack_count] =
            cell_is_active_gravity(ci, e); if
            (gravity_gpu_values_h->cell_active[pack_count] == 0) printf("active:
            %i\n", gravity_gpu_values_h->cell_active[pack_count]);
            gravity_gpu_values_h->gcounts[pack_count] = gcount;*/

            /*gravity_gpu_values_send_pair[r->gpu.grav_batch_pair_count*max_cell_size].cell_active
           = cell_is_active_gravity(ci, e);
            gravity_gpu_values_send_pair[(r->gpu.grav_batch_pair_count+1)*max_cell_size].cell_active
           = cell_is_active_gravity(cj, e);

            gravity_gpu_values_send_pair[r->gpu.grav_batch_pair_count*max_cell_size].gcounts
           = gcount_i;
            gravity_gpu_values_send_pair[(r->gpu.grav_batch_pair_count+1)*max_cell_size].gcounts
           = gcount_j;

            int use_full = 1;
            if (periodic) {
                double d0 = CoM_j[0] - CoM_i[0];
                double d1 = CoM_j[1] - CoM_i[1];
                double d2 = CoM_j[2] - CoM_i[2];
                d0 = nearest(d0, e->mesh->dim[0]);
                d1 = nearest(d1, e->mesh->dim[1]);
                d2 = nearest(d2, e->mesh->dim[2]);
                double r2 = d0*d0 + d1*d1 + d2*d2;
                double max_r = sqrt(r2) + rmax_i + rmax_j;
                use_full = (max_r <= min_trunc);
                }

           // store decision on BOTH blocks
           gravity_gpu_values_send_pair[r->gpu.grav_batch_pair_count *
           max_cell_size].use_full = use_full;
           gravity_gpu_values_send_pair[(r->gpu.grav_batch_pair_count + 1) *
           max_cell_size].use_full = use_full;

            //update that we packed a cell into our array
            r->gpu.grav_batch_pair_count += 2;
            packed_pair += 1;

            gravity_cache_zero_output(ci_cache, gcount_padded_i);
            gravity_cache_zero_output(cj_cache, gcount_padded_j);*/

            /*cell_gunlocktree(ci);
            cell_gunlocktree(cj);*/
            /*cell_gunlocktree(b);
            cell_gunlocktree(a);

            lock_lock(&sched->queues[r->qid].lock);
            sched->queues[r->qid].gpu_pair_tasks_left--;
            (void)lock_unlock(&sched->queues[r->qid].lock);*/

            // printf("qid: %i pack count: %i packed: %i\n", r->qid, pack_count,
            // packed);

            /*int acc = 0;
              #ifdef SWIFT_DEBUG_CHECKS*/
            /* Update the interaction counter if it's not a padded gpart */
            /*for (int j = 0; j < gcount; j++){
                    for (int i =0; i < gcount; i++){
                            if (i == j)
                                    continue;
                    //if (!gpart_is_inhibited(&grav_cells[j]->grav.parts[i], e))
                            acc++;
                            accumulate_inc_ll(&ci->grav.parts[j].num_interacted);
                    }
            }
         #endif*/

            // set what happens when the pack count is reached
            /*if (r->gpu.grav_batch_pair_count >= ncells){
                int ncells_orig = ncells;

                hipEvent_t startcopyH2D, stopcopyH2D;
                hipEventCreate(&startcopyH2D);
                hipEventCreate(&stopcopyH2D);

                hipEventRecord(startcopyH2D, stream);

                {TIMER_TIC;

                //now copy all the arrays to the device
                //gravity_gpu_H2D(gravity_gpu_values_h, gravity_gpu_values_d,
               ncells, max_cell_size, stream);
                hipMemcpyAsync(gravity_gpu_values_send_pair_d,
               gravity_gpu_values_send_pair, ncells * max_cell_size *
               sizeof(struct gravity_gpu_values_send), hipMemcpyHostToDevice,
               stream); hipMemcpyAsync(gravity_gpu_values_recv_pair_d,
               gravity_gpu_values_recv_pair, ncells * max_cell_size *
               sizeof(struct gravity_gpu_values_recv), hipMemcpyHostToDevice,
               stream);

                hipEventRecord(stopcopyH2D, stream);

                hipError_t err2 = hipGetLastError();
                if (err2 != hipSuccess)
                        printf("Error2: %s\n", hipGetErrorString(err2));

                hipEvent_t startker, stopker;
                hipEventCreate(&startker);
                hipEventCreate(&stopker);

                hipEventRecord(startker, stream);

                //run the GPU function
                runner_dopair_recursive_grav_new(r, ci, cj, 1,
               gravity_gpu_values_send_pair_d, gravity_gpu_values_recv_pair_d,
               ncells, max_cell_size, stream);

                hipEventRecord(stopker, stream);

                //hipDeviceSynchronize();

                hipEvent_t startcopyD2H, stopcopyD2H;
                hipEventCreate(&startcopyD2H);
                hipEventCreate(&stopcopyD2H);

                hipEventRecord(startcopyD2H, stream);

                //copy the arrays from device to host
                //gravity_gpu_D2H(gravity_gpu_values_h, gravity_gpu_values_d,
               ncells, max_cell_size, stream);
                hipMemcpyAsync(gravity_gpu_values_recv_pair,
               gravity_gpu_values_recv_pair_d, ncells * max_cell_size *
               sizeof(struct gravity_gpu_values_recv), hipMemcpyDeviceToHost,
               stream);

                hipEventRecord(stopcopyD2H, stream);

                hipStreamSynchronize(stream); //THIS ONE IS NEEDED!

                TIMER_TOC(timer_doself_grav_pp);}//TIMER_TOC(timer_gpu_copycalc);*/

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
            /*hipError_t err3 = hipGetLastError();
            if (err3 != hipSuccess)
                    printf("Error3: %s\n", hipGetErrorString(err3));

            {TIMER_TIC;*/

            /*send results back to relevant cell structs*/
            /*for (int j = 0; j < ncells; j+=2) {
                    struct cell *ci0 = grav_cells_pair[j];
                    struct cell *cj0 = grav_cells_pair[j+1];
                    struct cell *a = ci0, *b = cj0;

                    if (a > b) { struct cell *tmp = a; a = b; b = tmp; }

                    while (cell_glocktree(a)) { ; }
                    //printf("hunting for lock for cell %p\n", (void*)a); }
                    for (int i =0; i <
               gravity_gpu_values_send_pair[j*max_cell_size].gcounts; i++){
                            ci0->grav.parts[i].a_grav[0] +=
               gravity_gpu_values_recv_pair[i + j*max_cell_size].a_x_i;
                            ci0->grav.parts[i].a_grav[1] +=
               gravity_gpu_values_recv_pair[i + j*max_cell_size].a_y_i;
                            ci0->grav.parts[i].a_grav[2] +=
               gravity_gpu_values_recv_pair[i + j*max_cell_size].a_z_i;
                            ci0->grav.parts[i].potential +=
               gravity_gpu_values_recv_pair[i + j*max_cell_size].pot_i;*/

            /*if (ci0->grav.parts[i].a_grav[0] == 0){
            printf("cell:%i part:%i gcount:%i acceleration: [%f %f %f]\n", j, i,
            gravity_gpu_values_send_pair[j*max_cell_size].gcounts,
            ci0->grav.parts[i].a_grav[0], ci0->grav.parts[i].a_grav[1],
            ci0->grav.parts[i].a_grav[2]);}*/
            /*}
            cell_gunlocktree(a);

            while (cell_glocktree(b)){;}// {printf("hunting for lock for cell
            %p\n", (void*)b);} for (int i = 0; i <
            gravity_gpu_values_send_pair[(j+1)*max_cell_size].gcounts; i++) {
                    cj0->grav.parts[i].a_grav[0] +=
            gravity_gpu_values_recv_pair[i + (j+1)*max_cell_size].a_x_i;
                    cj0->grav.parts[i].a_grav[1] +=
            gravity_gpu_values_recv_pair[i + (j+1)*max_cell_size].a_y_i;
                    cj0->grav.parts[i].a_grav[2] +=
            gravity_gpu_values_recv_pair[i + (j+1)*max_cell_size].a_z_i;
                    cj0->grav.parts[i].potential +=
            gravity_gpu_values_recv_pair[i + (j+1)*max_cell_size].pot_i;*/

            /*if (cj0->grav.parts[i].a_grav[0] == 0){
            printf("cell:%i part:%i gcount:%i acceleration: [%f %f %f]\n", j, i,
            gravity_gpu_values_send_pair[j*max_cell_size].gcounts,
            cj0->grav.parts[i].a_grav[0], cj0->grav.parts[i].a_grav[1],
            cj0->grav.parts[i].a_grav[2]);}*/
            /*}
            cell_gunlocktree(b);

            scheduler_done(sched, grav_tasks_pair[j/2]);*/
            /*enqueue_dependencies(sched, grav_tasks_pair[j]);
            pthread_mutex_lock(&sched->sleep_mutex);
            atomic_dec(&sched->waiting);
            pthread_cond_broadcast(&sched->sleep_cond);
            pthread_mutex_unlock(&sched->sleep_mutex);*/
            //}

            // TIMER_TOC(timer_doself_grav_pp);}//TIMER_TOC(timer_gpu_unpack);

            // hipDeviceSynchronize();

            /*for(int i=0; i<ncells; i+=2){
                    struct cell *a = grav_cells_pair[i];
                    struct cell *b = grav_cells_pair[i+1];
                    if (a > b) { struct cell *tmp = a; a = b; b = tmp; }
                    cell_gunlocktree(b);
                    cell_gunlocktree(a);
                    //cell_gunlocktree(grav_cells_pair[i]);
                    //cell_gunlocktree(grav_cells_pair[i+1]);
                    enqueue_dependencies(sched, grav_tasks_pair[i]);
                    pthread_mutex_lock(&sched->sleep_mutex);
                    atomic_dec(&sched->waiting);
                    pthread_cond_broadcast(&sched->sleep_cond);
                    pthread_mutex_unlock(&sched->sleep_mutex);
            }*/

            // reset counter for next pack
            /*for (int i = 0; i < ncells; i+=2) {
                    grav_cells_pair[i] = NULL;
                    grav_cells_pair[i+1] = NULL;
                    grav_tasks_pair[i] = NULL;
            }
            r->gpu.grav_batch_pair_count = 0;

            }*/
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
          // printf("end grav \n");
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

      /*ntasks = sched->queues[r->qid].count; //how many tasks there are to do
      int ntasks_self = 0;
      int ntasks_pair = 0;
      for (int i = 0; i < ntasks; i++){
        struct task t1 = sched->queues[r->qid].tasks[i];
      if(t1.subtype == task_subtype_grav && t1.type == task_type_self){
                ntasks_self++;
                }
      if(t1.subtype == task_subtype_grav && t1.type == task_type_pair){
                ntasks_pair++;
                }

      }
      printf("ntasks %i self:%i pair:%i self_pack:%i pair_pack:%i\n", ntasks,
      ntasks_self, ntasks_pair, r->gpu.grav_batch_self_count,
      r->gpu.grav_batch_pair_count);*/

      /*int left = 0, left_grav_self = 0, left_grav_pair = 0;

for (int i = 0; i < sched->queues[r->qid].count; i++) {
  struct task *x = &sched->queues[r->qid].tasks[i];
  if (x->skip) continue;                 // <-- REQUIRED
  if (atomic_load(&x->wait) <= 0) continue;  // optionally count only blocked
ones left++; if (x->subtype == task_subtype_grav && x->type == task_type_self)
left_grav_self++; if (x->subtype == task_subtype_grav && x->type ==
task_type_pair) left_grav_pair++;
}

printf("LEFT %d (grav self %d, grav pair %d) pack_self=%d pack_pair=%d\n",
       left, left_grav_self, left_grav_pair, r->gpu.grav_batch_self_count,
r->gpu.grav_batch_pair_count);*/

      /* Check to see if this is the last task in the queue. If so,
       * setlaunch_leftovers to 1 and pack and launch on GPU */
      int self_launch = 0;
      lock_lock(&sched->queues[r->qid].lock);
      if (sched->queues[r->qid].gpu_self_tasks_left < 1) self_launch = 1;
      (void)lock_unlock(&sched->queues[r->qid].lock);

      if (self_launch == 1 && r->gpu.grav_batch_self_count !=
                                  0) {  //(ntasks_g == 0 && pack_count != 0){
        /*printf("qid:%i flushing self task \n", r->qid);
         fflush(stdout);*/
        // printf("qid:%i flush \n", r->qid);
        // printf("qid: %i Time to flush\n", r->qid);
        int ncells_flush_self = r->gpu.grav_batch_self_count;

        /*if (r->gpu.grav_batch_self_count != ncells){
                ncells = r->gpu.grav_batch_self_count; //updating ncells so that
           if pack_count < ncells at end then we aren't dealing with null data
                }  */

        {
          TIMER_TIC;

          // now copy all the arrays to the device
          hipMemcpyAsync(gravity_gpu_values_send_self_d,
                         gravity_gpu_values_send_self,
                         ncells_flush_self * max_cell_size *
                             sizeof(struct gravity_gpu_values_send),
                         hipMemcpyHostToDevice, stream);
          hipMemcpyAsync(gravity_gpu_values_recv_self_d,
                         gravity_gpu_values_recv_self,
                         ncells_flush_self * max_cell_size *
                             sizeof(struct gravity_gpu_values_recv),
                         hipMemcpyHostToDevice, stream);

          hipError_t err4 = hipGetLastError();
          if (err4 != hipSuccess)
            printf("Error4: %s\n", hipGetErrorString(err4));

          // run the GPU function
          runner_doself_recursive_grav_new(
              r, ci, 1, gravity_gpu_values_send_self_d,
              gravity_gpu_values_recv_self_d, ncells_flush_self, max_cell_size,
              stream);

          // hipDeviceSynchronize();

          // copy the arrays from device to host
          hipMemcpyAsync(gravity_gpu_values_recv_self,
                         gravity_gpu_values_recv_self_d,
                         ncells_flush_self * max_cell_size *
                             sizeof(struct gravity_gpu_values_recv),
                         hipMemcpyDeviceToHost, stream);

          hipStreamSynchronize(stream);  // THIS ONE IS NEEDED!

          TIMER_TOC(timer_doself_grav_pp);
        }  // TIMER_TOC(timer_gpu_copycalc);
        hipError_t err5 = hipGetLastError();
        if (err5 != hipSuccess) printf("Error5: %s\n", hipGetErrorString(err5));

        {
          TIMER_TIC;

          /*send results back to relevant cell structs*/
          for (int j = 0; j < ncells_flush_self; j++) {
            while (cell_glocktree(grav_cells_self[j])) {
              ; /* spin until we acquire the lock */
            }
            for (int i = 0;
                 i < gravity_gpu_values_send_self[j * max_cell_size].gcounts;
                 i++) {
              grav_cells_self[j]->grav.parts[i].a_grav[0] +=
                  gravity_gpu_values_recv_self[i + j * max_cell_size].a_x_i;
              grav_cells_self[j]->grav.parts[i].a_grav[1] +=
                  gravity_gpu_values_recv_self[i + j * max_cell_size].a_y_i;
              grav_cells_self[j]->grav.parts[i].a_grav[2] +=
                  gravity_gpu_values_recv_self[i + j * max_cell_size].a_z_i;
              grav_cells_self[j]->grav.parts[i].potential +=
                  gravity_gpu_values_recv_self[i + j * max_cell_size].pot_i;
              // printf("acceleration: [%f %f %f]\n",
              // grav_cells[j]->grav.parts[i].a_grav[0],
              // grav_cells[j]->grav.parts[i].a_grav[1],
              // grav_cells[j]->grav.parts[i].a_grav[2]);
            }
            cell_gunlocktree(grav_cells_self[j]);
          }

          TIMER_TOC(timer_doself_grav_pp);
        }  // TIMER_TOC(timer_gpu_unpack);

        for (int i = 0; i < ncells_flush_self; i++) {
          scheduler_done(sched, grav_tasks_self[i]);

          fprintf(stderr, "[FLUSH-DONE-SELF] task=%p qid=%d i=%d waiting=%i\n",
                  (void*)grav_tasks_self[i], r->qid, i, sched->waiting);
          // if (grav_cells[i] != NULL) { //skip if grav_cells[i] not filled in
          // final batch cell_gunlocktree(grav_cells_self[i]);//}
          // if (grav_tasks[i] != NULL){
          /*enqueue_dependencies(sched, grav_tasks_self[i]); //Line 3296 in Abou
          repo pthread_mutex_lock(&sched->sleep_mutex);
          atomic_dec(&sched->waiting);
          pthread_cond_broadcast(&sched->sleep_cond);
          pthread_mutex_unlock(&sched->sleep_mutex);*/
          //}
        }

        // reset counter for next pack
        for (int i = 0; i < ncells_flush_self; i++) {
          grav_cells_self[i] = NULL;
          grav_tasks_self[i] = NULL;
        }
        r->gpu.grav_batch_self_count = 0;
        // pack_done = 1;
        /*ncells = ncells_orig;*/
      }

      int pair_launch = 0;
      lock_lock(&sched->queues[r->qid].lock);
      // printf("qid:%i tasks left %i\n", r->qid,
      // sched->queues[r->qid].gpu_pair_tasks_left); fflush(stdout);
      if (sched->queues[r->qid].gpu_pair_tasks_left < 1) pair_launch = 1;
      (void)lock_unlock(&sched->queues[r->qid].lock);

      if (pair_launch == 1 && r->gpu.grav_batch_pair_count !=
                                  0) {  //(ntasks_g == 0 && pack_count != 0){
        // printf("qid:%i flushing pair task \n", r->qid);
        // fflush(stdout);
        // printf("qid:%i flush \n", r->qid);
        // printf("qid: %i Time to flush\n", r->qid);
        // printf("FLUSH ENTER qid=%d pack_count=%d tid=%ld\n", r->qid,
        // r->gpu.grav_batch_pair_count, pthread_self());
        /*int ncells_orig = ncells;

        if (r->gpu.grav_batch_pair_count != ncells){
                ncells = r->gpu.grav_batch_pair_count; //updating ncells so that
        if pack_count < ncells at end then we aren't dealing with null data
                }   */
        int ncells_flush_pair = r->gpu.grav_batch_pair_count;

        {
          TIMER_TIC;

          // now copy all the arrays to the device
          hipMemcpyAsync(gravity_gpu_values_send_pair_d,
                         gravity_gpu_values_send_pair,
                         ncells_flush_pair * max_cell_size *
                             sizeof(struct gravity_gpu_values_send),
                         hipMemcpyHostToDevice, stream);
          hipMemcpyAsync(gravity_gpu_values_recv_pair_d,
                         gravity_gpu_values_recv_pair,
                         ncells_flush_pair * max_cell_size *
                             sizeof(struct gravity_gpu_values_recv),
                         hipMemcpyHostToDevice, stream);

          hipError_t err4 = hipGetLastError();
          if (err4 != hipSuccess)
            printf("Error4: %s\n", hipGetErrorString(err4));

          // run the GPU function
          struct cell* ci_flush = grav_cells_pair[0];
          struct cell* cj_flush = grav_cells_pair[1];

          if (ci_flush == NULL || cj_flush == NULL)
            error("pair flush: NULL packed cells");

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
          const int ci_active = cell_is_active_gravity(ci_flush, e) &&
                                (ci_flush->nodeID == e->nodeID);
          const int cj_active = cell_is_active_gravity(cj_flush, e) &&
                                (cj_flush->nodeID == e->nodeID);

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

          // runner_dopair_recursive_grav_new(r, ci_flush, cj_flush, 1,
          // gravity_gpu_values_send_pair, gravity_gpu_values_send_pair_d,
          // gravity_gpu_values_recv_pair, gravity_gpu_values_recv_pair_d,
          // grav_cells_pair, grav_tasks_pair, t, sched, ncells_flush_pair,
          // max_cell_size, &r->gpu.grav_batch_pair_count, stream);

          pair_pp_offload_new(
              periodic, rmax_i, rmax_j, min_trunc, &r_s_inv, &gcount_i,
              &gcount_padded_i, &gcount_j, &gcount_padded_j, ci_active,
              cj_active, dim_0, dim_1, dim_2, /*symmetric =*/1,
              gravity_gpu_values_send_pair_d, gravity_gpu_values_recv_pair_d,
              ncells_flush_pair, max_cell_size, stream);

          // runner_dopair_recursive_grav_new(r, ci, cj, 1,
          // gravity_gpu_values_send_pair_d, gravity_gpu_values_recv_pair_d,
          // ncells_flush_pair, max_cell_size, stream);

          // hipDeviceSynchronize();

          // copy the arrays from device to host
          hipMemcpyAsync(gravity_gpu_values_recv_pair,
                         gravity_gpu_values_recv_pair_d,
                         ncells_flush_pair * max_cell_size *
                             sizeof(struct gravity_gpu_values_recv),
                         hipMemcpyDeviceToHost, stream);

          hipStreamSynchronize(stream);  // THIS ONE IS NEEDED!

          TIMER_TOC(timer_doself_grav_pp);
        }  // TIMER_TOC(timer_gpu_copycalc);
        hipError_t err5 = hipGetLastError();
        if (err5 != hipSuccess) printf("Error5: %s\n", hipGetErrorString(err5));

        {
          TIMER_TIC;

          /*send results back to relevant cell structs*/
          for (int j = 0; j < ncells_flush_pair; j += 2) {
            if (grav_cells_pair[j] == NULL || grav_cells_pair[j + 1] == NULL)
              error("PAIR UNPACK: NULL cell j=%d packed=%d qid=%d", j,
                    ncells_flush_pair, r->qid);

            if (grav_tasks_pair[j / 2] == NULL)
              error("PAIR UNPACK: NULL task k=%d (j=%d) packed=%d qid=%d",
                    j / 2, j, ncells_flush_pair, r->qid);
            // printf("[PAIR-UNPACK FLUSH] qid=%d j=%d cell_i=%p cell_j=%p
            // gcount_i=%d gcount_j=%d\n", r->qid, j, (void*)grav_cells_pair[j],
            // (void*)grav_cells_pair[j+1],
            // gravity_gpu_values_send_pair[j*max_cell_size].gcounts,
            // gravity_gpu_values_send_pair[(j+1)*max_cell_size].gcounts);
            struct cell* ci0 = grav_cells_pair[j];
            struct cell* cj0 = grav_cells_pair[j + 1];
            struct cell *a = ci0, *b = cj0;

            if (a > b) {
              struct cell* tmp = a;
              a = b;
              b = tmp;
            }

            while (cell_glocktree(a)) {
              ;
            }  //{printf("hunting for lock for cell %p\n", (void*)a); }
            /*while (cell_glocktree(grav_cells_pair[j])) {
            ; //spin until we acquire the lock
            }*/
            for (int i = 0;
                 i < gravity_gpu_values_send_pair[j * max_cell_size].gcounts;
                 i++) {
              ci0->grav.parts[i].a_grav[0] +=
                  gravity_gpu_values_recv_pair[i + j * max_cell_size].a_x_i;
              ci0->grav.parts[i].a_grav[1] +=
                  gravity_gpu_values_recv_pair[i + j * max_cell_size].a_y_i;
              ci0->grav.parts[i].a_grav[2] +=
                  gravity_gpu_values_recv_pair[i + j * max_cell_size].a_z_i;
              ci0->grav.parts[i].potential +=
                  gravity_gpu_values_recv_pair[i + j * max_cell_size].pot_i;

              /*if (ci0->grav.parts[i].a_grav[0] == 0){
              printf("cell:%i part:%i gcount:%i acceleration: [%f %f %f]\n", j,
              i, gravity_gpu_values_send_pair[j*max_cell_size].gcounts,
              ci0->grav.parts[i].a_grav[0], ci0->grav.parts[i].a_grav[1],
              ci0->grav.parts[i].a_grav[2]);}*/
            }
            cell_gunlocktree(a);

            while (cell_glocktree(b)) {
              ;
            }  // {printf("hunting for lock for cell %p\n", (void*)b);}
            for (int i = 0;
                 i <
                 gravity_gpu_values_send_pair[(j + 1) * max_cell_size].gcounts;
                 i++) {
              cj0->grav.parts[i].a_grav[0] +=
                  gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size]
                      .a_x_i;
              cj0->grav.parts[i].a_grav[1] +=
                  gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size]
                      .a_y_i;
              cj0->grav.parts[i].a_grav[2] +=
                  gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size]
                      .a_z_i;
              cj0->grav.parts[i].potential +=
                  gravity_gpu_values_recv_pair[i + (j + 1) * max_cell_size]
                      .pot_i;

              /*if (cj0->grav.parts[i].a_grav[0] == 0){
              printf("cell:%i part:%i gcount:%i acceleration: [%f %f %f]\n", j,
              i, gravity_gpu_values_send_pair[j*max_cell_size].gcounts,
              cj0->grav.parts[i].a_grav[0], cj0->grav.parts[i].a_grav[1],
              cj0->grav.parts[i].a_grav[2]);}*/
            }
            cell_gunlocktree(b);

            scheduler_done(sched, grav_tasks_pair[j / 2]);

            fprintf(stderr,
                    "[FLUSH-DONE-PAIR] task=%p qid=%d j=%d waiting=%i\n",
                    (void*)grav_tasks_pair[j / 2], r->qid, j, sched->waiting);

            /*enqueue_dependencies(sched, grav_tasks_pair[j]);
            pthread_mutex_lock(&sched->sleep_mutex);
            atomic_dec(&sched->waiting);
            pthread_cond_broadcast(&sched->sleep_cond);
            pthread_mutex_unlock(&sched->sleep_mutex);*/
          }

          TIMER_TOC(timer_doself_grav_pp);
        }  // TIMER_TOC(timer_gpu_unpack);

        /*for(int i=0; i<ncells; i+=2){
                struct cell *a = grav_cells_pair[i];
                struct cell *b = grav_cells_pair[i+1];
                if (a > b) { struct cell *tmp = a; a = b; b = tmp; }
                cell_gunlocktree(b);
                cell_gunlocktree(a);
                //cell_gunlocktree(grav_cells_pair[i]);
                //cell_gunlocktree(grav_cells_pair[i+1]);
                enqueue_dependencies(sched, grav_tasks_pair[i]);
                pthread_mutex_lock(&sched->sleep_mutex);
                atomic_dec(&sched->waiting);
                pthread_cond_broadcast(&sched->sleep_cond);
                pthread_mutex_unlock(&sched->sleep_mutex);
        }*/

        // reset counter for next pack
        for (int i = 0; i < ncells_flush_pair; i += 2) {
          grav_cells_pair[i] = NULL;
          grav_cells_pair[i + 1] = NULL;
          grav_tasks_pair[i / 2] = NULL;
        }

        lock_lock(&sched->queues[r->qid].lock);
        r->gpu.grav_batch_pair_count = 0;
        (void)lock_unlock(&sched->queues[r->qid].lock);
        // pack_done = 1;
        // ncells = ncells_orig;

        // printf("FLUSH EXIT  qid=%d pack_count=%d tid=%ld\n", r->qid,
        // r->gpu.grav_batch_pair_count, pthread_self());
      }

      r->active_time += (getticks() - task_beg);

      // printf("qid:%i packed:%i \n", r->qid, packed);
      // fflush(stdout);

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

    /*hipFreeHost(gravity_gpu_values_h->h_i);
    hipFreeHost(gravity_gpu_values_h->h_j);
    hipFreeHost(gravity_gpu_values_h->mass_i);
    hipFreeHost(gravity_gpu_values_h->mass_j);
    hipFreeHost(gravity_gpu_values_h->x_i);
    hipFreeHost(gravity_gpu_values_h->x_j);
    hipFreeHost(gravity_gpu_values_h->y_i);
    hipFreeHost(gravity_gpu_values_h->y_j);
    hipFreeHost(gravity_gpu_values_h->z_i);
    hipFreeHost(gravity_gpu_values_h->z_j);
    hipFreeHost(gravity_gpu_values_h->a_x_i);
    hipFreeHost(gravity_gpu_values_h->a_y_i);
    hipFreeHost(gravity_gpu_values_h->a_z_i);
    hipFreeHost(gravity_gpu_values_h->a_x_j);
    hipFreeHost(gravity_gpu_values_h->a_y_j);
    hipFreeHost(gravity_gpu_values_h->a_z_j);
    hipFreeHost(gravity_gpu_values_h->pot_i);
    hipFreeHost(gravity_gpu_values_h->pot_j);
    hipFreeHost(gravity_gpu_values_h->active_i);
    hipFreeHost(gravity_gpu_values_h->active_j);
    hipFreeHost(gravity_gpu_values_h->CoM_i);
    hipFreeHost(gravity_gpu_values_h->CoM_j);
    hipFreeHost(gravity_gpu_values_h->gcounts);
    free(gravity_gpu_values_h);

    hipFree(gravity_gpu_values_d->d_h_i);
    hipFree(gravity_gpu_values_d->d_h_j);
    hipFree(gravity_gpu_values_d->d_mass_i);
    hipFree(gravity_gpu_values_d->d_mass_j);
    hipFree(gravity_gpu_values_d->d_x_i);
    hipFree(gravity_gpu_values_d->d_x_j);
    hipFree(gravity_gpu_values_d->d_y_i);
    hipFree(gravity_gpu_values_d->d_y_j);
    hipFree(gravity_gpu_values_d->d_z_i);
    hipFree(gravity_gpu_values_d->d_z_j);
    hipFree(gravity_gpu_values_d->d_a_x_i);
    hipFree(gravity_gpu_values_d->d_a_y_i);
    hipFree(gravity_gpu_values_d->d_a_z_i);
    hipFree(gravity_gpu_values_d->d_a_x_j);
    hipFree(gravity_gpu_values_d->d_a_y_j);
    hipFree(gravity_gpu_values_d->d_a_z_j);
    hipFree(gravity_gpu_values_d->d_pot_i);
    hipFree(gravity_gpu_values_d->d_pot_j);
    hipFree(gravity_gpu_values_d->d_active_i);
    hipFree(gravity_gpu_values_d->d_active_j);
    hipFree(gravity_gpu_values_d->d_CoM_i);
    hipFree(gravity_gpu_values_d->d_CoM_j);
    hipFree(gravity_gpu_values_d->d_gcounts);
    free(gravity_gpu_values_d);*/

    hipFreeHost(gravity_gpu_values_send_self);
    hipFreeHost(gravity_gpu_values_recv_self);
    hipFree(gravity_gpu_values_send_self_d);
    hipFree(gravity_gpu_values_recv_self_d);

    hipFreeHost(gravity_gpu_values_send_pair);
    hipFreeHost(gravity_gpu_values_recv_pair);
    hipFree(gravity_gpu_values_send_pair_d);
    hipFree(gravity_gpu_values_recv_pair_d);

    free(grav_cells_self);
    free(grav_tasks_self);
    free(grav_cells_pair);
    free(grav_tasks_pair);
    free(cell_active);

    // printf("qid: %i selfgravs %i\n", r->qid, selfgravs);
  }
  hipStreamDestroy(stream);
  /* Be kind, rewind. */
  return NULL;
}

ticks runner_get_active_time(const struct runner* restrict r) {
  return r->active_time;
}

void runner_reset_active_time(struct runner* restrict r) { r->active_time = 0; }
