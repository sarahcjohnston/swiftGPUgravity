/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2026 Will Roper (w.roper@sussex.ac.uk)
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
#ifndef SWIFT_RUNNER_GPU_H
#define SWIFT_RUNNER_GPU_H

#include <hip/hip_runtime_api.h>

struct cell;
struct runner;
struct scheduler;
struct task;
struct gravity_gpu_values_recv;
struct gravity_gpu_values_send;

/**
 * @brief GPU-specific state owned by a single runner.
 */
struct gpu_runner {

  /*! Number of self cells currently packed in this runner's GPU batch. */
  int grav_batch_self_count;

  /*! Number of pair cells currently packed in this runner's GPU batch. */
  int grav_batch_pair_count;

  /*! Number of cells that fit in one GPU batch for this runner. */
  int grav_batch_ncells;

  /*! Maximum number of particles packed per cell. */
  int grav_max_cell_size;

  /*! Host and device buffers for packed self interactions. */
  struct gravity_gpu_values_send *gravity_gpu_values_send_self,
      *gravity_gpu_values_send_self_d;

  /*! Host and device buffers for packed pair interactions. */
  struct gravity_gpu_values_send *gravity_gpu_values_send_pair,
      *gravity_gpu_values_send_pair_d;

  /*! Host and device buffers for unpacked self results. */
  struct gravity_gpu_values_recv *gravity_gpu_values_recv_self,
      *gravity_gpu_values_recv_self_d;

  /*! Host and device buffers for unpacked pair results. */
  struct gravity_gpu_values_recv *gravity_gpu_values_recv_pair,
      *gravity_gpu_values_recv_pair_d;

  /*! Packed self-batch cell and task handles. */
  struct cell** grav_cells_self;
  struct task** grav_tasks_self;

  /*! Packed pair-batch cell and task handles. */
  struct cell** grav_cells_pair;
  struct task** grav_tasks_pair;

  /*! Per-cell activity flags used by the GPU path. */
  int* cell_active;

  /*! Stream used for this runner's GPU work. */
  hipStream_t stream;
};

/**
 * @brief Initialise the GPU-specific state attached to a runner.
 *
 * @param r The runner whose GPU state to initialise.
 */
void runner_gpu_init(struct runner* r);

/**
 * @brief Clean the GPU-specific state attached to a runner.
 *
 * @param r The runner whose GPU state to clean.
 */
void runner_gpu_clean(struct runner* r);

/**
 * @brief Pack, launch, and unpack a batched self-gravity GPU task.
 *
 * @param r The #runner.
 * @param c The #cell to pack.
 * @param t The #task being executed.
 * @param sched The #scheduler owning the task.
 * @param ncells The batch capacity in cells.
 * @param max_cell_size The maximum number of particles per packed cell.
 */
void runner_doself_grav_pp_task_new(struct runner* r, struct cell* c,
                                    struct task* t, struct scheduler* sched,
                                    int ncells, int max_cell_size);

/**
 * @brief Pack one leaf pair-gravity interaction into the runner GPU batch.
 */
void runner_dopair_grav_pp_new(
    struct runner* r, struct cell* ci, struct cell* cj, const int symmetric,
    const int allow_mpole,
    struct gravity_gpu_values_send* gravity_gpu_values_send,
    struct gravity_gpu_values_send* gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_d,
    struct cell** grav_cells_pair, struct task** grav_tasks_pair,
    struct task* t, struct scheduler* sched, int ncells, int max_cell_size,
    hipStream_t stream);

/**
 * @brief Recursively process a pair-gravity task with GPU batching.
 */
void runner_dopair_recursive_grav_new(
    struct runner* r, struct cell* ci, struct cell* cj, const int gettimer,
    struct gravity_gpu_values_send* gravity_gpu_values_send,
    struct gravity_gpu_values_send* gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_d,
    struct cell** grav_cells_pair, struct task** grav_tasks_pair,
    struct task* t, struct scheduler* sched, int ncells, int max_cell_size,
    int* packed, hipStream_t stream);

/**
 * @brief Flush any leftover packed self-gravity work owned by a runner.
 *
 * @param r The runner whose GPU batch should be flushed.
 * @param sched The scheduler owning the queued tasks.
 */
void runner_gpu_flush_leftover_self(struct runner* r, struct scheduler* sched);

/**
 * @brief Flush any leftover packed pair-gravity work owned by a runner.
 *
 * @param r The runner whose GPU batch should be flushed.
 * @param sched The scheduler owning the queued tasks.
 */
void runner_gpu_flush_leftover_pair(struct runner* r, struct scheduler* sched);

#endif /* SWIFT_RUNNER_GPU_H */
