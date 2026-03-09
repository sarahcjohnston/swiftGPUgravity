/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2013 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 *               2016 Matthieu Schaller (schaller@strw.leidenuniv.nl)
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
#ifndef SWIFT_RUNNER_DOIACT_GRAV_H
#define SWIFT_RUNNER_DOIACT_GRAV_H

#include <config.h>

#ifdef __cplusplus
extern "C" {
#endif
/* GPU headers */
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
}
#endif

struct runner;
struct cell;
struct task;
struct scheduler;
struct gravity_gpu_values_send;
struct gravity_gpu_values_recv;

void runner_do_grav_down(struct runner *r, struct cell *c, int timer);

void runner_dopair_grav_pp(struct runner *r, struct cell *ci, struct cell *cj,
                           const int symmetric, const int allow_mpole);

void runner_doself_recursive_grav(struct runner *r, struct cell *c,
                                  int gettimer, float *d_h_i, float *d_h_j, float *d_mass_i, float *d_mass_j, float *d_x_i, float *d_x_j, float *d_y_i, float *d_y_j, float *d_z_i, float *d_z_j, float *d_a_x_i, float *d_a_y_i, float *d_a_z_i, float *d_a_x_j, float *d_a_y_j, float *d_a_z_j, float *d_pot_i, float *d_pot_j, int *d_active_i, int *d_active_j, float *d_CoM_i, float *d_CoM_j, int ncells, int max_cell_size, int *d_gcounts, int *d_cell_active, cudaStream_t stream);
                                 
void runner_doself_recursive_grav_new(struct runner *r, struct cell *c, const int gettimer, struct gravity_gpu_values_send *gravity_gpu_values_send_d, struct gravity_gpu_values_recv *gravity_gpu_values_recv_d, int ncells, int max_cell_size, cudaStream_t stream);


void runner_dopair_recursive_grav(struct runner *r, struct cell *ci,
                                  struct cell *cj, int gettimer);

void runner_dopair_grav_mm_progenies(struct runner *r, const long long flags,
                                     struct cell *restrict ci,
                                     struct cell *restrict cj);

void runner_do_grav_long_range(struct runner *r, struct cell *ci, int timer);

/* Internal functions (for unit tests and debugging) */

void runner_doself_grav_pp(struct runner *r, struct cell *c, float *d_h_i, float *d_mass_i, float *d_x_i, float *d_y_i, float *d_z_i, float *d_a_x_i, float *d_a_y_i, float *d_a_z_i, float *d_pot_i,int *d_active_i, int ncells, int max_cell_size, int *gcounts, int *cell_active, cudaStream_t stream);

void runner_doself_grav_pp_new(struct runner *r, struct cell *c, struct gravity_gpu_values_send *gravity_gpu_values_send_d, struct gravity_gpu_values_recv *gravity_gpu_values_recv_d, int ncells, int max_cell_size, cudaStream_t stream);


void runner_dopair_grav_pp(struct runner *r, struct cell *ci, struct cell *cj,
                           const int symmetric, const int allow_mpole);
                           
void runner_dopair_grav_pp_new(struct runner *r, struct cell *ci, struct cell *cj,
                           const int symmetric, const int allow_mpole, struct gravity_gpu_values_send *gravity_gpu_values_send, struct gravity_gpu_values_send *gravity_gpu_values_send_d, struct gravity_gpu_values_recv *gravity_gpu_values_recv, struct gravity_gpu_values_recv *gravity_gpu_values_recv_d, struct cell** grav_cells_pair, struct task** grav_tasks_pair, struct task *t, struct scheduler *sched, int ncells, int max_cell_size, int *pack_count_pair, cudaStream_t stream);
                           
void runner_dopair_recursive_grav_new(struct runner *r, struct cell *ci,
                                  struct cell *cj, const int gettimer, struct gravity_gpu_values_send *gravity_gpu_values_send, struct gravity_gpu_values_send *gravity_gpu_values_send_d, struct gravity_gpu_values_recv *gravity_gpu_values_recv, struct gravity_gpu_values_recv *gravity_gpu_values_recv_d, struct cell** grav_cells_pair, struct task** grav_tasks_pair, struct task *t, struct scheduler *sched, int ncells, int max_cell_size, int *pack_count_pair, int *packed, cudaStream_t stream);

#endif /* SWIFT_RUNNER_DOIACT_GRAV_H */
