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

/**
 * @brief GPU-specific state owned by a single runner.
 */
struct gpu_runner {

  /*! Number of self cells currently packed in this runner's GPU batch. */
  int grav_batch_self_count;

  /*! Number of pair cells currently packed in this runner's GPU batch. */
  int grav_batch_pair_count;
};

void runner_gpu_init(struct gpu_runner* gpu);
void runner_gpu_clean(struct gpu_runner* gpu);

#endif /* SWIFT_RUNNER_GPU_H */
