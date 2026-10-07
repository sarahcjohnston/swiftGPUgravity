#ifndef SWIFT_GPU_RANK_H
#define SWIFT_GPU_RANK_H

/**
 * @brief Bind this MPI rank/process to one GPU.
 *
 * Must be called after MPI rank discovery and before any GPU allocation,
 * stream creation, host pinned allocation, or kernel launch.
 *
 * @param global_rank MPI rank, or 0 in non-MPI mode.
 * @return The selected GPU device id.
 */
int gpu_rank_bind(int global_rank);

#endif
