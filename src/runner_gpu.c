#include "runner_gpu.h"

/**
 * @brief Initialise the GPU-specific state attached to a runner.
 *
 * @param gpu The GPU state to initialise.
 */
void runner_gpu_init(struct gpu_runner* gpu) {
  gpu->grav_batch_self_count = 0;
  gpu->grav_batch_pair_count = 0;
}

/**
 * @brief Clean the GPU-specific state attached to a runner.
 *
 * @param gpu The GPU state to clean.
 */
void runner_gpu_clean(struct gpu_runner* gpu) {
  gpu->grav_batch_self_count = 0;
  gpu->grav_batch_pair_count = 0;
}
