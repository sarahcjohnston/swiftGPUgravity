/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2026 Sarah Johnston (sarah.c.johnston@durham.ac.uk)
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

#ifndef SWIFT_GPU_PIPELINE_H
#define SWIFT_GPU_PIPELINE_H

#include <config.h>

#if defined(WITH_CUDA)

#include <cuda_pipeline.h>

#define GPU_PIPELINE_COPY(dst, src, bytes) \
    __pipeline_memcpy_async(dst, src, bytes)

#define GPU_PIPELINE_COMMIT() \
    __pipeline_commit()

#define GPU_PIPELINE_WAIT() \
    __pipeline_wait_prior(0)

#elif defined(WITH_HIP)

/* Synchronous shared-memory copy for AMD */
#define GPU_PIPELINE_COPY(dst, src, bytes)  \
    do {                                   \
        static_assert((bytes) == sizeof(*(dst)), \
                      "GPU pipeline copy size mismatch"); \
        *(dst) = *(src);                    \
    } while (0)

#define GPU_PIPELINE_COMMIT() ((void)0)
#define GPU_PIPELINE_WAIT()   ((void)0)

#else
#error "No GPU backend selected"
#endif

#endif
