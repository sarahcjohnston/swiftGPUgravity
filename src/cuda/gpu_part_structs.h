/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2025 Abouzied M. A. Nasar (abouzied.nasar@manchester.ac.uk)
 *                    Mladen Ivkovic (mladen.ivkovic@durham.ac.uk)
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
#ifndef CUDA_GPU_PART_STRUCTS_H
#define CUDA_GPU_PART_STRUCTS_H

/* Configuration */
#include "../../config.h"
#include "../align.h"
#include "../timeline.h"

#define GPU_HYDRO_ALIGN16 __attribute__((aligned(16)))

#if defined(WITH_CUDA)

/* CUDA vector types work in both C and C++ */
#include <vector_types.h>

#define HYDRO_DOUBLE2 double2
#define HYDRO_DOUBLE4 double4
#define HYDRO_FLOAT4  float4
#define HYDRO_FLOAT3  float3
#define HYDRO_FLOAT2  float2
#define HYDRO_INT2    int2

#elif defined(WITH_HIP)

#ifdef __cplusplus

/* HIP device compilation */
#include <hip/hip_runtime.h>

#define HYDRO_DOUBLE2 double2
#define HYDRO_DOUBLE4 double4
#define HYDRO_FLOAT4  float4
#define HYDRO_FLOAT3  float3
#define HYDRO_FLOAT2  float2
#define HYDRO_INT2    int2

#else

/* HIP host C compilation */
/* Keep your existing gpu_hydro_* typedefs here */
/* HIP host C vector representations */

typedef struct __attribute__((aligned(16))) {
    double x, y;
} gpu_hydro_double2;

typedef struct __attribute__((aligned(32))) {
    double x, y, z, w;
} gpu_hydro_double4;

typedef struct __attribute__((aligned(16))) {
    float x, y, z, w;
} gpu_hydro_float4;

typedef struct {
    float x, y, z;
} gpu_hydro_float3;

typedef struct __attribute__((aligned(8))) {
    float x, y;
} gpu_hydro_float2;

typedef struct __attribute__((aligned(8))) {
    int x, y;
} gpu_hydro_int2;

/* Map particle types to HIP host C representations */

#define HYDRO_DOUBLE2 gpu_hydro_double2
#define HYDRO_DOUBLE4 gpu_hydro_double4
#define HYDRO_FLOAT4  gpu_hydro_float4
#define HYDRO_FLOAT3  gpu_hydro_float3
#define HYDRO_FLOAT2  gpu_hydro_float2
#define HYDRO_INT2    gpu_hydro_int2

#endif

#endif

#ifdef __cplusplus
extern "C" {
#endif

/*! Container for particle data required for density calcs */
struct gpu_part_data_d {
#if defined(WITH_CUDA) || defined(WITH_HIP)
//TODO: This needs changing to doubles too. Darn it...
/*! Particle position and h -> x, y, z, h */
  HYDRO_DOUBLE2 x_y GPU_HYDRO_ALIGN16;
  HYDRO_DOUBLE2 z_h GPU_HYDRO_ALIGN16;

  /*! Particle predicted velocity and mass -> ux, uy, uz, m */
  HYDRO_FLOAT4 vx_m GPU_HYDRO_ALIGN16;
#endif
};

/*! Container for cell positions */
struct gpu_cell_pos {
#if defined(WITH_CUDA) || defined(WITH_HIP)

  /*! Cell position. This is set as the last entry in the
   * range of particles contained within a cell (i.e
   * N+1 contains info for cell position)*/
  HYDRO_DOUBLE4 x;
#endif
};

/*! Over-arching union used to switch
 * between particle data and cell position. Saves us copying
 * cell positions to GPU individually*/
struct gpu_part_send_d {
#if defined(WITH_CUDA) || defined(WITH_HIP)
  union {
    /*! Container for particle data required for density calcs */
    struct gpu_part_data_d p_data;
    /*! Container for cell positions for density calcs */
    struct gpu_cell_pos c_loc;
  };
  #endif
};

/*! Container for particle data sent back to CPU for density calcs */
struct gpu_part_recv_d {
#if defined(WITH_CUDA) || defined(WITH_HIP)

  /*! rho, rho_dh, wcount, wcount_dh */
  HYDRO_FLOAT4 rho_rhodh_wcount_wcount_dh;

  /*! Particle velocity curl; rot_ux and velocity divergence; div_v */
  HYDRO_FLOAT4 rot_vx_div_v;
#endif
};

/*! Container for particle data required for gradient calcs */
struct gpu_part_data_g {
#if defined(WITH_CUDA) || defined(WITH_HIP)

  /*! Particle position & smoothing length */
  HYDRO_DOUBLE2 x_y GPU_HYDRO_ALIGN16;
  HYDRO_DOUBLE2 z_h GPU_HYDRO_ALIGN16;

  /*! Particle velocity and mass */
  HYDRO_FLOAT4 vx_m GPU_HYDRO_ALIGN16;

  /*TODO: aviscmax is no longer needed as we do not use it on GPU. Remove!*/
  /*! Particle internal energy u, density speed of sound and alpha visc
   * Better to use float4 than float3 even if we only need space for 3 vars*/
  HYDRO_FLOAT4 u_rho_c_aviscmax GPU_HYDRO_ALIGN16;

  /*! viscosity information required for particle i when comparing to neighbours*/
  /*Leave as float4 for now. Gives best perf for pipeline async prefetching in kernel
   * due to GPU_HYDRO_ALIGN16 filling bus*/
  HYDRO_FLOAT4 avisc_vsig GPU_HYDRO_ALIGN16;
#endif
};

/*Particle and cell position data required for GPU density computations*/
struct gpu_part_send_g{
#if defined(WITH_CUDA) || defined(WITH_HIP)
  union {
    /*! Container for particle data required for density calcs */
    struct gpu_part_data_g p_data;
    /*! Container for cell positions for density calcs */
    struct gpu_cell_pos c_loc;
  };
  #endif
} ;

/*! Container for particle data sent back to CPU for gradient calcs */
struct gpu_part_recv_g {
#if defined(WITH_CUDA) || defined(WITH_HIP)

  /*! viscosity information results. Better to use float4 than float3
   * even if we have 3 vars */
  HYDRO_FLOAT4 aviscmax_vsig_lapu;
#endif
};

/*! Container for particle data required for force calcs */
struct gpu_part_data_f {
#if defined(WITH_CUDA) || defined(WITH_HIP)

  /* Data required for the calculation: Values read to local GPU memory */

  /*! Particle positions, smoothing length */
  HYDRO_DOUBLE2 x_y GPU_HYDRO_ALIGN16;
  HYDRO_DOUBLE2 z_h GPU_HYDRO_ALIGN16;

  /*! Particle predicted velocity and mass */
  HYDRO_FLOAT4 vx_m GPU_HYDRO_ALIGN16;

  /*! internal energy, density, variable smoothing length term f, pressure */
  HYDRO_FLOAT4 u_rho_f_p GPU_HYDRO_ALIGN16;

  /*! balsara, Particle speed of sound, alpha constants for
   * viscosity and diffusion */
  HYDRO_FLOAT4 bals_c_avisc_adiff GPU_HYDRO_ALIGN16;

  /*! Particle timebin, initial value of min neighbour timebin, start
   * and end index of particles to be interacted with in particle buffer
   * arrays */
  HYDRO_INT2 timebin_minngbtimebin GPU_HYDRO_ALIGN16;
#endif
};

/*! Container for particle data required for force calcs */
struct gpu_part_send_f {
#if defined(WITH_CUDA) || defined(WITH_HIP)
  union {
    /*! Container for particle data required for density calcs */
    struct gpu_part_data_f p_data;
    /*! Container for cell positions for density calcs */
    struct gpu_cell_pos c_loc;
  };
  #endif
} ;

/*! Container for particle data sent back to CPU for force calcs */
struct gpu_part_recv_f {
#if defined(WITH_CUDA) || defined(WITH_HIP)

  /*! Particle acceleration vector */
  HYDRO_FLOAT3 a_hydro;

  /*Note: Needed to revert back to float2 and int
   * due to CUDA atomicMin requiring int*/
  /*! change of u and h with dt, v_sig */
  HYDRO_FLOAT2 udt_hdt;

  /*! change of u and h with dt, v_sig */
  int minngbtb;
#endif
};

#ifdef __cplusplus
}
#endif

#endif  // CUDA_GPU_PART_STRUCTS_H
