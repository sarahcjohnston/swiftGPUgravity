#pragma once
#include "error.h"
#include "gpu_functions.h"
#include "gravity_derivatives.h"
#include "multipole_struct.h"

#include <config.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

__device__ float nearestf1(float dx, const float box_size) {

  return ((dx > 0.5 * box_size)
              ? (dx - box_size)
              : ((dx < -0.5 * box_size) ? (dx + box_size) : dx));

  /*float signx = round(dx/abs(dx+0.0000000000001));
  float adjustx = round(dx/box_size);
  dx -= adjustx*signx*box_size;

return dx;*/
}

__device__ float grav_force_eval(const float u) {

  float W;
#ifdef GADGET2_SOFTENING_CORRECTION
  float W_f_less = 10.6666667f + u * u * (32.f * u - 38.4f);
  float W_f_more = 21.3333333f - 48.f * u + 38.4f * u * u -
                   10.6666667f * u * u * u - 0.06666667f / (u * u * u);
  W = abs(round(round(round(u) / u) - 1)) * W_f_less +
      round(round(u) / u) * W_f_more;
#else

  /* W(u) = 21u^5 - 90u^4 + 140u^3 - 84u^2 + 14 */
  W = 21.f * u - 90.f;
  W = W * u + 140.f;
  W = W * u - 84.f;
  W = W * u;
  W = W * u + 14.f;
#endif
  return W;
}

__device__ float grav_pot_eval(const float u) {

  float W;
#ifdef GADGET2_SOFTENING_CORRECTION
  float W_pot_less =
      -2.8f + u * u * (5.333333333333f + u * u * (6.4f * u - 9.6f));
  float W_pot_more =
      -3.2f + 0.066666666667f / u +
      u * u *
          (10.666666666667f + u * (-16.f + u * (9.6f - 2.133333333333f * u)));
  W = abs(round(round(round(u) / u) - 1)) * W_pot_less +
      round(round(u) / u) * W_pot_more;
#else

  /* W(u) = 3u^7 - 15u^6 + 28u^5 - 21u^4 + 7u^2 - 3 */
  W = 3.f * u - 15.f;
  W = W * u + 28.f;
  W = W * u - 21.f;
  W = W * u;
  W = W * u + 7.f;
  W = W * u;
  W = W * u - 3.f;
#endif
  return W;
}

__device__ float long_grav_eval(const float r_over_r_s, float* corr_f,
                                float* corr_pot) {
#ifdef GADGET2_LONG_RANGE_CORRECTION

  const float two_over_sqrt_pi = ((float)M_2_SQRTPI);

  const float u = 0.5f * r_over_r_s;
  const float u2 = u * u;
  const float exp_u2 = expf(-u2);

  /* Compute erfcf(u) using eq. 7.1.26 of
   * Abramowitz & Stegun, 1972.
   *
   * This has a *relative* error of less than 3.4e-3 over
   * the range of interest (0 < u < 5)\
   *
   * This is a good approximation to use since we already
   * need exp(-u2) */

  const float t = 1.f / (1.f + 0.3275911f * u);

  const float a1 = 0.254829592f;
  const float a2 = -0.284496736f;
  const float a3 = 1.421413741f;
  const float a4 = -1.453152027;
  const float a5 = 1.061405429f;

  /* a1 * t + a2 * t^2 + a3 * t^3 + a4 * t^4 + a5 * t^5 */
  float a = a5 * t + a4;
  a = a * t + a3;
  a = a * t + a2;
  a = a * t + a1;
  a = a * t;

  const float erfc_u = a * exp_u2;

  *corr_pot = erfc_u;
  *corr_f = erfc_u + two_over_sqrt_pi * u * exp_u2;

#else
  const float x = 2.f * r_over_r_s;
  const float exp_x = expf(x);  // good_approx_expf(x);
  const float alpha = 1.f / (1.f + exp_x);

  /* We want 2 - 2 exp(x) * alpha */
  float W = 1.f - alpha * exp_x;
  W = W * 2.f;

  *corr_pot = W;

  /* We want 2*(x*alpha - x*alpha^2 - exp(x)*alpha + 1) */
  W = 1.f - alpha;
  W = W * x - exp_x;
  W = W * alpha + 1.f;
  W = W * 2.f;

  *corr_f = W;
#endif
}

__device__ void iact_grav_pp_full(const float r2, const float h2,
                                  const float h_inv, const float h_inv3,
                                  const float mass, float* f_ij,
                                  float* pot_ij) {

  /* Get the inverse distance */
  const float r_inv = 1.f / sqrtf(r2 + FLT_MIN);

  /* Should we soften ? */
  if (r2 >= h2) {

    /* Get Newtonian gravity */
    *f_ij = mass * r_inv * r_inv * r_inv;
    *pot_ij = -mass * r_inv;

  } else {

    const float r = r2 * r_inv;
    const float ui = r * h_inv;
    const float W_f_ij = grav_force_eval(ui);
    const float W_pot_ij = grav_pot_eval(ui);

    /* Get softened gravity */
    *f_ij = mass * h_inv3 * W_f_ij;
    *pot_ij = mass * h_inv * W_pot_ij;
  }

  /* Get the inverse distance */
  // const float r_inv = 1.f / sqrtf(r2); //no buffer

  /* Should we soften ? */
  /*float f_ij_full = mass * r_inv * r_inv * r_inv;
  float pot_ij_full = -mass * r_inv;

    const float r = r2 * r_inv;
    const float ui = r * h_inv;
    const float W_f_ij = grav_force_eval(ui);
    const float W_pot_ij = grav_pot_eval(ui);*/

  /* Get softened gravity */
  /*float f_ij_soft = mass * h_inv3 * W_f_ij;
  float pot_ij_soft = mass * h_inv * W_pot_ij;


*f_ij = f_ij_full;
*pot_ij = pot_ij_full;

if (r2 < h2){
      *f_ij = f_ij_soft;
      *pot_ij = pot_ij_soft;
      }*/
}

__device__ void iact_grav_pp_truncated(const float r2, const float h2,
                                       const float h_inv, const float h_inv3,
                                       const float mass, const float r_s_inv,
                                       float* f_ij, float* pot_ij) {

  /* Get the inverse distance */
  const float r_inv = 1.f / sqrtf(r2 + FLT_MIN);
  const float r = r2 * r_inv;

  /* Should we soften ? */
  if (r2 >= h2) {

    /* Get Newtonian gravity */
    *f_ij = mass * r_inv * r_inv * r_inv;
    *pot_ij = -mass * r_inv;

  } else {

    const float ui = r * h_inv;
    const float W_f_ij = grav_force_eval(ui);
    const float W_pot_ij = grav_pot_eval(ui);

    /* Get softened gravity */
    *f_ij = mass * h_inv3 * W_f_ij;
    *pot_ij = mass * h_inv * W_pot_ij;
  }

  /* Get long-range correction */
  const float u_lr = r * r_s_inv;
  float corr_f_lr, corr_pot_lr;
  long_grav_eval(u_lr, &corr_f_lr, &corr_pot_lr);
  *f_ij *= corr_f_lr;
  *pot_ij *= corr_pot_lr;

  ////////////////////////////////////////////////////

  /* Get the inverse distance */
  /*const float r_inv = 1.f / sqrtf(r2); //no buffer
  const float r = r2 * r_inv;*/

  /* Should we soften ? */

  /* Get Newtonian gravity */
  /*float f_ij_full = mass * r_inv * r_inv * r_inv;
  float pot_ij_full = -mass * r_inv;

  const float ui = r * h_inv;
  const float W_f_ij = grav_force_eval(ui);
  const float W_pot_ij = grav_pot_eval(ui);*/

  /* Get softened gravity */
  /*float f_ij_soft = mass * h_inv3 * W_f_ij;
  float pot_ij_soft = mass * h_inv * W_pot_ij;

*f_ij = f_ij_full;
*pot_ij = pot_ij_full;

if (r2 < h2){
      *f_ij = f_ij_soft;
      *pot_ij = pot_ij_soft;
      }*/

  /* Get long-range correction */
  /*const float u_lr = r * r_s_inv;
  float corr_f_lr, corr_pot_lr;
  long_grav_eval(u_lr, &corr_f_lr, &corr_pot_lr);
  *f_ij *= corr_f_lr;
  *pot_ij *= corr_pot_lr;*/
}

// PP FULL INTERACTIONS
__device__ void pair_grav_pp_full(
    int* active, float dim_0, float dim_1, float dim_2, float* h_i, float* h_j,
    float* mass_j_arr, float r_s_inv, const float* x_i, const float* x_j,
    const float* y_i, const float* y_j, const float* z_i, const float* z_j,
    float* a_x_i, float* a_y_i, float* a_z_i, float* pot_i, const int gcount_i,
    const int gcount_padded_j, const int periodic, int ci_active, int cj_active,
    int symmetric, int max_r_decision) {

  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int T = blockDim.x * gridDim.x;
  int s = blockIdx.y * blockDim.y + threadIdx.y;
  int S = blockDim.y * gridDim.y;

  for (int pid = t; pid < gcount_i; pid += T) {

    // Local accumulators for the acceleration and potential
    float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

    // Loop over every particle in the other cell.
    for (int pjd = s; pjd < gcount_padded_j; pjd += S) {

      float mass_j = mass_j_arr[pjd];

      // Compute the pairwise distance.
      float dx = x_j[pjd] - x_i[pid];
      float dy = y_j[pjd] - y_i[pid];
      float dz = z_j[pjd] - z_i[pid];

      // Correct for periodic BCs
      dx = nearestf1(dx, dim_0);
      dy = nearestf1(dy, dim_1);
      dz = nearestf1(dz, dim_2);

      const float r2 = dx * dx + dy * dy + dz * dz;

      // Pick the maximal softening length of i and j
      const float h = max(h_i[pid], h_j[pjd]);
      const float h2 = h * h;
      const float h_inv = 1.f / h;
      const float h_inv_3 = h_inv * h_inv * h_inv;

      // Interact!
      float f_ij, pot_ij;
      iact_grav_pp_full(r2, h2, h_inv, h_inv_3, mass_j, &f_ij, &pot_ij);

      // Store it back
      a_x += f_ij * dx;
      a_y += f_ij * dy;
      a_z += f_ij * dz;
      pot += pot_ij;
    }

    // Store everything back in cache
    // accounting for all 4 possibilities of whether treating cell i or j and
    // whether periodic or not
    atomicAdd(
        &a_x_i[pid],
        a_x * active[pid] * ci_active * abs(periodic - 1) +
            a_x * active[pid] * cj_active * symmetric * abs(periodic - 1) +
            a_x * active[pid] * ci_active * periodic * max_r_decision +
            a_x * active[pid] * cj_active * symmetric * periodic *
                max_r_decision);
    atomicAdd(
        &a_y_i[pid],
        a_y * active[pid] * ci_active * abs(periodic - 1) +
            a_y * active[pid] * cj_active * symmetric * abs(periodic - 1) +
            a_y * active[pid] * ci_active * periodic * max_r_decision +
            a_y * active[pid] * cj_active * symmetric * periodic *
                max_r_decision);
    atomicAdd(
        &a_z_i[pid],
        a_z * active[pid] * ci_active * abs(periodic - 1) +
            a_z * active[pid] * cj_active * symmetric * abs(periodic - 1) +
            a_z * active[pid] * ci_active * periodic * max_r_decision +
            a_z * active[pid] * cj_active * symmetric * periodic *
                max_r_decision);
    atomicAdd(
        &pot_i[pid],
        pot * active[pid] * ci_active * abs(periodic - 1) +
            pot * active[pid] * cj_active * symmetric * abs(periodic - 1) +
            pot * active[pid] * ci_active * periodic * max_r_decision +
            pot * active[pid] * cj_active * symmetric * periodic *
                max_r_decision);
  }
}

// PP TRUNCATED INTERACTIONS
__device__ void pair_grav_pp_truncated(
    int* active, float dim_0, float dim_1, float dim_2, float* h_i, float* h_j,
    float* mass_j_arr, const float r_s_inv, const float* x_i, const float* x_j,
    const float* y_i, const float* y_j, const float* z_i, const float* z_j,
    float* a_x_i, float* a_y_i, float* a_z_i, float* pot_i, const int gcount_i,
    const int gcount_padded_j, const int periodic, int ci_active, int cj_active,
    int symmetric, int max_r_decision) {

  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int T = blockDim.x * gridDim.x;
  int s = blockIdx.y * blockDim.y + threadIdx.y;
  int S = blockDim.y * gridDim.y;

  /* Loop over all particles in ci... */
  for (int pid = t; pid < gcount_i; pid += T) {

    /* Local accumulators for the acceleration and potential */
    float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

    /* Loop over every particle in the other cell. */
    for (int pjd = s; pjd < gcount_padded_j; pjd += S) {

      const float mass_j = mass_j_arr[pjd];

      // Compute the pairwise distance.
      float dx = x_j[pjd] - x_i[pid];
      float dy = y_j[pjd] - y_i[pid];
      float dz = z_j[pjd] - z_i[pid];

      /* Correct for periodic BCs */
      dx = nearestf1(dx, dim_0);
      dy = nearestf1(dy, dim_1);
      dz = nearestf1(dz, dim_2);

      const float r2 = dx * dx + dy * dy + dz * dz;

      /* Pick the maximal softening length of i and j */
      const float h = max(h_i[pid], h_j[pjd]);
      const float h2 = h * h;
      const float h_inv = 1.f / h;
      const float h_inv_3 = h_inv * h_inv * h_inv;

      /* Interact! */
      float f_ij, pot_ij;
      iact_grav_pp_truncated(r2, h2, h_inv, h_inv_3, mass_j, r_s_inv, &f_ij,
                             &pot_ij);

      /* Store it back */
      a_x += f_ij * dx;
      a_y += f_ij * dy;
      a_z += f_ij * dz;
      pot += pot_ij;
    }

    /* Store everything back in cache */
    // treating both possibilities of whether treating cell i or cell j
    atomicAdd(&a_x_i[pid], a_x * active[pid] * ci_active * periodic *
                                   abs(max_r_decision - 1) +
                               a_x * active[pid] * cj_active * symmetric *
                                   periodic * abs(max_r_decision - 1));
    atomicAdd(&a_y_i[pid], a_y * active[pid] * ci_active * periodic *
                                   abs(max_r_decision - 1) +
                               a_y * active[pid] * cj_active * symmetric *
                                   periodic * abs(max_r_decision - 1));
    atomicAdd(&a_z_i[pid], a_z * active[pid] * ci_active * periodic *
                                   abs(max_r_decision - 1) +
                               a_z * active[pid] * cj_active * symmetric *
                                   periodic * abs(max_r_decision - 1));
    atomicAdd(&pot_i[pid], pot * active[pid] * ci_active * periodic *
                                   abs(max_r_decision - 1) +
                               pot * active[pid] * cj_active * symmetric *
                                   periodic * abs(max_r_decision - 1));
  }
}

// PP FULL INTERACTIONS
__global__ void pair_grav_pp_full_refactor(
    struct gravity_gpu_values_send* gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_d, float dim_0,
    float dim_1, float dim_2, const float r_s_inv, const int gcount_i,
    const int gcount_padded_j, const int periodic, int ci_active, int cj_active,
    int symmetric, int max_r_decision, int ncells, int max_cell_size) {

  int cell = blockIdx.x;
  if (cell >= ncells) return;

  int cell_space = cell * max_cell_size;
  int counts = gravity_gpu_values_send_d[cell_space].gcounts;

  int pid = blockIdx.y * blockDim.x + threadIdx.x;
  if (pid >= gcount_i) return;

  float xi = gravity_gpu_values_send_d[cell_space + pid].x_i;
  float yi = gravity_gpu_values_send_d[cell_space + pid].y_i;
  float zi = gravity_gpu_values_send_d[cell_space + pid].z_i;
  float hi = gravity_gpu_values_send_d[cell_space + pid].h_i;

  const int act =
      (gravity_gpu_values_send_d[cell_space + pid].active_i > 0) ? 1 : 0;
  if (!act) return;

  float nonper = (float)abs(periodic - 1);
  float per = (float)(periodic * max_r_decision);

  float factor_ci = (float)ci_active * (nonper + per);
  float factor_cj = (float)(cj_active * symmetric) * (nonper + per);
  float factor = factor_ci + factor_cj;
  const float scale = (float)act * factor;

  // Local accumulators for the acceleration and potential
  float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

  // Loop over every particle in the other cell.
  for (int pjd = 0; pjd < gcount_padded_j; ++pjd) {

    const gravity_gpu_values_send pj =
        gravity_gpu_values_send_d[cell_space + pjd];

    float mass_j = gravity_gpu_values_send_d[pjd + cell * max_cell_size].mass_j;

    // Compute the pairwise distance.
    float dx = pj.x_i - xi;
    float dy = pj.y_i - yi;
    float dz = pj.z_i - zi;

    // Correct for periodic BCs
    dx = nearestf1(dx, dim_0);
    dy = nearestf1(dy, dim_1);
    dz = nearestf1(dz, dim_2);

    const float r2 = dx * dx + dy * dy + dz * dz;

    // Pick the maximal softening length of i and j
    const float h =
        max(gravity_gpu_values_send_d[pid + cell * max_cell_size].h_i,
            gravity_gpu_values_send_d[cell_space + pjd].h_j);
    const float h2 = h * h;
    const float h_inv = 1.f / h;
    const float h_inv_3 = h_inv * h_inv * h_inv;

    // Interact!
    float f_ij, pot_ij;
    iact_grav_pp_full(r2, h2, h_inv, h_inv_3, mass_j, &f_ij, &pot_ij);

    // Store it back
    a_x += f_ij * dx;
    a_y += f_ij * dy;
    a_z += f_ij * dz;
    pot += pot_ij;
  }

  // Store everything back in cache
  // accounting for all 4 possibilities of whether treating cell i or j and
  // whether periodic or not
  gravity_gpu_values_recv_d[cell_space + pid].a_x_i += a_x * scale;
  gravity_gpu_values_recv_d[cell_space + pid].a_y_i += a_y * scale;
  gravity_gpu_values_recv_d[cell_space + pid].a_z_i += a_z * scale;
  gravity_gpu_values_recv_d[cell_space + pid].pot_i += pot * scale;
}

// PP TRUNCATED INTERACTIONS
__global__ void pair_grav_pp_truncated_refactor(
    struct gravity_gpu_values_send* gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_d, float dim_0,
    float dim_1, float dim_2, const float r_s_inv, const int gcount_i,
    const int gcount_padded_j, const int periodic, int ci_active, int cj_active,
    int symmetric, int max_r_decision, int ncells, int max_cell_size) {

  int cell = blockIdx.x;
  if (cell >= ncells) return;

  int cell_space = cell * max_cell_size;
  int counts = gravity_gpu_values_send_d[cell_space].gcounts;

  // One thread per i-particle, like your doself kernel style (blockIdx.y used
  // for pid-tiling)
  int pid = blockIdx.y * blockDim.x + threadIdx.x;
  if (pid >= counts) return;

  /*values for particle*/
  float xi = gravity_gpu_values_send_d[cell_space + pid].x_i;
  float yi = gravity_gpu_values_send_d[cell_space + pid].y_i;
  float zi = gravity_gpu_values_send_d[cell_space + pid].z_i;
  float hi = gravity_gpu_values_send_d[cell_space + pid].h_i;

  const int act =
      (gravity_gpu_values_send_d[cell_space + pid].active_i > 0) ? 1 : 0;
  if (!act) return;

  const float per_trunc = (float)periodic * (float)abs(max_r_decision - 1);
  const float factor =
      per_trunc * ((float)ci_active + (float)(cj_active * symmetric));
  const float scale = act * factor;

  /* Local accumulators for the acceleration and potential */
  float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

  /* Loop over every particle in the other cell. */
  for (int pjd = 0; pjd < gcount_padded_j; ++pjd) {

    const gravity_gpu_values_send pj =
        gravity_gpu_values_send_d[cell_space + pjd];
    const float mass_j =
        gravity_gpu_values_send_d[pjd + cell * max_cell_size].mass_j;

    // Compute the pairwise distance.
    float dx = pj.x_i - xi;
    float dy = pj.y_i - yi;
    float dz = pj.z_i - zi;

    /* Correct for periodic BCs */
    dx = nearestf1(dx, dim_0);
    dy = nearestf1(dy, dim_1);
    dz = nearestf1(dz, dim_2);

    const float r2 = dx * dx + dy * dy + dz * dz;

    /* Pick the maximal softening length of i and j */
    const float h =
        max(gravity_gpu_values_send_d[pid + cell * max_cell_size].h_i,
            gravity_gpu_values_send_d[cell_space + pjd].h_j);
    const float h2 = h * h;
    const float h_inv = 1.f / h;
    const float h_inv_3 = h_inv * h_inv * h_inv;

    /* Interact! */
    float f_ij, pot_ij;
    iact_grav_pp_truncated(r2, h2, h_inv, h_inv_3, mass_j, r_s_inv, &f_ij,
                           &pot_ij);

    /* Store it back */
    a_x += f_ij * dx;
    a_y += f_ij * dy;
    a_z += f_ij * dz;
    pot += pot_ij;
  }

  /* Store everything back in cache */
  // treating both possibilities of whether treating cell i or cell j
  gravity_gpu_values_recv_d[cell_space + pid].a_x_i += a_x * scale;
  gravity_gpu_values_recv_d[cell_space + pid].a_y_i += a_y * scale;
  gravity_gpu_values_recv_d[cell_space + pid].a_z_i += a_z * scale;
  gravity_gpu_values_recv_d[cell_space + pid].pot_i += pot * scale;
}

// SELF PP FULL INTERACTIONS
__device__ void doself_grav_pp_full(
    int* active, float* h_i, float* mass_i_arr, const float* x_i,
    const float* y_i, const float* z_i, float* a_x_i, float* a_y_i,
    float* a_z_i, float* pot_i, const int gcount_i, const int gcount_padded_i,
    const int periodic, int ci_active, int max_r_decision, int ncells,
    int max_cell_size, int* gcounts, int* cell_active) {

  for (int cell = 0; cell < ncells; cell++) {

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int T = blockDim.x * gridDim.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int S = blockDim.y * gridDim.y;

    /* Loop over all particles in ci... */
    for (int pid = t; pid < gcount_i; pid += T) {

      if (pid >= gcount_i) continue;

      /* Local accumulators for the acceleration */
      float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

      /* Loop over every other particle in the cell. */
      for (int pjd = s; pjd < gcount_padded_i; pjd += S) {

        /* No self interaction */
        if (pid == pjd) continue;

        const float mass_i = mass_i_arr[pjd + cell * max_cell_size];

        /* Compute the pairwise (square) distance. */
        /* Note: no need for periodic wrapping inside a cell */
        float dx =
            x_i[pjd + cell * max_cell_size] - x_i[pid + cell * max_cell_size];
        float dy =
            y_i[pjd + cell * max_cell_size] - y_i[pid + cell * max_cell_size];
        float dz =
            z_i[pjd + cell * max_cell_size] - z_i[pid + cell * max_cell_size];
        const float r2 = dx * dx + dy * dy + dz * dz;

        /* Pick the maximal softening length of i and j */
        const float h = max(h_i[pid + cell * max_cell_size],
                            h_i[pjd + cell * max_cell_size]);
        const float h2 = h * h;
        const float h_inv = 1.f / h;
        const float h_inv_3 = h_inv * h_inv * h_inv;

        /* Interact! */
        float f_ij, pot_ij;
        iact_grav_pp_full(r2, h2, h_inv, h_inv_3, mass_i, &f_ij, &pot_ij);

        /* Store it back */
        a_x += f_ij * dx;
        a_y += f_ij * dy;
        a_z += f_ij * dz;
        pot += pot_ij;
      }
      int act = 0;
      if (active[pid] > 0) act = 1;

      /* Store everything back into values */
      atomicAdd(&a_x_i[pid + cell * max_cell_size],
                a_x * act * cell_active[cell] * abs(periodic - 1) +
                    a_x * act * cell_active[cell] * periodic * max_r_decision);
      atomicAdd(&a_y_i[pid + cell * max_cell_size],
                a_y * act * cell_active[cell] * abs(periodic - 1) +
                    a_y * act * cell_active[cell] * periodic * max_r_decision);
      atomicAdd(&a_z_i[pid + cell * max_cell_size],
                a_z * act * cell_active[cell] * abs(periodic - 1) +
                    a_z * act * cell_active[cell] * periodic * max_r_decision);
      atomicAdd(&pot_i[pid + cell * max_cell_size],
                pot * act * cell_active[cell] * abs(periodic - 1) +
                    pot * act * cell_active[cell] * periodic * max_r_decision);
    }
  }
}

// SELF PP TRUNCATED INTERACTIONS
__device__ void doself_grav_pp_truncated(
    int* active, float* h_i, float* mass_i_arr, float r_s_inv, const float* x_i,
    const float* y_i, const float* z_i, float* a_x_i, float* a_y_i,
    float* a_z_i, float* pot_i, const int gcount_i, const int gcount_padded_i,
    const int periodic, int ci_active, int max_r_decision, int ncells,
    int max_cell_size, int* gcounts, int* cell_active) {

  for (int cell = 0; cell < ncells; cell++) {

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int T = blockDim.x * gridDim.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int S = blockDim.y * gridDim.y;

    /* Loop over all particles in ci... */
    for (int pid = t; pid < gcounts[cell]; pid += T) {

      if (pid >= gcounts[cell]) continue;

      /* Local accumulators for the acceleration and potential */
      float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

      /* Loop over every other particle in the cell. */
      for (int pjd = s; pjd < gcounts[cell]; pjd += S) {

        /* No self interaction */
        if (pid == pjd) continue;

        /* Get info about j */
        const float mass_i = mass_i_arr[pjd + cell * max_cell_size];

        /* Compute the pairwise (square) distance. */
        /* Note: no need for periodic wrapping inside a cell */
        float dx =
            x_i[pjd + cell * max_cell_size] - x_i[pid + cell * max_cell_size];
        float dy =
            y_i[pjd + cell * max_cell_size] - y_i[pid + cell * max_cell_size];
        float dz =
            z_i[pjd + cell * max_cell_size] - z_i[pid + cell * max_cell_size];

        const float r2 = dx * dx + dy * dy + dz * dz;

        /* Pick the maximal softening length of i and j */
        const float h = max(h_i[pid + cell * max_cell_size],
                            h_i[pjd + cell * max_cell_size]);
        const float h2 = h * h;
        const float h_inv = 1.f / h;
        const float h_inv_3 = h_inv * h_inv * h_inv;

        /* Interact! */
        float f_ij, pot_ij;
        iact_grav_pp_truncated(r2, h2, h_inv, h_inv_3, mass_i, r_s_inv, &f_ij,
                               &pot_ij);

        /* Store it back */
        a_x += f_ij * dx;
        a_y += f_ij * dy;
        a_z += f_ij * dz;
        pot += pot_ij;
      }
      int act = 0;
      if (active[pid] > 0) act = 1;
      /*if (active[pid] == 0)
          printf("active: %i \n", active[pid]);*/

      int per = 0;
      if (periodic > 0) per = 1;

      /* Store everything back into values */
      // printf("cell active: %i\n", cell_active[cell]);
      atomicAdd(&a_x_i[pid + cell * max_cell_size],
                a_x * cell_active[cell] * per *
                    abs(max_r_decision - 1));  //*act*ci_active
      atomicAdd(&a_y_i[pid + cell * max_cell_size],
                a_y * cell_active[cell] * per *
                    abs(max_r_decision - 1));  //*act*ci_active
      atomicAdd(&a_z_i[pid + cell * max_cell_size],
                a_z * cell_active[cell] * per *
                    abs(max_r_decision - 1));  //*act*ci_active
      atomicAdd(&pot_i[pid + cell * max_cell_size],
                pot * cell_active[cell] * per *
                    abs(max_r_decision - 1));  //*act*ci_active
    }
  }
}

// SELF PP FULL INTERACTIONS
__device__ void doself_grav_pp_full_new(
    struct gravity_gpu_values_send* gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_d, float r_s_inv,
    const int gcount_i, const int gcount_padded_i, const int periodic,
    int ci_active, int max_r_decision, int ncells, int max_cell_size) {

  for (int cell = 0; cell < ncells; cell++) {

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int T = blockDim.x * gridDim.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int S = blockDim.y * gridDim.y;

    /* Loop over all particles in ci... */
    for (int pid = t;
         pid < gravity_gpu_values_send_d[cell * max_cell_size].gcounts;
         pid += T) {

      if (pid >= gcount_i) continue;

      /* Local accumulators for the acceleration */
      float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

      /* Loop over every other particle in the cell. */
      for (int pjd = s;
           pjd < gravity_gpu_values_send_d[cell * max_cell_size].gcounts;
           pjd += S) {

        /* No self interaction */
        if (pid == pjd) continue;

        /* Get info about j */
        const float mass_i =
            gravity_gpu_values_send_d[pjd + cell * max_cell_size].mass_i;

        /* Compute the pairwise (square) distance. */
        /* Note: no need for periodic wrapping inside a cell */
        float dx = gravity_gpu_values_send_d[pjd + cell * max_cell_size].x_i -
                   gravity_gpu_values_send_d[pid + cell * max_cell_size].x_i;
        float dy = gravity_gpu_values_send_d[pjd + cell * max_cell_size].y_i -
                   gravity_gpu_values_send_d[pid + cell * max_cell_size].y_i;
        float dz = gravity_gpu_values_send_d[pjd + cell * max_cell_size].z_i -
                   gravity_gpu_values_send_d[pid + cell * max_cell_size].z_i;

        const float r2 = dx * dx + dy * dy + dz * dz;

        /* Pick the maximal softening length of i and j */
        const float h =
            max(gravity_gpu_values_send_d[pid + cell * max_cell_size].h_i,
                gravity_gpu_values_send_d[pjd + cell * max_cell_size].h_i);
        const float h2 = h * h;
        const float h_inv = 1.f / h;
        const float h_inv_3 = h_inv * h_inv * h_inv;

        /* Interact! */
        float f_ij, pot_ij;
        iact_grav_pp_full(r2, h2, h_inv, h_inv_3, mass_i, &f_ij, &pot_ij);

        /* Store it back */
        a_x += f_ij * dx;
        a_y += f_ij * dy;
        a_z += f_ij * dz;
        pot += pot_ij;
      }
      int act = 0;
      if (gravity_gpu_values_send_d[pid + cell * max_cell_size].active_i > 0)
        act = 1;

      /* Store everything back into values */
      atomicAdd(
          &gravity_gpu_values_recv_d[pid + cell * max_cell_size].a_x_i,
          a_x * act *
                  gravity_gpu_values_send_d[cell * max_cell_size].cell_active *
                  abs(periodic - 1) +
              a_x * act *
                  gravity_gpu_values_send_d[cell * max_cell_size].cell_active *
                  periodic * max_r_decision);

      atomicAdd(
          &gravity_gpu_values_recv_d[pid + cell * max_cell_size].a_y_i,
          a_y * act *
                  gravity_gpu_values_send_d[cell * max_cell_size].cell_active *
                  abs(periodic - 1) +
              a_y * act *
                  gravity_gpu_values_send_d[cell * max_cell_size].cell_active *
                  periodic * max_r_decision);

      atomicAdd(
          &gravity_gpu_values_recv_d[pid + cell * max_cell_size].a_z_i,
          a_z * act *
                  gravity_gpu_values_send_d[cell * max_cell_size].cell_active *
                  abs(periodic - 1) +
              a_z * act *
                  gravity_gpu_values_send_d[cell * max_cell_size].cell_active *
                  periodic * max_r_decision);

      atomicAdd(
          &gravity_gpu_values_recv_d[pid + cell * max_cell_size].pot_i,
          pot * act *
                  gravity_gpu_values_send_d[cell * max_cell_size].cell_active *
                  abs(periodic - 1) +
              pot * act *
                  gravity_gpu_values_send_d[cell * max_cell_size].cell_active *
                  periodic * max_r_decision);
    }
  }
}

// SELF PP TRUNCATED INTERACTIONS
__device__ void doself_grav_pp_truncated_new(
    struct gravity_gpu_values_send* gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_d, float r_s_inv,
    const int gcount_i, const int gcount_padded_i, const int periodic,
    int ci_active, int max_r_decision, int ncells, int max_cell_size) {

  for (int cell = 0; cell < ncells; cell++) {

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int T = blockDim.x * gridDim.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int S = blockDim.y * gridDim.y;

    /* Loop over all particles in ci... */
    for (int pid = t;
         pid < gravity_gpu_values_send_d[cell * max_cell_size].gcounts;
         pid += T) {

      /* Local accumulators for the acceleration and potential */
      float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

      /* Loop over every other particle in the cell. */
      for (int pjd = s;
           pjd < gravity_gpu_values_send_d[cell * max_cell_size].gcounts;
           pjd += S) {

        /* No self interaction */
        if (pid == pjd) continue;

        /* Get info about j */
        const float mass_i =
            gravity_gpu_values_send_d[pjd + cell * max_cell_size].mass_i;

        /* Compute the pairwise (square) distance. */
        /* Note: no need for periodic wrapping inside a cell */
        float dx = gravity_gpu_values_send_d[pjd + cell * max_cell_size].x_i -
                   gravity_gpu_values_send_d[pid + cell * max_cell_size].x_i;
        float dy = gravity_gpu_values_send_d[pjd + cell * max_cell_size].y_i -
                   gravity_gpu_values_send_d[pid + cell * max_cell_size].y_i;
        float dz = gravity_gpu_values_send_d[pjd + cell * max_cell_size].z_i -
                   gravity_gpu_values_send_d[pid + cell * max_cell_size].z_i;

        const float r2 = dx * dx + dy * dy + dz * dz;

        /* Pick the maximal softening length of i and j */
        const float h =
            max(gravity_gpu_values_send_d[pid + cell * max_cell_size].h_i,
                gravity_gpu_values_send_d[pjd + cell * max_cell_size].h_i);
        const float h2 = h * h;
        const float h_inv = 1.f / h;
        const float h_inv_3 = h_inv * h_inv * h_inv;

        /* Interact! */
        float f_ij, pot_ij;
        iact_grav_pp_truncated(r2, h2, h_inv, h_inv_3, mass_i, r_s_inv, &f_ij,
                               &pot_ij);

        /* Store it back */
        a_x += f_ij * dx;
        a_y += f_ij * dy;
        a_z += f_ij * dz;
        pot += pot_ij;
      }
      int act = 0;
      if (gravity_gpu_values_send_d[pid + cell * max_cell_size].active_i > 0)
        act = 1;
      /*if (active[pid] == 0)
          printf("active: %i \n", active[pid]);*/

      int per = 0;
      if (periodic > 0) per = 1;

      /* Store everything back into values */
      // printf("cell active: %i\n", cell_active[cell]);
      atomicAdd(
          &gravity_gpu_values_recv_d[pid + cell * max_cell_size].a_x_i,
          a_x * gravity_gpu_values_send_d[cell * max_cell_size].cell_active *
              per * abs(max_r_decision - 1));  //*act*ci_active
      atomicAdd(
          &gravity_gpu_values_recv_d[pid + cell * max_cell_size].a_y_i,
          a_y * gravity_gpu_values_send_d[cell * max_cell_size].cell_active *
              per * abs(max_r_decision - 1));  //*act*ci_active
      atomicAdd(
          &gravity_gpu_values_recv_d[pid + cell * max_cell_size].a_z_i,
          a_z * gravity_gpu_values_send_d[cell * max_cell_size].cell_active *
              per * abs(max_r_decision - 1));  //*act*ci_active
      atomicAdd(
          &gravity_gpu_values_recv_d[pid + cell * max_cell_size].pot_i,
          pot * gravity_gpu_values_send_d[cell * max_cell_size].cell_active *
              per * abs(max_r_decision - 1));  //*act*ci_active
    }
  }
}

__global__ void doself_grav_pp_full_new_refactor(
    struct gravity_gpu_values_send* gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_d, float r_s_inv,
    const int gcount_i, const int gcount_padded_i, const int periodic,
    int ci_active, int max_r_decision, int ncells, int max_cell_size) {

  int cell = blockIdx.x;
  if (cell >= ncells) return;

  int cell_space = cell * max_cell_size;
  int counts = gravity_gpu_values_send_d[cell_space].gcounts;

  float factor1 =
      gravity_gpu_values_send_d[cell_space].cell_active * abs(periodic - 1);
  float factor2 = gravity_gpu_values_send_d[cell_space].cell_active * periodic *
                  max_r_decision;

  int pid = blockIdx.y * blockDim.x + threadIdx.x;
  if (pid >= counts) return;

  /*values for particle*/
  float xi = gravity_gpu_values_send_d[cell_space + pid].x_i;
  float yi = gravity_gpu_values_send_d[cell_space + pid].y_i;
  float zi = gravity_gpu_values_send_d[cell_space + pid].z_i;
  float hi = gravity_gpu_values_send_d[cell_space + pid].h_i;

  /* Local accumulators for the acceleration and potential */
  float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

  /*extern __shared__ gravity_gpu_values_send send[];

  for (int i0 = 0; i0 < counts; i0 += blockDim.x) {

          int i = i0 + threadIdx.x;
          if (i < counts) send[threadIdx.x] =
  gravity_gpu_values_send_d[cell_space + i];
          __syncthreads();

  int tile = min(blockDim.x, counts - i0);

  for (int j = 0; j < tile; j++) {
          int pjd = i0 + j;*/

  /* Loop over every other particle in the cell. */
  for (int pjd = 0; pjd < counts; pjd++) {

    /* No self interaction */
    if (pid == pjd) continue;

    const gravity_gpu_values_send pj =
        gravity_gpu_values_send_d[cell_space + pjd];

    /* Get info about j */
    const float mass_i = pj.mass_i;

    /* Compute the pairwise (square) distance. */
    /* Note: no need for periodic wrapping inside a cell */
    float dx = pj.x_i - xi;
    float dy = pj.y_i - yi;
    float dz = pj.z_i - zi;

    const float r2 = dx * dx + dy * dy + dz * dz;

    /* Pick the maximal softening length of i and j */
    const float h = max(hi, pj.h_i);
    const float h2 = h * h;
    const float h_inv = 1.f / h;
    const float h_inv_3 = h_inv * h_inv * h_inv;

    /* Interact! */
    float f_ij, pot_ij;
    iact_grav_pp_full(r2, h2, h_inv, h_inv_3, mass_i, &f_ij, &pot_ij);

    /* Store it back */
    a_x += f_ij * dx;
    a_y += f_ij * dy;
    a_z += f_ij * dz;
    pot += pot_ij;
  }
  //__syncthreads();

  int act = 0;
  if (gravity_gpu_values_send_d[pid + cell_space].active_i > 0) act = 1;

  gravity_gpu_values_recv_d[cell_space + pid].a_x_i +=
      a_x * act * (factor1 + factor2);
  gravity_gpu_values_recv_d[cell_space + pid].a_y_i +=
      a_y * act * (factor1 + factor2);
  gravity_gpu_values_recv_d[cell_space + pid].a_z_i +=
      a_z * act * (factor1 + factor2);
  gravity_gpu_values_recv_d[cell_space + pid].pot_i +=
      pot * act * (factor1 + factor2);
}

// SELF PP TRUNCATED INTERACTIONS
__global__ void doself_grav_pp_truncated_new_refactor(
    struct gravity_gpu_values_send* gravity_gpu_values_send_d,
    struct gravity_gpu_values_recv* gravity_gpu_values_recv_d, float r_s_inv,
    const int gcount_i, const int gcount_padded_i, const int periodic,
    int ci_active, int max_r_decision, int ncells, int max_cell_size) {

  int cell = blockIdx.x;
  if (cell >= ncells) return;

  int cell_space = cell * max_cell_size;
  int counts = gravity_gpu_values_send_d[cell_space].gcounts;
  // printf("counts: %i \n", counts);

  float factor = gravity_gpu_values_send_d[cell_space].cell_active *
                 (periodic ? 1.f : 0.f) * abs(max_r_decision - 1);
  if (factor == 0) return;

  int pid = blockIdx.y * blockDim.x + threadIdx.x;
  if (pid >= counts) return;

  /*values for particle*/
  float xi = gravity_gpu_values_send_d[cell_space + pid].x_i;
  float yi = gravity_gpu_values_send_d[cell_space + pid].y_i;
  float zi = gravity_gpu_values_send_d[cell_space + pid].z_i;
  float hi = gravity_gpu_values_send_d[cell_space + pid].h_i;

  /* Local accumulators for the acceleration and potential */
  float a_x = 0.f, a_y = 0.f, a_z = 0.f, pot = 0.f;

  /*extern __shared__ gravity_gpu_values_send send[];

  for (int i0 = 0; i0 < counts; i0 += blockDim.x) {

          int i = i0 + threadIdx.x;
          if (i < counts) send[threadIdx.x] =
  gravity_gpu_values_send_d[cell_space + i];
          __syncthreads();

  int tile = min(blockDim.x, counts - i0);

  for (int j = 0; j < tile; j++) {
          int pjd = i0 + j;*/

  /* Loop over every other particle in the cell. */
  for (int pjd = 0; pjd < counts; pjd++) {

    /* No self interaction */
    if (pid == pjd) continue;

    const gravity_gpu_values_send pj =
        gravity_gpu_values_send_d[cell_space + pjd];

    /* Get info about j */
    const float mass_i = pj.mass_i;

    /* Compute the pairwise (square) distance. */
    /* Note: no need for periodic wrapping inside a cell */
    float dx = pj.x_i - xi;
    float dy = pj.y_i - yi;
    float dz = pj.z_i - zi;

    const float r2 = dx * dx + dy * dy + dz * dz;

    /* Pick the maximal softening length of i and j */
    const float h = max(hi, pj.h_i);
    const float h2 = h * h;
    const float h_inv = 1.f / h;
    const float h_inv_3 = h_inv * h_inv * h_inv;

    /* Interact! */
    float f_ij, pot_ij;
    iact_grav_pp_truncated(r2, h2, h_inv, h_inv_3, mass_i, r_s_inv, &f_ij,
                           &pot_ij);

    /* Store it back */
    a_x += f_ij * dx;
    a_y += f_ij * dy;
    a_z += f_ij * dz;
    pot += pot_ij;
  }
  //__syncthreads();

  gravity_gpu_values_recv_d[cell_space + pid].a_x_i = a_x * factor;
  gravity_gpu_values_recv_d[cell_space + pid].a_y_i = a_y * factor;
  gravity_gpu_values_recv_d[cell_space + pid].a_z_i = a_z * factor;
  gravity_gpu_values_recv_d[cell_space + pid].pot_i = pot * factor;
}

__global__ void pair_grav_pp_kernel(
    const gravity_gpu_values_send* send, gravity_gpu_values_recv* recv,
    int periodic, float r_s_inv, int symmetric,
    int swap,  // 0: compute i-side (base_i interacts with base_j) 1: compute
               // j-side (base_j interacts with base_i)
    float dim_0, float dim_1, float dim_2, int max_cell_size, int ncells) {

  // printf("In kernel \n");

  int pair_id = blockIdx.x;
  int base0 = (2 * pair_id) * max_cell_size;
  int base1 = base0 + max_cell_size;

  int base_i = swap ? base1 : base0;
  int base_j = swap ? base0 : base1;

  int count_i = send[base_i].gcounts;
  int count_j = send[base_j].gcounts;

  int pid = blockIdx.y * blockDim.x + threadIdx.x;
  if (pid >= count_i) return;

  // Only update if this target cell is active (matches CPU)
  int ci_active = send[base_i].cell_active;
  if (!ci_active) return;

  // active particle gating (matches your current logic)
  int act = (send[base_i + pid].active_i > 0);
  if (!act) return;

  float xi = send[base_i + pid].x_i;
  float yi = send[base_i + pid].y_i;
  float zi = send[base_i + pid].z_i;
  float hi = send[base_i + pid].h_i;

  float ax = 0.f, ay = 0.f, az = 0.f, pot = 0.f;

  int use_full = send[base_i].use_full;  // 1 full, 0 truncated

  for (int pjd = 0; pjd < count_j; ++pjd) {

    // Option A “mirror cj into _i fields”: you said you already did this.
    float xj = send[base_j + pjd].x_i;
    float yj = send[base_j + pjd].y_i;
    float zj = send[base_j + pjd].z_i;
    float mj = send[base_j + pjd].mass_i;  // mirror mass too

    float dx = xj - xi;
    float dy = yj - yi;
    float dz = zj - zi;

    if (periodic) {
      dx = nearestf1(dx, dim_0);  // you’ll pass dims or store globally
      dy = nearestf1(dy, dim_1);
      dz = nearestf1(dz, dim_2);
    }

    float r2 = dx * dx + dy * dy + dz * dz;

    float hj = send[base_j + pjd].h_i;  // mirrored
    float h = max(hi, hj);
    float h2 = h * h;
    float hinv = 1.f / h;
    float hinv3 = hinv * hinv * hinv;

    float fij, potij;
    if (use_full) {
      iact_grav_pp_full(r2, h2, hinv, hinv3, mj, &fij, &potij);
    } else {
      iact_grav_pp_truncated(r2, h2, hinv, hinv3, mj, r_s_inv, &fij, &potij);
    }

    ax += fij * dx;
    ay += fij * dy;
    az += fij * dz;
    pot += potij;
  }

  // Write into the target block’s “i” outputs (so swap=1 fills cjblock a_*_i)
  recv[base_i + pid].a_x_i += ax;
  recv[base_i + pid].a_y_i += ay;
  recv[base_i + pid].a_z_i += az;
  recv[base_i + pid].pot_i += pot;
}
