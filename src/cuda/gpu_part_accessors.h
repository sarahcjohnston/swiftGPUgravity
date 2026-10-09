/*******************************************************************************
 * This file is part of SWIFT.
 *
 * Compatibility accessors for the GPU SPHENIX hydro implementation.
 *
 * The original GPU hydro branch used a SPHENIX particle structure with
 * accessor functions such as part_get_rho(), part_get_h(), etc. Current
 * SWIFT stores these quantities directly in struct part.
 *
 * These helpers allow the GPU hydro packing/unpacking code to retain the
 * accessor-based interface without modifying the current SWIFT particle
 * structure.
 ******************************************************************************/

#ifndef SWIFT_GPU_PART_ACCESSORS_H
#define SWIFT_GPU_PART_ACCESSORS_H

/* Config parameters. */
#include <config.h>

/* Local headers. */
#include "inline.h"
#include "part.h"

/*
 * The GPU hydro implementation being ported here is currently written
 * specifically for SPHENIX.
 */
#ifndef SPHENIX_SPH
#error "GPU hydro particle accessors currently require SPHENIX."
#endif

/* ------------------------------------------------------------------------- */
/* Position.                                                                  */
/* ------------------------------------------------------------------------- */

static __attribute__((always_inline)) INLINE double *part_get_x(
    struct part *restrict p) {
  return p->x;
}

static __attribute__((always_inline)) INLINE const double *part_get_const_x(
    const struct part *restrict p) {
  return p->x;
}

static __attribute__((always_inline)) INLINE double part_get_x_ind(
    const struct part *restrict p, const size_t i) {
  return p->x[i];
}

static __attribute__((always_inline)) INLINE void part_set_x(
    struct part *restrict p, const double x[3]) {
  p->x[0] = x[0];
  p->x[1] = x[1];
  p->x[2] = x[2];
}

static __attribute__((always_inline)) INLINE void part_set_x_ind(
    struct part *restrict p, const size_t i, const double x) {
  p->x[i] = x;
}

/* ------------------------------------------------------------------------- */
/* Velocity.                                                                  */
/* ------------------------------------------------------------------------- */

static __attribute__((always_inline)) INLINE float *part_get_v(
    struct part *restrict p) {
  return p->v;
}

static __attribute__((always_inline)) INLINE const float *part_get_const_v(
    const struct part *restrict p) {
  return p->v;
}

static __attribute__((always_inline)) INLINE float part_get_v_ind(
    const struct part *restrict p, const size_t i) {
  return p->v[i];
}

static __attribute__((always_inline)) INLINE void part_set_v(
    struct part *restrict p, const float v[3]) {
  p->v[0] = v[0];
  p->v[1] = v[1];
  p->v[2] = v[2];
}

static __attribute__((always_inline)) INLINE void part_set_v_ind(
    struct part *restrict p, const size_t i, const float v) {
  p->v[i] = v;
}

/* ------------------------------------------------------------------------- */
/* Hydro acceleration.                                                        */
/* ------------------------------------------------------------------------- */

static __attribute__((always_inline)) INLINE float *part_get_a_hydro(
    struct part *restrict p) {
  return p->a_hydro;
}

static __attribute__((always_inline)) INLINE const float *
part_get_const_a_hydro(const struct part *restrict p) {
  return p->a_hydro;
}

static __attribute__((always_inline)) INLINE float part_get_a_hydro_ind(
    const struct part *restrict p, const size_t i) {
  return p->a_hydro[i];
}

static __attribute__((always_inline)) INLINE void part_set_a_hydro(
    struct part *restrict p, const float a_hydro[3]) {
  p->a_hydro[0] = a_hydro[0];
  p->a_hydro[1] = a_hydro[1];
  p->a_hydro[2] = a_hydro[2];
}

static __attribute__((always_inline)) INLINE void part_set_a_hydro_ind(
    struct part *restrict p, const size_t i, const float a_hydro) {
  p->a_hydro[i] = a_hydro;
}

/* ------------------------------------------------------------------------- */
/* Basic hydro quantities.                                                    */
/* ------------------------------------------------------------------------- */

static __attribute__((always_inline)) INLINE float part_get_mass(
    const struct part *restrict p) {
  return p->mass;
}

static __attribute__((always_inline)) INLINE void part_set_mass(
    struct part *restrict p, const float mass) {
  p->mass = mass;
}

static __attribute__((always_inline)) INLINE float part_get_h(
    const struct part *restrict p) {
  return p->h;
}

static __attribute__((always_inline)) INLINE void part_set_h(
    struct part *restrict p, const float h) {
  p->h = h;
}

static __attribute__((always_inline)) INLINE float part_get_u(
    const struct part *restrict p) {
  return p->u;
}

static __attribute__((always_inline)) INLINE void part_set_u(
    struct part *restrict p, const float u) {
  p->u = u;
}

static __attribute__((always_inline)) INLINE float part_get_u_dt(
    const struct part *restrict p) {
  return p->u_dt;
}

static __attribute__((always_inline)) INLINE void part_set_u_dt(
    struct part *restrict p, const float u_dt) {
  p->u_dt = u_dt;
}

static __attribute__((always_inline)) INLINE float part_get_rho(
    const struct part *restrict p) {
  return p->rho;
}

static __attribute__((always_inline)) INLINE void part_set_rho(
    struct part *restrict p, const float rho) {
  p->rho = rho;
}

/* ------------------------------------------------------------------------- */
/* Density-loop quantities.                                                   */
/* ------------------------------------------------------------------------- */

static __attribute__((always_inline)) INLINE float part_get_rho_dh(
    const struct part *restrict p) {
  return p->density.rho_dh;
}

static __attribute__((always_inline)) INLINE void part_set_rho_dh(
    struct part *restrict p, const float rho_dh) {
  p->density.rho_dh = rho_dh;
}

static __attribute__((always_inline)) INLINE float part_get_wcount(
    const struct part *restrict p) {
  return p->density.wcount;
}

static __attribute__((always_inline)) INLINE void part_set_wcount(
    struct part *restrict p, const float wcount) {
  p->density.wcount = wcount;
}

static __attribute__((always_inline)) INLINE float part_get_wcount_dh(
    const struct part *restrict p) {
  return p->density.wcount_dh;
}

static __attribute__((always_inline)) INLINE void part_set_wcount_dh(
    struct part *restrict p, const float wcount_dh) {
  p->density.wcount_dh = wcount_dh;
}

static __attribute__((always_inline)) INLINE float *part_get_rot_v(
    struct part *restrict p) {
  return p->density.rot_v;
}

static __attribute__((always_inline)) INLINE const float *part_get_const_rot_v(
    const struct part *restrict p) {
  return p->density.rot_v;
}

static __attribute__((always_inline)) INLINE float part_get_rot_v_ind(
    const struct part *restrict p, const size_t i) {
  return p->density.rot_v[i];
}

static __attribute__((always_inline)) INLINE void part_set_rot_v(
    struct part *restrict p, const float rot_v[3]) {
  p->density.rot_v[0] = rot_v[0];
  p->density.rot_v[1] = rot_v[1];
  p->density.rot_v[2] = rot_v[2];
}

static __attribute__((always_inline)) INLINE void part_set_rot_v_ind(
    struct part *restrict p, const size_t i, const float rot_v) {
  p->density.rot_v[i] = rot_v;
}

/* ------------------------------------------------------------------------- */
/* Viscosity quantities.                                                      */
/* ------------------------------------------------------------------------- */

static __attribute__((always_inline)) INLINE float part_get_div_v(
    const struct part *restrict p) {
  return p->viscosity.div_v;
}

static __attribute__((always_inline)) INLINE void part_set_div_v(
    struct part *restrict p, const float div_v) {
  p->viscosity.div_v = div_v;
}

static __attribute__((always_inline)) INLINE float part_get_div_v_dt(
    const struct part *restrict p) {
  return p->viscosity.div_v_dt;
}

static __attribute__((always_inline)) INLINE void part_set_div_v_dt(
    struct part *restrict p, const float div_v_dt) {
  p->viscosity.div_v_dt = div_v_dt;
}

static __attribute__((always_inline)) INLINE float
part_get_div_v_previous_step(const struct part *restrict p) {
  return p->viscosity.div_v_previous_step;
}

static __attribute__((always_inline)) INLINE void
part_set_div_v_previous_step(struct part *restrict p,
                             const float div_v_previous_step) {
  p->viscosity.div_v_previous_step = div_v_previous_step;
}

static __attribute__((always_inline)) INLINE float part_get_alpha_av(
    const struct part *restrict p) {
  return p->viscosity.alpha;
}

static __attribute__((always_inline)) INLINE void part_set_alpha_av(
    struct part *restrict p, const float alpha) {
  p->viscosity.alpha = alpha;
}

static __attribute__((always_inline)) INLINE float part_get_v_sig(
    const struct part *restrict p) {
  return p->viscosity.v_sig;
}

static __attribute__((always_inline)) INLINE void part_set_v_sig(
    struct part *restrict p, const float v_sig) {
  p->viscosity.v_sig = v_sig;
}

/* ------------------------------------------------------------------------- */
/* Thermal-diffusion quantities.                                              */
/* ------------------------------------------------------------------------- */

static __attribute__((always_inline)) INLINE float part_get_laplace_u(
    const struct part *restrict p) {
  return p->diffusion.laplace_u;
}

static __attribute__((always_inline)) INLINE void part_set_laplace_u(
    struct part *restrict p, const float laplace_u) {
  p->diffusion.laplace_u = laplace_u;
}

static __attribute__((always_inline)) INLINE float part_get_alpha_diff(
    const struct part *restrict p) {
  return p->diffusion.alpha;
}

static __attribute__((always_inline)) INLINE void part_set_alpha_diff(
    struct part *restrict p, const float alpha) {
  p->diffusion.alpha = alpha;
}

/* ------------------------------------------------------------------------- */
/* Force-loop quantities.                                                     */
/* ------------------------------------------------------------------------- */

static __attribute__((always_inline)) INLINE float part_get_f_gradh(
    const struct part *restrict p) {
  return p->force.f;
}

static __attribute__((always_inline)) INLINE void part_set_f_gradh(
    struct part *restrict p, const float f_gradh) {
  p->force.f = f_gradh;
}

static __attribute__((always_inline)) INLINE float part_get_pressure(
    const struct part *restrict p) {
  return p->force.pressure;
}

static __attribute__((always_inline)) INLINE void part_set_pressure(
    struct part *restrict p, const float pressure) {
  p->force.pressure = pressure;
}

static __attribute__((always_inline)) INLINE float part_get_soundspeed(
    const struct part *restrict p) {
  return p->force.soundspeed;
}

static __attribute__((always_inline)) INLINE void part_set_soundspeed(
    struct part *restrict p, const float soundspeed) {
  p->force.soundspeed = soundspeed;
}

static __attribute__((always_inline)) INLINE float part_get_h_dt(
    const struct part *restrict p) {
  return p->force.h_dt;
}

static __attribute__((always_inline)) INLINE void part_set_h_dt(
    struct part *restrict p, const float h_dt) {
  p->force.h_dt = h_dt;
}

static __attribute__((always_inline)) INLINE float part_get_balsara(
    const struct part *restrict p) {
  return p->force.balsara;
}

static __attribute__((always_inline)) INLINE void part_set_balsara(
    struct part *restrict p, const float balsara) {
  p->force.balsara = balsara;
}

static __attribute__((always_inline)) INLINE float
part_get_alpha_visc_max_ngb(const struct part *restrict p) {
  return p->force.alpha_visc_max_ngb;
}

static __attribute__((always_inline)) INLINE void
part_set_alpha_visc_max_ngb(struct part *restrict p,
                            const float alpha_visc_max_ngb) {
  p->force.alpha_visc_max_ngb = alpha_visc_max_ngb;
}

/* ------------------------------------------------------------------------- */
/* Tree depth / timestep information.                                         */
/* ------------------------------------------------------------------------- */

static __attribute__((always_inline)) INLINE char part_get_depth_h(
    const struct part *restrict p) {
  return p->depth_h;
}

static __attribute__((always_inline)) INLINE void part_set_depth_h(
    struct part *restrict p, const char depth_h) {
  p->depth_h = depth_h;
}

static __attribute__((always_inline)) INLINE timebin_t part_get_time_bin(
    const struct part *restrict p) {
  return p->time_bin;
}

static __attribute__((always_inline)) INLINE void part_set_time_bin(
    struct part *restrict p, const timebin_t time_bin) {
  p->time_bin = time_bin;
}

static __attribute__((always_inline)) INLINE timebin_t
part_get_timestep_limiter_min_ngb_time_bin(
    const struct part *restrict p) {
  return p->limiter_data.min_ngb_time_bin;
}

static __attribute__((always_inline)) INLINE void
part_set_timestep_limiter_min_ngb_time_bin(
    struct part *restrict p, const timebin_t min_ngb_time_bin) {
  p->limiter_data.min_ngb_time_bin = min_ngb_time_bin;
}

#endif /* SWIFT_GPU_PART_ACCESSORS_H */
