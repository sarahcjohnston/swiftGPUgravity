/* GPU headers */
#include "gpu_mapping.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef __cplusplus
}
#endif

struct gravity_gpu_values_host {
  /* floats needed for GPU calculations on CPU*/
  float* h_i;
  float* h_j;
  float* mass_i;
  float* mass_j;
  float* x_i;
  float* x_j;
  float* y_i;
  float* y_j;
  float* z_i;
  float* z_j;
  float* a_x_i;
  float* a_y_i;
  float* a_z_i;
  float* a_x_j;
  float* a_y_j;
  float* a_z_j;
  float* pot_i;
  float* pot_j;
  int* active_i;
  int* active_j;
  float* CoM_i;
  float* CoM_j;
  int* gcounts;
  int* cell_active;
};

struct gravity_gpu_values_device {
  /* floats needed for GPU calculations on GPU*/
  float* d_h_i;
  float* d_h_j;
  float* d_mass_i;
  float* d_mass_j;
  float* d_x_i;
  float* d_x_j;
  float* d_y_i;
  float* d_y_j;
  float* d_z_i;
  float* d_z_j;
  float* d_a_x_i;
  float* d_a_y_i;
  float* d_a_z_i;
  float* d_a_x_j;
  float* d_a_y_j;
  float* d_a_z_j;
  float* d_pot_i;
  float* d_pot_j;
  int* d_active_i;
  int* d_active_j;
  float* d_CoM_i;
  float* d_CoM_j;
  int* d_gcounts;
  int* d_cell_active;
};

struct gravity_gpu_values_send {
  /* floats needed for GPU calculations*/
  float h_i;
  float h_j;
  float mass_i;
  float mass_j;
  float x_i;
  float x_j;
  float y_i;
  float y_j;
  float z_i;
  float z_j;
  int active_i;
  int active_j;
  int gcounts;
  int cell_active;
  int use_full;
};

struct gravity_gpu_values_recv {
  /* floats needed for GPU calculations*/
  float a_x_i;
  float a_y_i;
  float a_z_i;
  float pot_i;
  float a_x_j;
  float a_y_j;
  float a_z_j;
  float pot_j;
};
