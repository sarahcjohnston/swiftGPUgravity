/* GPU headers */
#include "gpu_mapping.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef __cplusplus
}
#endif

struct gravity_gpu_values_send {
  /* floats needed for GPU calculations*/
  float4 values_i; //x:a_x, y:a_y, z:a_z, w:h for cell i
  float4 values_j; //x:a_x, y:a_y, z:a_z, w:h for cell j
  float4 mass; //x:mass_i, y:mass_j, z:0, w:0
  int4 flags0; //x:active_i, y:active_j, z:gcounts, w:cell_active
  int4 flags1; //x:use_full, y:0, z:0, w:0
};

struct gravity_gpu_values_recv {
  /* floats needed for GPU calculations*/
  float4 values_i; //x:a_x, y:a_y, z:a_z, w:pot for cell i
  float4 values_j; //x:a_x, y:a_y, z:a_z, w:pot for cell j
};
