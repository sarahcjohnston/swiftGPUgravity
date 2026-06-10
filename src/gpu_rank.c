#include "config.h"

#if defined(WITH_CUDA) || defined(WITH_HIP)

#include "error.h"
#include "gpu_mapping.h"
#include <stdio.h>

#ifdef WITH_MPI
#include <mpi.h>
#endif

int gpu_rank_bind(int global_rank) {

  int ndevices = 0;
  GPUError err = GPUGetDeviceCount(&ndevices);

  if (err != GPU_SUCCESS)
    error("Could not query GPU device count: %s", GPUGetErrorString(err));

  if (ndevices <= 0)
    error("No GPU devices visible to rank %d.", global_rank);

  int local_rank = 0;

#ifdef WITH_MPI
  MPI_Comm local_comm;
  int mpi_res = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                    MPI_INFO_NULL, &local_comm);

  if (mpi_res != MPI_SUCCESS)
    error("MPI_Comm_split_type failed while assigning GPUs.");

  mpi_res = MPI_Comm_rank(local_comm, &local_rank);

  if (mpi_res != MPI_SUCCESS)
    error("MPI_Comm_rank on local communicator failed while assigning GPUs.");

  MPI_Comm_free(&local_comm);
#endif

  const int device_id = local_rank % ndevices;

  err = GPUSetDevice(device_id);
  if (err != GPU_SUCCESS)
    error("Rank %d could not select GPU device %d: %s", global_rank, device_id,
          GPUGetErrorString(err));

  GPUDeviceProp prop;
  err = GPUGetDeviceProperties(&prop, device_id);
  if (err != GPU_SUCCESS)
    error("Rank %d could not read GPU device %d properties: %s", global_rank,
          device_id, GPUGetErrorString(err));

  printf("Rank %d bound to GPU %d/%d: %s\n",
       global_rank, device_id, ndevices, prop.name);
	fflush(stdout);

  return device_id;
}

#endif
