#ifndef SWIFT_GPU_MAPPING_H
#define SWIFT_GPU_MAPPING_H

#include <config.h>
#include <stddef.h>

#if defined(WITH_CUDA) && defined(WITH_HIP)
#error "WITH_CUDA and WITH_HIP are both defined"
#endif

#if !defined(WITH_CUDA) && !defined(WITH_HIP)
#error "WITH_CUDA or WITH_HIP must be defined"
#endif

#if defined(WITH_CUDA)
#ifdef __cplusplus
#include <cuda_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

typedef cudaStream_t GPUStream;
typedef cudaEvent_t GPUEvent;
typedef cudaError_t GPUError;
typedef struct cudaDeviceProp GPUDeviceProp;

#define GPU_SUCCESS cudaSuccess

#define GPU_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define GPU_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define GPU_MEMCPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice

#define GPUGetLastError cudaGetLastError
#define GPUGetPeekAtLastError cudaPeekAtLastError
#define GPUGetErrorString cudaGetErrorString

#define GPUStreamCreate cudaStreamCreate
#define GPUStreamDestroy cudaStreamDestroy
#define GPUStreamSynchronize cudaStreamSynchronize

#define GPUEventCreate cudaEventCreate
#define GPUEventRecord cudaEventRecord

#define GPUMalloc cudaMalloc
#define GPUFree cudaFree
#define GPUHostMalloc gpu_host_malloc
#define GPUFreeHost cudaFreeHost

#define GPUMemcpyAsync cudaMemcpyAsync

#define GPUSetDevice cudaSetDevice
#define GPUGetDeviceProperties cudaGetDeviceProperties

/**
 * @brief Allocate pinned host memory for GPU transfers.
 *
 * @param ptr Output pointer to the allocated memory.
 * @param size Number of bytes to allocate.
 */
static inline GPUError gpu_host_malloc(void** ptr, size_t size) {
  return cudaMallocHost(ptr, size);
}
#endif

#if defined(WITH_HIP)
#ifdef __cplusplus
#include <hip/hip_runtime.h>
#else
#include <hip/hip_runtime_api.h>
#endif

typedef hipStream_t GPUStream;
typedef hipEvent_t GPUEvent;
typedef hipError_t GPUError;
typedef hipDeviceProp_t GPUDeviceProp;

#define GPU_SUCCESS hipSuccess

#define GPU_MEMCPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define GPU_MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define GPU_MEMCPY_DEVICE_TO_DEVICE hipMemcpyDeviceToDevice

#define GPUGetLastError hipGetLastError
#define GPUGetPeekAtLastError hipPeekAtLastError
#define GPUGetErrorString hipGetErrorString

#define GPUStreamCreate hipStreamCreate
#define GPUStreamDestroy hipStreamDestroy
#define GPUStreamSynchronize hipStreamSynchronize

#define GPUEventCreate hipEventCreate
#define GPUEventRecord hipEventRecord

#define GPUMalloc hipMalloc
#define GPUFree hipFree
#define GPUHostMalloc gpu_host_malloc
#define GPUFreeHost hipFreeHost

#define GPUMemcpyAsync hipMemcpyAsync

#define GPUSetDevice hipSetDevice
#define GPUGetDeviceProperties hipGetDeviceProperties

/**
 * @brief Allocate pinned host memory for GPU transfers.
 *
 * @param ptr Output pointer to the allocated memory.
 * @param size Number of bytes to allocate.
 */
static inline GPUError gpu_host_malloc(void** ptr, size_t size) {
  return hipHostMalloc(ptr, size, 0);
}
#endif

#endif
