.. GPU compilation
   Originally by Mladen Ivkovic, June 2025
   Extended by Sarah Johnston, October 2026

.. _gpu_compilation_label:


.. GPU compilation
   Based on hydrodynamics documentation by Mladen Ivkovic.
   Extended for GPU gravity and HIP support.

.. _gpu_compilation_label:

Compiling SWIFT with GPU acceleration
====================================

Overview
--------

SWIFT supports GPU acceleration through two backends:
* CUDA, for NVIDIA GPUs.
* HIP, for AMD GPUs.

The GPU backends are selected when configuring SWIFT. CUDA and HIP are mutually exclusive within a single build. If you select both, the code will error.

GPU gravity and hydrodynamics use the selected backend and share parts of the GPU runtime infrastructure.

.. warning::

   GPU offloading is under active development. Successful compilation does not guarantee that every hydrodynamics or gravity kernel is supported and numerically validated on the selected device.

CUDA
----

Configuration
~~~~~~~~~~~~~

To enable CUDA support, configure SWIFT with:

.. code-block:: bash

   ./configure --with-cuda

If CUDA is installed in a non-standard location, specify the installation directory:

.. code-block:: bash

   ./configure --with-cuda=/path/to/cuda

A recommended configuration for the gravity GPU branch is:

.. code-block:: bash

   ./configure --with-cuda=/path/to/cuda \
       --enable-ipo \
       --with-tbbmalloc \
       --disable-hand-vec

   make -j

The relevant CUDA toolkit, C/C++ compilers and required SWIFT dependencies must be available in the build environment.

CUDA compiler options
~~~~~~~~~~~~~~~~~~~~~

Additional CUDA compiler flags can be supplied using:

.. code-block:: bash

   ./configure --with-cuda \
       --with-cuda-flags="..."

Alternatively, compiler and linker environment variables
can be used where supported by the build system:

* ``CUDA_CFLAGS``: CUDA-related host compiler flags.
* ``CUDA_LDFLAGS``: CUDA-related host linker flags.
* ``CUDA_LIBS``: Additional CUDA host libraries.
* ``NVCC_FLAGS``: Flags passed to the NVIDIA CUDA compiler.
* ``NVCC_LDFLAGS``: Device linker options.
* ``NVCC_LIBS``: Additional device libraries.

For example:

.. code-block:: bash

   NVCC_FLAGS="-allow-unsupported-compiler" \
       ./configure --with-cuda

These flags may be necessary when working with compiler versions not officially supported by the CUDA toolkit.

GPU architecture
~~~~~~~~~~~~~~~~

The CUDA architecture can be selected using:

.. code-block:: bash

   ./configure --with-cuda \
       --with-cuda-arch=sm_90

The architecture must correspond to the intended GPU and be supported by the CUDA compiler version. This information can be found in the NVIDIA documentation for the device or online in the CUDA programming guide.

The ``native`` architecture setting may be used where supported when compiling on a node containing the target GPU.

For available CUDA architectures, consult the NVIDIA CUDA documentation.


HIP
---

Configuration
~~~~~~~~~~~~~

HIP is supported as an alternative GPU backend.

Configure with:

.. code-block:: bash

   ./configure --with-hip=/path/to/rocm

For example:

.. code-block:: bash

   ./configure --with-hip=/path/to/rocm \
       --with-hip-flags="--offload-arch=gfxXXX \
       -fgpu-default-stream=per-thread \
       -munsafe-fp-atomics" \
       --enable-ipo \
       --with-tbbmalloc \
       --enable-compiler-warnings

   make -j

Replace ``gfxXXX`` with the appropriate GPU architecture, as supported by the installed ROCm compiler.

The flags shown are those recommended for the GPU gravity development branch. Their suitability should be checked against the target ROCm and GPU versions.

HIP compiler options
~~~~~~~~~~~~~~~~~~~~

HIP-specific compiler settings include:
* ``HIP_CFLAGS``: Additional host compilation flags.
* ``HIPCC_FLAGS``: Additional flags for ``hipcc``.
* ``--with-hip-flags``: Additional HIP flags supplied
  directly during configuration.


Portability considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~
Although the GPU gravity implementation supports both CUDA and HIP compilation, differences between GPU devices and compilers can affect:
* Supported GPU architectures.
* Device vector types and intrinsics.
* Atomic operations.
* Memory allocation and availability.
* Kernel launch configurations.
* Floating-point numerical behaviour.
* Synchronisation and error handling.


Build system
------------
GPU support introduces both host-side and device-side code into the SWIFT build.

Host-side GPU code manages device allocation, packing, task scheduling and transfers.

Device-side code implements the computational kernels launched on the GPU.

The code is written in vendor agnostic language e.g. ``GPUMalloc`` instead of ``cudaMalloc`` or ``hipMalloc`` and then a macro is used with the configuration option to make the code for CUDA or HIP. Mappings can be found in ``gpu_mapping.h``.

The CUDA build uses additional libraries for GPU host and device code. GPU device objects require compilation and linking with the CUDA toolchain.

When adding GPU source files, update ``src/Makefile.am`` so they are included in the appropriate host or device build targets.

CUDA device implementations generally use ``.cu`` files, with associated ``.cuh`` headers, while host-side implementations use C source and header files.

Code specific to a GPU backend should be protected by the appropriate compilation macros, including ``WITH_CUDA`` and ``WITH_HIP`` where applicable.

Care should be taken to preserve compatibility with CPU-only SWIFT builds.

Compilation on HPC systems
--------------------------
When compiling on an HPC system such as COSMA, load a consistent set of compiler, MPI, GPU toolkit and numerical library modules before configuration.

Typical dependencies include the usual SWIFT dependencies plus a CUDA or ROCm library:
* A supported C/C++ compiler.
* CUDA or ROCm.
* MPI, if building the MPI-enabled executable.
* HDF5.
* FFTW.
* METIS and ParMETIS, as required.
* Other libraries required by the SWIFT configuration.

The precise module names and versions depend on the available software environment.

The host compiler must also be compatible with the selected GPU compiler.

Troubleshooting
---------------

Missing C++ runtime symbols
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Undefined references to C++ runtime functions can indicate that the C++ standard library has not been linked correctly.

Depending on the compiler environment, adding the C++ standard library to the linker options may help:

.. code-block:: bash

   LIBS="-lstdc++" ./configure --with-cuda

AddressSanitizer
~~~~~~~~~~~~~~~

AddressSanitizer can help identify host-side memory errors, including incorrect allocation sizes and out-of-bounds accesses in GPU packing code.

However, CUDA and host sanitizers may interact unpredictably.

Some development environments may require:

.. code-block:: bash

   ASAN_OPTIONS=protect_shadow_gap=0 ./swift ...

This is a development workaround and should not replace proper memory validation.

Runtime GPU errors
~~~~~~~~~~~~~~~~~~

A successful compilation does not rule out runtime problems such as:
* Insufficient device memory.
* Invalid kernel launch dimensions.
* Unsupported GPU instructions.
* Device-side memory errors.
* Incorrectly sized packing buffers.
* Incomplete GPU task synchronisation.

When debugging, start with a small simulation, enable the available diagnostics and validate the GPU results against a CPU reference.

You can also use tools like nvidia-smi and rocm-smi to monitor jobs while they are running and for CUDA can us nvtop to see the utilisation of the GPU live during runs.



