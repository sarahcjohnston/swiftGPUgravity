.. GPU gravity documentation
   Written by Sarah Johnston, October 2026

.. _gpu_gravity_label:

GPU gravity offloading
======================

Overview
--------

The GPU gravity implementation accelerates the short-range particle-particle (PP) gravitational interactions performed by SWIFT.

SWIFT normally evaluates gravity using a combination of direct particle interactions, multipole approximations and, where applicable, long-range mesh-based methods.

The GPU implementation targets the computationally intensive particle-particle interactions.

The remaining gravitational calculations continue to use the appropriate CPU implementations.

The offloading implementation is integrated with SWIFT's existing task-based scheduler, allowing GPU work to be batched without replacing the overall simulation framework.

Supported interactions
----------------------

Two types of gravitational interactions are offloaded to the GPU:

Self interactions
~~~~~~~~~~~~~~~~~

Self interactions calculate gravitational forces between particles belonging to the same cell.

For a cell containing N particles, the kernel evaluates the required particle-particle interactions within that cell.

The GPU implementation uses tiled interactions to organise particle data and computation.

Pair interactions
~~~~~~~~~~~~~~~~~

Pair interactions calculate gravitational forces between particles belonging to two different cells.

The GPU implementation evaluates interactions between the particles of both cells, updating the appropriate particle accelerations.

Pair interactions use a separate GPU kernel and packing strategy.

The packing implementation can avoid duplicating particle data when cells appear in multiple pair interactions within the same batch.

Data representation
-------------------

Before a GPU kernel is launched, the necessary particle properties are transferred from SWIFT's host-side particle structures into GPU-compatible arrays.

These include:
* Particle positions.
* Particle masses.
* Gravitational softening lengths.
* Active-particle and cell metadata.
* Interaction and packing information.

Positions and masses are packed using ``float4`` structures, while softening information is stored separately.

The GPU kernels calculate gravitational accelerations and associated quantities using the packed particle information.

The results are transferred to host memory and accumulated into the corresponding SWIFT particle data.

Gravity batching
----------------

Individual SWIFT gravity tasks may contain too little work to efficiently occupy a GPU.

The implementation therefore collects multiple cell interactions into batches.

Batching reduces the overhead of repeatedly launching kernels and transferring relatively small amounts of particle data.

The requested gravity batch size is controlled by:

``GPU:ncells_per_gpu_grav_pack``

This specifies the number of cell slots available in a GPU gravity batch.

The actual batch capacity may be limited by available GPU memory and the requirements of individual interaction types.

Pair interactions require space for both participating cells and their metadata.

The gravity implementation supports both explicitly configured and automatically selected batch sizes.

GPU streams
-----------

The gravity implementation uses GPU streams to manage device operations.

Streams allow data transfers and kernel execution to be organised asynchronously.

The requested number of streams is controlled by ``GPU:nstreams``.

GPU streams are managed through runner-level GPU state.

Gravity and hydrodynamics use separate stream management within the integrated implementation.

The most effective stream count depends on the GPU architecture, batch sizes and available memory.

Memory management
-----------------

GPU gravity offloading requires memory for:

* Packed particle positions and masses.
* Gravitational softening values.
* Returned particle accelerations.
* Cell and interaction metadata.
* Self and pair interaction buffers.
* Stream-specific working data.

GPU memory requirements depend on the number and size of the cells in each batch.

The implementation estimates GPU memory usage and limits allocations according to the available device resources.

In the integrated version, gravity and hydrodynamics allocations must share the same device memory budget.

Gravity cell splitting
----------------------

The ``GPU:gpu_grav_cell_size`` parameter controls the gravity cell splitting threshold used in GPU-enabled builds.

The cell splitting threshold influences the sizes of gravity cells and the number of particle-particle interactions that must be evaluated. It can also influence the balance between direct particle interactions and other gravity approximations. Changing this parameter can therefore affect both simulation performance and accuracy. Accuracy comparisons should use consistent cell splitting parameters for CPU and GPU simulations wherever possible.

Scheduler integration
---------------------

GPU gravity uses the existing SWIFT task scheduler.

When suitable gravity tasks are encountered, they can be collected into GPU batches rather than executed immediately on the CPU. The implementation maintains information about the tasks associated with each batch. After a batch has completed, its results are returned to the host and accumulated into the particle data. The corresponding tasks are then marked as completed, allowing dependent scheduler tasks to proceed.

This is necessary to preserve SWIFT's existing task dependencies while enabling asynchronous GPU execution.

CUDA and HIP
------------

The gravity implementation supports both NVIDIA CUDA and AMD HIP backends. The CUDA and HIP implementations use a common GPU offloading abstraction where possible, allowing much of the host-side batching and scheduling code to be shared. Backend-specific functionality is used for device allocation, memory transfers, streams, kernel launches and synchronisation.

The supported GPU architectures depend on the installed toolkit and compiler.

MPI execution
-------------

GPU gravity can be used with SWIFT's MPI-enabled execution. GPU devices are assigned to MPI ranks according to the device selection logic in the runner GPU implementation. Multiple MPI ranks may share a GPU, subject to the device memory resources and the configuration of the system. Each MPI rank maintains its own SWIFT state and GPU offloading resources.

GPU memory usage must therefore be considered when running multiple ranks on the same device.

Performance scaling should be tested on the intended hardware and MPI configuration.

Timing and profiling
--------------------

The GPU gravity implementation includes instrumentation for measuring different stages of offloading.

When GPU timing instrumentation is enabled, individual gravity batches can record:
* Task type (self or pair).
* Simulation step.
* Runner and GPU stream.
* Number of packed cells.
* Number of particles.
* Host-to-device transfer size.
* Packing time.
* Host-to-device transfer time.
* Kernel execution time.
* Device-to-host transfer time.
* Unpacking time.

To enable these timings are recorded the code must be compiled with ``--enable-gpu-timing``

These timings help identify bottlenecks associated with packing, transfers, kernel execution and unpacking.

However, the timings of individual GPU operations should not simply be added together to estimate the total simulation runtime, as operations may overlap.

Numerical validation
--------------------

Correctness validation is essential when offloading gravitational calculations from the CPU to GPU devices.

The primary validation method is to compare CPU and GPU gravitational forces at identical particle positions.

The following simulation properties should be consistent:

* Particle IDs and positions.
* Particle masses.
* Gravitational softening lengths.
* Simulation box size.
* Boundary conditions.
* Gravity configuration.
* Cell splitting parameters.

Force comparisons
~~~~~~~~~~~~~~~~~

CPU and GPU gravitational accelerations should be compared before particles are advanced by additional integration steps. For N matched particles, the absolute root-mean-square vector force difference can be calculated as:

.. math::

   \mathrm{RMS}_{\rm abs}
   =
   \sqrt{
   \frac{1}{N}
   \sum_{i=1}^{N}
   \left|
   \mathbf{a}_{i,\rm GPU}
   -
   \mathbf{a}_{i,\rm CPU}
   \right|^2
   }.

The force checks can be activated by configuring with SWIFT's inbuilt ``--enable-gravity-force-checks=N`` option where every N particles is sampled for their force.

Cell splitting and accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gravity cell splitting threshold affects the division of work between particle-particle interactions and other gravity approximations.

CPU and GPU comparisons should therefore be repeated at multiple cell splitting thresholds.

A difference between CPU and GPU simulations does not necessarily imply an error in the GPU kernel if the underlying interaction selection differs. It is important to distinguish differences caused by cell splitting from differences in the actual particle-particle force calculations. Small numerical differences between CPU and GPU calculations may grow over time as particle trajectories diverge. Consequently, differences between late-time particle positions are not sufficient on their own to establish whether the GPU gravity kernels are correct. Direct comparisons of forces at matched positions provide a more useful test of kernel correctness.

Performance considerations
--------------------------
GPU gravity performance depends on several interacting factors:

* The number of particle interactions.
* Gravity cell sizes.
* Batch sizes.
* GPU memory capacity.
* Kernel execution efficiency.
* Host-to-device transfer overheads.
* GPU stream concurrency.
* CPU scheduler overheads.
* MPI configuration and device sharing.

Larger batches can reduce offloading overheads but may increase memory use and delay task completion.

Performance should be evaluated using complete simulation timesteps as well as individual GPU kernel timings.

Implementation files
--------------------

The main implementation files include:

``src/runner_gpu.c``
   GPU runner initialisation, gravity 
   batching, memory management, data
   transfers, and task completion.

``src/runner_gpu.h``
   GPU runner data structures and state.
   
``src/gpu_gravity_functions.cuh``
   The header file with additional funcions
   used for force calculations.

``src/runner_main.c``
   Task execution and GPU offloading
   integration with the SWIFT runner.

``src/space.c``
   Gravity cell splitting configuration.



