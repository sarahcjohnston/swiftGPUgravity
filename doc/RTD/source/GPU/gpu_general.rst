.. GPU execution
   Originally by Mladen Ivkovic, June 2025
   Extended by Sarah Johnston, October 2026

.. _gpu_general_label:


How GPU offloading works
========================

Introduction
------------

SWIFT is a task-based simulation code in which particles are organised into a hierarchy of spatial cells. The scheduler distributes computational tasks between CPU threads. GPU acceleration allows selected interactions to be performed on GPUs while retaining the existing SWIFT scheduling framework.

As individual SWIFT tasks may contain relatively few particles, offloading each task independently would result in excessive GPU kernel launch and data-transfer overheads. Instead, multiple tasks or cell interactions are collected into larger batches before being transferred to the GPU.

Shared infrastructure
---------------------

The gravity and hydrodynamics implementations share GPU management functionality.

GPU state is associated with SWIFT's runner structures, allowing individual CPU runner threads to manage GPU work.

The shared infrastructure is responsible for:
* Selecting and initialising GPU devices.
* Allocating GPU memory and managing memory budgets.
* Creating and managing GPU streams.
* Packing particle data into GPU-compatible buffers.
* Transferring data between host and device memory.
* Launching computational kernels.
* Synchronising GPU work and unpacking results.
* Marking completed tasks in the SWIFT scheduler.
* Releasing GPU resources during cleanup.

The main implementation is contained in ``src/runner_gpu.c`` and ``src/runner_gpu.h``.

Hydrodynamics offloading
-----------------------
The hydrodynamics implementation targets SPHENIX interactions.

The supported interaction stages are:
* Density calculations.
* Gradient calculations.
* Hydrodynamical force calculations.

Each stage includes interactions within an individual cell (self interactions) and between neighbouring cells (pair interactions).

GPU hydrodynamics recursively identifies leaf-cell interactions that can be packed together.

These interactions are collected into a pack, which is divided into smaller bundles for GPU execution.

The size of a complete pack is controlled by ``GPU:hydro_pack_size``.

The bundle size is controlled independently by ``GPU:hydro_bundle_size``.

The number of bundles is calculated as:
.. math::

   N_{\rm bundles}
   =
   \left\lceil
   \frac{N_{\rm pack}}{N_{\rm bundle}}
   \right\rceil.

For example, a pack size of 10000 and a bundle size of 1000 produces 10 bundles.

This separation allows the total number of collected interactions and the amount of work launched together to be tuned independently.

Gravity offloading
------------------

The GPU gravity implementation accelerates short-range particle-particle (P-P) gravitational interactions.

Two interaction types are supported:
* Self interactions: gravitational forces between particles belonging to the same cell.
* Pair interactions: gravitational forces between particles belonging to two different cells.

The number of cell slots requested per batch is controlled by ``GPU:ncells_per_gpu_grav_pack``.

Each batch involves packing particle positions, masses, softening information and interaction metadata before launching the relevant GPU kernel.

Results are copied back to host memory and accumulated into the SWIFT particle data.

GPU streams allow transfers and kernel execution to be managed independently for multiple batches, subject to the memory and scheduling constraints.

The number of gravity streams is controlled by ``GPU:nstreams``. If not specified the deafault value is 1 stream per CPU runner thread.

The two workloads have independent batching parameters. Changing the gravity batch size does not automatically change the hydrodynamics pack or bundle sizes.

Task scheduling
---------------

GPU execution must remain consistent with SWIFT's task dependencies. Tasks selected for GPU execution are tracked until the associated GPU computations have completed and their results have been returned to the host. The runner is then responsible for notifying the scheduler that these tasks are complete, allowing dependent tasks to proceed. This ensures that GPU offloading can be incorporated into the existing task graph without requiring an entirely separate GPU scheduling system.

