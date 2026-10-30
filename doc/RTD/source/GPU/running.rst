.. GPU runtime configuration
   Originally by Mladen Ivkovic, June 2025
   Extended by Sarah Johnston, October 2026

.. _gpu_running_label:

Running SWIFT with GPU acceleration
===================================

Introduction
------------

GPU-enabled SWIFT simulations use the normal SWIFT parameter files and command-line options.

Additional GPU parameters control the packing of gravity and hydrodynamics interactions, GPU stream allocation and memory usage.

GPU configuration
-----------------

GPU parameters are grouped in the ``GPU`` section of the YAML parameter file.

The following example shows a combined gravity and hydrodynamics configuration.

.. code-block:: yaml

   GPU:
     # Gravity parameters
     ncells_per_gpu_grav_pack: 1024
     gpu_grav_cell_size: 1000
     nstreams: 1

     # Hydrodynamics parameters
     hydro_pack_size: 10000
     hydro_bundle_size: 1000
     hydro_tester_param: 0
     hydro_recursion_max_depth: 4
     hydro_part_buffer_size: -1

The values above are illustrative and may need adjustment depending on the problem size, GPU architecture and available memory.

.. important::

   The unified GPU configuration requires the hydrodynamics parameter lookups in ``runner_gpu_hydro_set_params()`` to use ``GPU:`` rather than ``Scheduler:``. Other versions of the hydro branch read hydrodynamics settings from ``Scheduler:`` and use ``gpu_pack_size`` for both sizes.

Gravity parameters
------------------

ncells_per_gpu_grav_pack
~~~~~~~~~~~~~~~~~~~~~~~

Controls the requested number of cell slots collected into a GPU gravity batch.

Larger batches can reduce the relative overhead associated with data transfers and kernel launches, but require more GPU memory.

Smaller batches may reduce memory requirements and improve responsiveness to scheduler tasks.

The implementation can restrict the effective batch size to satisfy memory constraints.

The parameter is optional, with ``-1`` requesting automatic batch-size selection.

gpu_grav_cell_size
~~~~~~~~~~~~~~~~~~

Controls the gravity cell splitting threshold used by the GPU-enabled gravity implementation.

This affects the sizes of the cells passed to the GPU gravity kernels and can affect the balance between particle-particle and other gravitational interactions.

Changing the cell splitting threshold can affect both performance and numerical accuracy.


nstreams
~~~~~~~~

Requests the number of GPU streams used for gravity offloading per runner.

Streams allow GPU transfers and computations to be organised independently.

Additional streams may improve concurrency, but also require resources and may increase memory usage.

The default requested value is ``1``.

This parameter controls gravity streams and is independent of the hydrodynamics bundle count.

Hydrodynamics parameters
------------------------

hydro_pack_size
~~~~~~~~~~~~~

The number of leaf computations we want to pack before offloading. This usually tweaked for each test case and GPU but a good estimate for large sims (>10M particles per GPU is 2048). Leaf computations are SPH interactions between two cells at the finest level we can recurse to (See Nasar et al. 2026 for details)

hydro_bundle_size
~~~~~~~~~~~~~~~

For the current approach we use one bundle per CPU thread instead of splitting a full pack of work into smaller bundles. We currently sort cells before offloading to ensure we do not pack the same leaf level cell more than once so splitting a pack into streams is no longer necessary and is now too complicated to be worth the effort. With current code always set bundle size to be equal to pack size

hydro_tester_param
~~~~~~~~~~~~~~~~

This is used to define at which ratio of ci_count/cj_count where ci_count is the number of particles in ci do we choose to use the source target path in the kernel function. For cases where cj has O(1000) particles while ci has O(1-10) particles we do not have enough particles to parallelise the kernel over ci_count so we parallelise over cj_count and do somne reduction and atomic magic to scatter contributions from cj particles onto the particles in ci. Testing has shown this is best set to 8-16 for Evrard collapse type test cases with large variations in smoothing length

hydro_recursion_max_depth
~~~~~~~~~~~~~~~~~~~~~~~
What is the max depth in the tree do we expect to recurse to for all the cells in our simulation? This is used to approximate how big we need to make the CPU-GPU buffers when sizing the buffers up

hydro_part_buffer_size
~~~~~~~~~~~~~~~~~~~~

hydro_part_buffer_size is not used in the latest version of the code but listed here for completeness. This was used to define the size of the CPU-GPU buffer which is used to send and recieve particle data to and from the GPU. We now split the available GPU memory between CPU host threads evenly and throw an error message if we somehow end up trying to pack more particles into the buffer than it can fit


Memory considerations
---------------------

GPU memory is shared between particle buffers, interaction metadata, transfer buffers and other device allocations.

The gravity and hydrodynamics implementations maintain separate workload buffers but share the GPU device memory resources.

Increasing batch sizes, bundle counts or stream counts may increase memory consumption.

The available memory may also be affected by multiple MPI ranks sharing one device.

Parameter tuning should therefore consider the combined memory requirements of both workloads.

Running simulations
-------------------

After compiling SWIFT with the selected GPU backend, run the appropriate GPU executable with a normal simulation parameter file.

For example, using a CUDA build:

.. code-block:: bash

   ./swift -G -s parameters.yml

The hydrodynamics and gravity simulation options should be chosen to match the physical problem being run.

A GPU build does not necessarily imply that every computational task will be offloaded. Tasks not supported by the GPU implementation continue to use the available CPU paths.

Validation and performance
--------------------------

Before using GPU acceleration for production simulations, test a small problem and check:

* Correct GPU device selection.
* Reported packing and stream parameters.
* Successful device memory allocation.
* Completion of GPU tasks without errors.
* Consistency between CPU and GPU results.
* Simulation-level runtime performance.


