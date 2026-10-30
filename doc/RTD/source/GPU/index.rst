
.. GPU documentation
   Hydrodynamics GPU documentation originally by Mladen Ivkovic June 2025.
   Extended for integrated GPU gravity and hydrodynamics by Sarah Johnston October 2026.

.. _GPU_label:

GPU acceleration in SWIFT
=========================

SWIFT supports GPU offloading for selected gravitational and hydrodynamical interactions. The implementation uses CUDA for NVIDIA GPUs and HIP for AMD GPUs.

GPU acceleration is integrated with SWIFT's existing task-based scheduler. Rather than moving the whole simulation onto the GPU, selected computationally intensive workloads are collected into batches and executed asynchronously on GPU devices.

The two main GPU workloads are:

* **Gravity:** Particle-particle gravitational interactions, including self and pair interactions between cells.
* **Hydrodynamics:** SPHENIX density, gradient and force interactions, including recursive self and pair tasks.

Both implementations share GPU device management, runner-level initialisation and cleanup, and GPU offloading infrastructure. However, they use different particle representations, batching parameters and computational kernels.

.. warning::

   GPU acceleration is under active development. The gravity and hydrodynamics implementations should be validated independently against CPU results before use in production simulations.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   gpu_general
   compilation
   running
   gravity


