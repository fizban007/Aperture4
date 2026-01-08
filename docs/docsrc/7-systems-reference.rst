==================
Systems Reference
==================

The *Aperture* framework implements a sophisticated collection of systems that handle different aspects of particle-in-cell simulations. Each system is designed using policy-based templates that allow flexible combinations of coordinate systems, execution targets (CPU/GPU), and physics models.

This section provides detailed documentation for each major system category, explaining their implementation, usage, and performance characteristics.

.. toctree::
   :maxdepth: 2
   :caption: Core Systems

   7.1-field-solvers
   7.2-particle-systems
   7.3-radiative-transfer
   7.4-grid-systems
   ..
      7.5-parallel-computing
      7.6-data-io
      7.7-physics-modules

System Architecture Overview
============================

All systems in *Aperture* inherit from the ``system_t`` base class and follow a common lifecycle:

1. **Initialization** - Systems register with the simulation environment and set up internal data structures
2. **Update Loop** - Systems are called in sequence during each simulation timestep
3. **Finalization** - Systems clean up resources and write final outputs

The key architectural principles are:

**Policy-Based Design**
  Systems use template policies to abstract coordinate systems, execution targets, and physics models. This allows the same algorithmic code to work across different coordinate systems (Cartesian, spherical, general relativistic) and execution targets (CPU, GPU, OpenMP).

**Data Dependency Management**
  Systems declare their data dependencies, allowing the framework to ensure proper initialization order and efficient memory management.

**Modular Physics**
  Complex physics processes are decomposed into composable policies that can be mixed and matched for different simulation scenarios.

Template Structure
==================

Most systems follow this template pattern::

    template <class Conf, class ExecPolicy, class CoordPolicy, class PhysicsPolicy = empty_physics_policy>
    class system_name : public system_t {
        // System implementation
    };

Where:

- ``Conf`` - Configuration class defining simulation parameters and data types
- ``ExecPolicy`` - Execution policy (``exec_policy_host``, ``exec_policy_gpu``, etc.)
- ``CoordPolicy`` - Coordinate system policy (``coord_policy_cartesian``, ``coord_policy_spherical``, etc.)
- ``PhysicsPolicy`` - Optional physics-specific behavior (radiation, cooling, etc.)

This design enables the same algorithmic implementation to work across:

- **Coordinate systems**: Cartesian, spherical, cylindrical, general relativistic
- **Execution targets**: Single-threaded CPU, OpenMP CPU, CUDA GPU, HIP GPU
- **Physics models**: Basic electrodynamics, radiation, cooling, gravity, etc.
