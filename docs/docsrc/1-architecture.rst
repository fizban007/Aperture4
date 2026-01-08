.. _architecture:

Architecture Overview
=======================

Aperture is built on a modular, component-based architecture that promotes flexibility and extensibility. Understanding this architecture is key to effectively using and extending the framework.

The framework is composed of three main concepts:

1. The Simulation Environment (`sim_environment`)
2. Systems (`system_t`)
3. Data Components (`data_t`)

---------------------------

The Simulation Environment
---------------------------

The `sim_environment` is the heart of the Aperture framework. It acts as a central coordinator that manages the entire simulation lifecycle. Its primary responsibilities include:

*   **System and Data Management:** It maintains a registry of all active `systems` and `data` components in the simulation.
*   **Simulation Loop:** It drives the main simulation loop, calling the `update()` method of each system at every time step.
*   **Configuration:** It handles command-line arguments and provides a centralized location for simulation parameters.
*   **MPI Environment:** It initializes and finalizes the MPI environment for parallel execution.

There is only one `sim_environment` instance per simulation.

--------

Systems
--------

A `system` is a module that performs a specific task within the simulation. For example, there are systems for:

*   Solving the electromagnetic fields (`field_solver`)
*   Updating particle positions and momenta (`ptc_updater`)
*   Handling domain decomposition for parallel simulations (`domain_comm`)
*   Writing simulation data to disk (`data_exporter`)

Each system is a class that inherits from the `system_t` base class. The `sim_environment` manages the creation and execution of these systems. Systems are initialized in the order they are registered, and their `update()` methods are called in that same order at every time step.

----------------

Data Components
----------------

`data` components are the data structures that `systems` operate on. They are the fundamental building blocks that hold the simulation state. Examples include:

*   The simulation grid (`grid_t`)
*   Particle data (e.g., positions, momenta, weights)
*   Electromagnetic fields

Like systems, data components are managed by the `sim_environment`. Systems can access data components through the environment, allowing them to interact and share information.

--------------------

How They Fit Together
--------------------

The typical workflow of an Aperture simulation is as follows:

1.  **Setup:** An application entry point (a `main()` function in one of the `problems/` directories) creates the `sim_environment`.
2.  **System Registration:** The `main()` function registers all the necessary `systems` for the specific simulation. This step configures the parameters of each system.
3.  **Data Registration:** As systems are registered, they in turn register the `data` components they require.
4.  **System Initialization:** The `sim_environment` calls the `init()` method of each registered system, allowing them to perform one-time setup tasks.
5.  **Initial Conditions:** After the systems are initialized, the `main()` function is responsible for setting the initial physical conditions of the simulation. This is typically done by getting direct access to the relevant `data` components (like particle and field data) and calling problem-specific functions to populate them with their initial values.
6.  **Execution:** The `sim_environment` enters the main simulation loop. In each step, it calls the `update()` method of each system in the order they were registered.
7.  **Finalization:** Once the simulation is complete, the `sim_environment` cleans up all resources.

This component-based design allows for great flexibility. By combining different systems, a wide variety of simulations can be constructed without modifying the core framework. The separation of system configuration from the setting of initial conditions allows for a clean and modular way to define new simulation problems.
