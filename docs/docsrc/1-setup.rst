=========================
 Setting up a Simulation
=========================

Here's a minimal example of setting up a PIC simulation in *Aperture*. This example shows the essential components needed for a basic 2D simulation:

.. code-block:: cpp

    #include "framework/config.h"
    #include "framework/environment.h"
    #include "systems/data_exporter.h"
    #include "systems/domain_comm.h"
    #include "systems/field_solver_cartesian.h"
    #include "systems/ptc_updater.h"

    using namespace Aperture;

    int main(int argc, char* argv[]) {
        // Define a 2D simulation configuration
        typedef Config<2> Conf;
        
        // Initialize the simulation environment
        auto &env = sim_environment::instance(&argc, &argv);
        
        // Set up the execution policy (automatically handles GPU/CPU)
        using exec_policy = exec_policy_dynamic<Conf>;

        // Create the domain communicator for parallel processing
        domain_comm<Conf, exec_policy_dynamic> comm;
        
        // Initialize the simulation grid
        grid_t<Conf> grid(comm);

        // Register the core simulation systems
        // 1. Particle pusher - handles particle motion
        auto pusher = env.register_system<
            ptc_updater<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid, &comm);
        
        // 2. Field solver - updates electromagnetic fields
        auto solver = env.register_system<
            field_solver<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid, &comm);
        
        // 3. Data exporter - handles output of simulation data
        auto exporter = env.register_system<data_exporter<Conf, exec_policy_dynamic>>(grid, &comm);

        // Initialize all registered systems
        env.init();

        // Set up initial conditions here
        // ... (see below for examples)

        // Start the simulation
        env.run();
    }

The above code sets up a basic 2D PIC simulation with three essential components:

1. **Particle Pusher**: Handles the motion of particles in the simulation
2. **Field Solver**: Updates the electromagnetic fields based on particle positions
3. **Data Exporter**: Manages the output of simulation data

The ``Config`` class is a template class that defines compile-time configurations for your simulation. The template parameter (2 in this example) specifies the dimensionality of the simulation. Other configurations include:
- Data type for floating-point numbers
- Particle pusher type
- Indexing scheme
- Other compile-time parameters

The simulation environment (``sim_environment``) manages the lifecycle of all systems. When you register a system using ``register_system<T>``, it:
1. Creates an instance of the system
2. Adds it to the system registry
3. Returns a pointer to the system for further configuration

Systems are executed in the order they are registered. In this example, each timestep will:
1. Update particle positions (pusher)
2. Update electromagnetic fields (solver)
3. Export data if needed (exporter)

There are two main ways to customize your simulation:

1. Through the configuration file (``config.toml``)
2. Programmatically in the source code

Config File
-----------

Every ``system`` has a number of parameters that can be customized through run
time parameters. All parameters are read from a configuration file in the
`toml <https://github.com/toml-lang/toml>`_ format. By default, the code will
look for a file named `config.toml` in the same directory as the executable. A
different config file can also be specified through a launch parameter:

.. code-block:: console

   $ ./aperture -c some/other/config/file.toml

To see all available parameters and their default values without running the simulation,
you can use the dry-run option:

.. code-block:: console

   $ ./aperture --dry-run

This will print out all parameters that can be configured, along with their current
values, making it easier to understand what can be customized in your simulation.

Parameters are stored in an instance of :ref:`params_store` in the
:ref:`sim_environment` class. One can also define all the required parameters
programmatically:

.. code-block:: cpp

   sim_env().params().add("dt", 0.01);
   sim_env().params().add("max_ptc_num", 100);

Since systems may use parameters in their constructors, one should add
whatever needed parameters before initializing any systems.

Source Code
-----------

Some things need to be specified in the source code and require a recompile,
e.g. non-trivial initial conditions. For example, one can assign an initial
function to some field:

.. code-block:: cpp

   vector_field<Conf> *B0;  // Declare a pointer to the background B
   env.get_data("B0", &B0); // Point it to the "B0" data component in the registry
   double Bp = 100.0;       // Set a characteristic value for B
   B0->set_value(0, [Bp](auto r, auto theta, auto phi) {
       return Bp / square(r);
   }); // Set the 0th component (B_r) to a monopole field in spherical coordinates

Nontrivial boundary conditions can be more difficult to set up, especially
time-dependent ones which requires the user to write a customized ``system``.
Please refer to :doc:`The Aperture Framework <2-framework>` for an explanation of how to write a custom ``system``.
