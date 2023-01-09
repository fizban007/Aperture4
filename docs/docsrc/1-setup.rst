=========================
 Setting up a Simulation
=========================

The following code is the boiler plate for setting up a PIC simulation in *Aperture*:

.. code-block:: cpp

    #include "framework/config.h"
    #include "framework/environment.h"
    #include "systems/data_exporter.h"
    #include "systems/domain_comm.h"
    #include "systems/field_solver_cartesian.h"
    #include "systems/ptc_updater.h"

    using namespace Aperture;

    int main(int argc, char* argv[]) {
        typedef Config<2> Conf; // Specify that this is a 2D simulation
        // Initialize the simulation environment
        auto &env = sim_environment::instance(&argc, &argv);
        // Choose execution policy depending on compile options (GPU or CPU)
        using exec_policy = exec_policy_dynamic<Conf>;

        // Setting up domain decomposition
        domain_comm<Conf, exec_policy_dynamic> comm;
        // Setting up the simulation grid
        grid_t<Conf> grid(comm);
        // Add a particle pusher
        auto pusher = env.register_system<
          ptc_updater<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
          &comm);
        // Add a field solver
        auto solver = env.register_system<
          field_solver<Conf, exec_policy_dynamic, coord_policy_cartesian>>(grid,
          &comm);
        // Setup data output
        auto exporter = env.register_system<data_exporter<Conf,
          exec_policy_dynamic>>(grid, &comm);

        // Call the init() method of all systems
        env.init();

        // Prepare initial conditions
        ...

        // Enter the main simulation loop
        env.run();
    }

The ``Config`` class contains compile-time configurations for the code,
including the dimensionality (2 in the above example), data type for floating
point numbers, particle pusher type, and indexing scheme.

The method ``register_system<T>`` constructs a :doc:`/api/framework/system`, puts it in the registry, and returns a pointer to it. When ``init()`` or ``run()`` is
called, all the ``init()`` and ``run()`` methods of the registered systems are
run in the order they are registered. In the above example, at every timestep,
the code will first call the ``ptc_updater``, then the ``field_solver``, then
the ``data_exporter``.

There are two main ways to customize the problem setup, namely through the
config file ``config.toml``, or programmatically through coding initial or
boundary conditions.

Config File
-----------

Every ``system`` has a number of parameters that can be customized through run
time parameters. All parameters are read from a configuration file in the
`toml <https://github.com/toml-lang/toml>`_ format. By default, the code will
look for a file named `config.toml` in the same directory as the executable. A
different config file can also be specified through a launch parameter:

.. code-block:: console

   $ ./aperture -c some/other/config/file.toml

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
