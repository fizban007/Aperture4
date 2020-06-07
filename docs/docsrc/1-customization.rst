=========================
 Setting up a Simulation
=========================

The following code sets up a simple PIC simulation:

.. code-block:: cpp

    using namespace Aperture;

    int main(int argc, char* argv[]) {
        typedef Config<2> Conf; // Specify that this is a 2D simulation
        sim_environment env(&argc, &argv); // Initialize the coordinator

        // Setup the simulation grid
        auto grid = env.register_system<grid_t<Conf>>(env);
        // Add a particle pusher
        auto pusher = env.register_system<ptc_updater<Conf>>(env, *grid);
        // Add a field solver
        auto solver = env.register_system<field_solver<Conf>>(env, *grid);
        // Setup data output
        auto exporter = env.register_system<data_exporter<Conf>>(env, *grid);

        // Call the init() method of all systems
        env.init();

        // Enter the main simulation loop
        env.run();
    }

The ``Config`` class contains compile-time configurations for the code,
including the dimensionality (2 in the above example), data type for floating point
numbers, and indexing scheme.

The method ``register_system<T>`` constructs a ``system``, put it in the
registry, and returns a pointer to it. When ``init()`` or ``run()`` is called,
all the ``init()`` and ``run()`` methods of the registered systems are run in
the order they are registered. In the above example, at every timestep, the code
will first call the ``ptc_updater``, then the ``field_solver``, then the
``data_exporter``.

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

   sim_environment env;
   env.params().add("dt", 0.01);
   env.params().add("max_ptc_num", int64_t(100));

Note that the parameters can only be one of these types:

* ``bool``
* ``int64_t``
* ``double``
* ``std::string``
* ``std::vector<bool>``
* ``std::vector<int64_t>``
* ``std::vector<double>``
* ``std::vector<std::string>``

When there is type ambiguity, an explicit cast is required when add a parameter
this way. Since systems may use parameters in their constructors, one should add
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
   B0->set_value(0, [Bp](Scalar r, Scalar theta, Scalar phi) {
       return Bp / square(r);
   }); // Set the 0th component (B_r) to a monopolar field

Nontrivial boundary conditions can be more difficult to set up, especially
time-dependent ones which requires the user to write a customized ``system``.
Please refer to the tutorial for how to do that.
