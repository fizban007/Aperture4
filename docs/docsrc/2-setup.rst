=======================
 Setting Up a Problem
=======================

This guide provides a step-by-step walkthrough for creating a new simulation problem in Aperture. We will use the existing ``reconnection`` problem as a reference.

A new problem is essentially a new executable that links against the core ``Aperture`` library. It defines the specific systems, parameters, and initial conditions for a simulation run.

Step 1: Create the Directory Structure
---------------------------------------

First, create a new directory for your problem inside the ``problems/`` directory. For this example, we'll call it ``my_new_problem``.

.. code-block:: console

   $ cd /path/to/Aperture4
   $ mkdir problems/my_new_problem
   $ mkdir problems/my_new_problem/src

All source code for your problem will reside in the ``problems/my_new_problem/src/`` directory.

Step 2: Create the CMakeLists.txt File
---------------------------------------

Each problem needs its own ``CMakeLists.txt`` file to tell the build system how to compile it. Create the file ``problems/my_new_problem/CMakeLists.txt`` with the following content:

.. code-block:: cmake

   add_aperture_executable(my_new_problem src/main.cpp)

This command, provided by the Aperture build system, creates a new executable target named ``my_new_problem`` from the source file ``src/main.cpp``.

Step 3: Write the main.cpp Entry Point
---------------------------------------

The ``main.cpp`` file is the heart of your problem. It's where you assemble the simulation components. Here is a breakdown of its structure, based on the ``reconnection`` problem.

1. **Includes and Namespace:**
   Start by including the necessary headers for the framework and the systems you intend to use.

   .. code-block:: cpp

      #include "framework/config.h"
      #include "framework/environment.h"
      #include "framework/system.h"
      #include "systems/data_exporter.h"
      #include "systems/field_solver_default.h"
      #include "systems/grid.h"
      #include "systems/ptc_updater.h"
      #include "systems/domain_comm.h"

      #include <iostream>

      using namespace std;
      using namespace Aperture;


2. **Main Function and Environment Setup:**
   The ``main`` function initializes the ``sim_environment``, which manages the entire simulation. You also define the configuration (e.g., ``Config<2>`` for 2D).

   .. code-block:: cpp

      int main(int argc, char *argv[]) {
        typedef Config<2> Conf;
        sim_environment env(&argc, &argv);
        // ...
      }

3. **Register Systems:**
   Next, you register all the systems (modules) required for the simulation. The order of registration can be important, as some systems depend on others.

   .. code-block:: cpp

      auto comm = env.register_system<domain_comm<Conf>>(env);
      auto grid = env.register_system<grid_t<Conf>>(env, *comm);
      auto solver = env.register_system<field_solver_default<Conf>>(env, *grid, comm);
      auto pusher = env.register_system<ptc_updater<Conf>>(env, *grid, &comm);
      auto exporter = env.register_system<data_exporter<Conf>>(env, *grid, comm);

4. **Set Initial Conditions:**
   This is a critical step where you define the problem-specific initial state. As discussed in the architecture overview, this is distinct from system configuration. The typical pattern is:

   a. Get pointers to the data components managed by the systems (e.g., particles and fields).
   b. Call a dedicated function to populate these components with initial values.

   Following the ``reconnection`` example:

   .. code-block:: cpp

      // Get pointers to data components
      auto *ptcs = pusher->get_ptcs();
      auto *ems = solver->get_ems();

      // Define and call the initial condition function
      auto IC = [&](const typename Conf::coord_t &x) {
        // ... logic for setting initial particle and field values ...
        // Example: set density, temperature, magnetic field, etc.
        double B0 = env.params().get_or("B0", 1.0);
        ems->b(x, 0) = B0 * tanh(x[1] / 0.5);
        // ... more initial condition logic ...
      };

      grid->init_data(ptcs, IC);
      grid->init_data(ems, IC);


5. **Run the Simulation:**
   Finally, initialize the environment and start the simulation loop.

   .. code-block:: cpp

      env.init();
      env.run();
      return 0;

Step 4: Add to the Main Build
-----------------------------

The last step is to tell the main Aperture build system about your new problem. Open the file ``problems/CMakeLists.txt`` and add your problem's directory to the list:

.. code-block:: cmake

   # ... existing problems ...
   add_subdirectory(reconnection)
   add_subdirectory(my_new_problem) # Add this line

After completing these steps, you can re-run CMake and build your new problem executable.