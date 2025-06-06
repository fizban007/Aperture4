========================
 The Aperture Framework
========================

The *Aperture* framework is inspired by the Entity-Component-System (ECS)
paradigm of modern game engines. In a sense, a numerical simulation is very
similar to a video game, where different quantities are evolved over time in a
giant loop, inside which every module is called sequentially to do their job. A
simulation can be much simpler than a video game, since usually no interactivity
is needed. However, there can still be very complex logical dependency between
different modules, and this framework is designed to make it relatively easy to
add new modules or to setup new physical scenarios.

A simulation code does not (usually) need to create and destroy "entities" in
real time, like a game would. Therefore, in *Aperture*, there are only two main
categories of classes: ``system`` and ``data``. ``data`` is what holds the
simulation data, e.g. fields and particles, while ``system`` refers to any
module that works on the data, e.g. pushing particles or evolving fields.

The benefit of such a configuration is that both ``system`` and ``data`` are
flexible and can be plugged in and out depending on the problem. It is also very
straightforward to handle data IO, since we can simply serialize a list of named
``data`` objects.

For lack of a better name, the :ref:`sim_environment` class is a coordinator
that ties things together. It keeps a registry of systems and data components,
and calls every system in order in a giant loop for the duration of the
simulation.

Systems
-------

Every system derives from the common base class :ref:`system_t`. There are three
virtual functions a system can override: ``register_data_components()``,
``init()``, and ``update()``.

register_data_components()
^^^^^^^^^^^^^^^^^^^^^^^^^^

A system needs to work on some data components. Depending on what systems are
initialized, some data components may or may not be used. Therefore, systems are
responsible for managing their own data dependency. For example, a
``field_solver`` system needs to work on :math:`\mathbf{E}` and
:math:`\mathbf{B}` fields, so it needs to register this dependency by overriding the ``register_data_components()`` function:

.. code-block:: cpp

   void register_data_components() {
       // E is a raw pointer to vector_field<Conf>
       E = m_env.register_data<vector_field<Conf>>(
             "E", m_grid, field_type::edge_centered);
       B = m_env.register_data<vector_field<Conf>>(
             "B", m_grid, field_type::edge_centered);
   }

``E`` and ``B`` are pointers to ``vector_field<Conf>`` and are constructed here.
``register_data_components()`` takes an ``std::string`` for a name, followed by
parameters that are passed to the constructor of the data component. If these
data components are registered already by another system under the same name,
then the register_data() function will only return the pointer. No two
components with the same name can co-exist. ``register_data_components()`` is
called right after the constructor of the system, in ``register_system()``.

init()
^^^^^^

The ``init()`` function is called by the simulation environment after all systems
have registered their data components. This is where systems should perform any
necessary initialization of their internal state, parameters, or data structures.
For example, a particle updater system might initialize its random number
generator states or allocate temporary buffers here.

update()
^^^^^^^^

The ``update()`` function is the main workhorse of each system. It is called at
every timestep with two parameters:

- ``dt``: The size of the timestep
- ``step``: The current timestep number

This is where systems perform their core functionality. For example:
- A field solver updates electromagnetic fields
- A particle updater pushes particles and computes currents
- A data exporter writes output files

Data Components
---------------

Every data component derives from the common base class :ref:`data_t`. Data
components are responsible for storing and managing simulation data. They provide
a unified interface for:

1. Initialization (``init()``)
2. Memory management (host/device allocation)
3. Data I/O (serialization/deserialization)
4. Snapshot handling

Common data component types include:

- Fields (``vector_field``, ``scalar_field``)
- Particles (``particle_data_t``, ``photon_data_t``)
- Phase space distributions (``phase_space``, ``phase_space_vlasov``)
- Tracked particles (``tracked_particles_t``, ``tracked_photons_t``)

Each data component can be configured to:
- Skip output (``skip_output()``)
- Include in snapshots (``include_in_snapshot()``)
- Reset after output (``reset_after_output()``)

The Simulation Environment
-------------------------

The simulation environment (``sim_environment``) is the central coordinator that:

1. Manages the registry of systems and data components
2. Handles system initialization and updates
3. Controls the main simulation loop
4. Provides parameter management
5. Handles MPI communication and domain decomposition
6. Manages data I/O and snapshots

Key features:
- Systems are called in registration order
- Data components are uniquely identified by name
- Parameters can be loaded from configuration files
- Support for restarting from snapshots
- Built-in performance monitoring

Example Usage
------------

Here's a typical workflow for setting up a simulation:

.. code-block:: cpp

   // Create simulation environment
   sim_environment env;
   
   // Register systems
   auto grid = env.register_system<grid_t<Config<3>>>();
   auto field_solver = env.register_system<field_solver<Config<3>>>(grid);
   auto ptc_updater = env.register_system<ptc_updater<Config<3>>>(grid);
   auto data_exporter = env.register_system<data_exporter<Config<3>>>(grid);
   
   // Initialize everything
   env.init();
   
   // Run simulation
   env.run();
