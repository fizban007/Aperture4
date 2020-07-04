===============
 The Framework
===============

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

For lack of a better name, the :ref:`sim_environment` class is a coordinator that
ties things together. It keeps a registry of systems and data components, and
calls every system in order in a giant loop for the duration of the simulation.

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
``register_data`` takes an ``std::string`` for a name, followed by parameters
that are passed to the constructor of the data component. If these data
components are registered already by another system under the same name, then
the register_data() function will only return the pointer. No two components
with the same name can co-exist. ``register_data_components()`` is called right
after the constructor of the system, in ``register_system()``.

init()
^^^^^^

update()
^^^^^^^^

Data Components
---------------

Every data component derives from the common base class :ref:`data_t`.
