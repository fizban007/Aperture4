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

For detailed information about systems and data components, please refer to:

- :doc:`2.1-system` - Documentation about systems and their lifecycle
- :doc:`2.2-data` - Documentation about data components and their usage
