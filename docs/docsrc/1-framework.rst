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
add new modules to setup new physical scenarios.

A simulation code does not (usually) need to create and destroy "entities" in
real time, like a game would. Therefore, in *Aperture*, there are only two main
categories of classes: ``system`` and ``data``. ``data`` is what holds the
simulation data, e.g. fields and particles, while ``system`` refers to any
module that works on the data, e.g. pushing particles or evolving fields.

For lack of a better name, the ``sim_environment`` class is a coordinator that
ties things together. It keeps a registry of systems and data components, and
calls every system in order in a giant loop for the duration of the simulation.

Systems
-------

Every system derives from the common base class :ref:`system_t`.


Data Components
---------------

Every data component derives from the common base class :ref:`data_t`.
