==============
 Introduction
==============

The name *Aperture* is a recursive acronym that stands for: Aperture is a code for
Particles, Electrodynamics, and Radiative Transfer at Ultra-Relativistic
Energies. As the name suggests, its main goal is to simulation the interaction
of ultra-relativistic particles with electromagnetic field and radiation, which
occurs very often in astrophysical scenarios.

Quickstart
----------

To download and compile *Aperture*, you can clone it from the github repo:

.. code-block:: console

   $ git clone https://github.com/fizban007/Aperture4.git

To compile, you will need to have several libraries ready:

* A modern C++ compiler that supports ``c++17``, e.g.:

  * ``gcc>=7.0``
  * ``clang>=3.9``
  * ``intel>=19.0``

* An MPI implementation, e.g. ``openmpi``, ``intel-mpi``, ``mpich``, etc.
* ``hdf5`` library, preferably an ``mpi``-enabled version
* ``boost>=1.54``
* The Cuda toolkit ``cuda>=8.0`` that supports at least ``c++14``

To help with this, a bunch of configuration scripts under the ``machines``
directory should contain the necessary ``module load`` for the given machine to
compile the code.

After you make sure the dependencies are all there, simply do:

.. code-block:: console

   $ mkdir build & cd build
   $ cmake ..
   $ make

This will compile the main *Aperture* library. To compile problem specific code,
either use ``make problems``, or ``make`` the specific target under the
``problems`` directory. The binary file will be generated inside the
corresponding ``problems/XXX/bin`` directory.
