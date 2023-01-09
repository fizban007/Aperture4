==============
 Introduction
==============

The name *Aperture* is a recursive acronym that stands for: Aperture is a code for
Particles, Electrodynamics, and Radiative Transfer at Ultra-Relativistic
Energies. As the name suggests, its main goal is to simulation the interaction
of ultra-relativistic particles with electromagnetic field and radiation, which
occurs very often in astrophysical scenarios.

Downloading Aperture
--------------------

To download and compile *Aperture*, you can clone it from the github repo:

.. code-block:: console

   $ git clone https://github.com/fizban007/Aperture4.git

Development in a Container
--------------------------

The easiest way to compile and develop *Aperture* is to use a Docker container.
The official development container is ``fizban007/aperture_dev``. The image is
generated using the ``develop.Dockerfile`` under the main *Aperture* git repo. It
contains all the necessary libraries to develop *Aperture*.

In order to avoid using the root user in the container, a default user called
``developer`` is created by default. It is recommended to mount a Docker volume
(or a host directory) to the root directory ``/code`` which is writable by user ``developer``. One can do this using:

.. code-block:: console

   $ docker pull fizban007/aperture_dev
   $ docker volume create vol1
   $ docker run -it --name Aperture_dev --mount source=vol1,target=/code fizban007/aperture_dev

This will create a container named ``Aperture_dev`` which mounts the volume at the
correct location. Once in the container, one can clone the main code repo, use
VS Code to attach to the container and carry out development and debugging
there.

Development on a Host Machine
-----------------------------

To compile *Aperture* from scratch, you will need to have several libraries ready:

* A modern C++ compiler that supports ``c++17``, e.g.: ``gcc>=7.0`` or ``clang>=3.9`` or ``intel>=19.0``

* An MPI implementation, e.g. ``openmpi``, ``intel-mpi``, ``mpich``, etc.
* An ``hdf5`` library, preferably an ``mpi``-enabled version
* The C++ library ``boost>=1.54``
* (Optional) The Cuda toolkit ``cuda>=11`` that supports at least ``c++17``
* (Optional) The ROCm toolkit provided by AMD

To help with this, a bunch of configuration scripts under the ``machines``
directory should contain the necessary ``module load`` for the given machine to
compile the code.

Compiling the Code
------------------

Whether you use a Docker container or manage the libraries on your own, you can compile the code using the following:

.. code-block:: console

   $ mkdir build & cd build
   $ cmake ..
   $ make -j8

This will compile the main *Aperture* library, the test cases, and the problems.
The binary files for specific setups will be generated inside the corresponding
``problems/XXX/bin`` directory.

The following are the options (and their default values) to ``cmake`` which
controls some compile-time configurations:

* ``-Duse_cuda=0``: If this option is 1, include the GPU part of *Aperture* and compile it as CUDA code.
* ``-Duse_hip=0``: If this option is 1, include the GPU part of *Aperture* and
  compile it as the AMD HIP code. Note that one can specify both of these two
  options. In that case, it will try to compile the code through the HIP but
  using its CUDA implementation.
* ``-Dbuild_tests=1``: If this option is 1, build the unit test suite.
* ``-Duse_double=0``: If this option is 1, use double precision for all floating
  point calculations.
* ``-Dlib_only=0``: If this option is 1, only build the *Aperture* library, but
  not any simulation setups in the ``problems`` directory.
* ``-Duse_libcpp=0``: This option is specifically for the situation where you want
  to compile the code with ``clang`` and link to ``libc++`` instead of ``libstdc++``.
  Generally keep it off unless you know what you are doing. Even then, it is not
  guaranteed to work.

Apart from these options, standard CMake options apply as well. For example, the
code will compile in `Release` mode by default. To change it to `Debug`, append
``-DCMAKE_BUILD_TYPE=Debug`` to the ``cmake`` command.

To run a small suite of unit tests, run ``make check`` in the build directory. To
generate this documentation, run ``make docs``. The dependencies for generating
documentation include ``Doxygen``, ``Sphinx``, ``Breathe``, and ``sphinx_rtd_theme``.
The latter 3 are python packages and preferably installed in a python virtual
environment.

Run Simulations
---------------

To run a quick test simulation, checkout the ``training`` directory in
``problems``. It contains several self-contained simple setups that can run
without fiddling with configuration files. For a more in-depth guide on
simulation setup, checkout :doc:`Setting up a Simulation <1-setup>`.
