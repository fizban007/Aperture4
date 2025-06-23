Configuration File
=================

The simulation is configured using a TOML (Tom's Obvious, Minimal Language) configuration file. This file contains all the parameters needed to set up and run the simulation. Below is a detailed explanation of the configuration parameters, using one of the reconnection setups ``config_kn.toml`` as an example.

Basic Structure
--------------

The configuration file is organized into several sections:

1. General Simulation Parameters
2. Grid Parameters
3. Particle Parameters
4. Output Parameters
5. Physics Parameters

Let's go through each section in detail.

General Simulation Parameters
---------------------------

These parameters control the basic behavior of the simulation:

.. code-block:: toml

    log_level = 0
    num_species = 2
    ion_mass = 1.0
    dt = 0.05
    q_e = 1.0
    max_steps = 80000

- ``log_level``: Controls the verbosity of the simulation output (0 for minimal output)
- ``num_species``: Number of particle species in the simulation. 2 for electron-positron plasma. 3 for electron-positron-ion plasma.
- ``ion_mass``: Mass of ions in the simulation. Only meaningful for 3-species simulations.
- ``dt``: Time step size
- ``q_e``: Elementary charge. This affects how much charge every particle deposit when computing the current density, but not how the charges move. The charge-to-mass ratio of particles are not affected by this parameter.
- ``max_steps``: Maximum number of simulation steps

Grid Parameters
-------------

These parameters define the computational domain and its properties:

.. code-block:: toml

    N = [4096, 4096]
    ranks = [4, 1]
    guard = [2, 2]
    lower = [0.0, -250.0]
    size = [500.0, 500.0]
    periodic_boundary = [false, false]
    damping_boundary = [true, true, true, true]

- ``N``: Number of grid points in each direction (excluding guard cells). For 2D simulations, use a 2-element array. For 3D simulations, use a 3-element array.
- ``ranks``: Number of MPI ranks in each direction
- ``guard``: Number of guard cells on each end. Total number of cells per dimension is ``N + 2 * guard``.
- ``lower``: Lower limit of the coordinate system.
- ``size``: Size of the simulation box in each dimension. Upper limit of the coordinate grid is ``lower + size``.
- ``periodic_boundary``: Whether boundaries are periodic.
- ``damping_boundary``: Whether to use damping at boundaries. Damping method depends on the coordinate system. For Cartesian coordinates, use PML damping. For spherical coordinates we use a simple damping method.

Particle Parameters
----------------

These parameters control particle behavior and memory allocation:

.. code-block:: toml

    max_ptc_num = 500_000_000
    max_ph_num = 100_000_000
    ptc_buffer_size = 5_000_000
    ph_buffer_size = 5_000_000
    ptc_segment_size = 10_000_000
    ph_segment_size = 10_000_000
    max_tracked_num = 5_000_000

- ``max_ptc_num``: Maximum number of particles on each rank. Simulation will crash if the number of particles exceeds this value.
- ``max_ph_num``: Maximum number of photons on each rank.
- ``ptc_buffer_size``: Communication buffer size for particles. For 2D simulations, there are 9 MPI communication buffers, while for 3D simulations there are 27 MPI communication buffers.
- ``ph_buffer_size``: Communication buffer size for photons
- ``ptc_segment_size``: Sorting segment size for particle data. This only affects how many particles are sorted at a time. Typically it's set to be ~1/20 of the total number of particles.
- ``ph_segment_size``: Sorting segment size for photon data.
- ``max_tracked_num``: Maximum number of tracked particles. This is used for tracked particles output. It should be set to be ``tracked_fraction * max_ptc_num``.

Output Parameters
--------------

These parameters control simulation output:

.. code-block:: toml

    fld_output_interval = 400
    ptc_output_interval = 400
    rho_interval = 1
    tracked_fraction = 0.01
    snapshot_interval = 0
    sort_interval = 13
    downsample = 4

- ``fld_output_interval``: Interval for field output.
- ``ptc_output_interval``: Interval for particle output.
- ``rho_interval``: Interval for charge density calculation. Default to be equal to ``fld_output_interval``, however some algorithms may require charge density every time step. In that case, set it to be 1.
- ``tracked_fraction``: Fraction of particles to track and output.
- ``snapshot_interval``: Interval for taking snapshots. By default, Aperture keeps two snapshots, and overwrites them alternatively. Set it to be 0 to disable snapshots.
- ``sort_interval``: Interval for particle sorting. Typically set this to 13-23. If particles are not sorted, the overall performance will be degraded over time.
- ``downsample``: Field output downsample factor. This reduces the output size. By default, field values are averaged over the ``downsample`` cells. This number should divide ``N`` evenly.

Physics Parameters
---------------

These parameters control the physical setup of the simulation. For reconnection setups, we use the following parameters:

.. code-block:: toml

    # Reconnection setup
    guide_field = 0.0
    sigma = 25.0
    upstream_kT = 0.01
    upstream_n = 5
    upstream_rho = 1.0
    current_sheet_kT = 0.3
    current_sheet_drift = 0.5
    current_sheet_n = 10
    perturbation_amp = 0.1
    perturbation_phase = 0
    perturbation_wavelength = 2000.0

- ``guide_field``: Guide field strengt
- ``sigma``: Magnetization parameter
- ``upstream_kT``: Upstream temperature
- ``upstream_n``: Upstream density
- ``upstream_rho``: Upstream charge density
- ``current_sheet_kT``: Current sheet temperature
- ``current_sheet_drift``: Current sheet drift velocity
- ``current_sheet_n``: Current sheet density
- ``perturbation_amp``: Amplitude of initial perturbation
- ``perturbation_phase``: Phase of initial perturbation
- ``perturbation_wavelength``: Wavelength of initial perturbation

Cooling Parameters
---------------

These parameters control radiative cooling:

.. code-block:: toml

    cooling = true
    sync_compactness = 0.75e-4
    sync_gamma_rad = 400.0
    B_Q = 1.0e-7
    ph_num_bins = 128
    sync_spec_lower = 1.0e-8
    sync_spec_upper = 1.0
    momentum_downsample = 16
    ph_dist_n_th = 32
    ph_dist_n_phi = 64
    IC_bb_kT = 0.01
    IC_bg_spectrum = "black_body"
    IC_opacity = 0.12
    ph_spec_lower = 1.0e-6
    ph_spec_upper = 1.0e3
    emit_photons = false
    produce_pairs = false

- ``cooling``: Whether to enable radiative cooling
- ``sync_compactness``: Synchrotron compactness parameter
- ``sync_gamma_rad``: Synchrotron radiation gamma
- ``B_Q``: Quantum critical field
- ``ph_num_bins``: Number of photon energy bins
- ``sync_spec_lower/upper``: Synchrotron spectrum bounds
- ``momentum_downsample``: Momentum space downsample factor
- ``ph_dist_n_th/phi``: Photon distribution angular resolution
- ``IC_bb_kT``: Inverse Compton blackbody temperature
- ``IC_bg_spectrum``: Inverse Compton background spectrum type
- ``IC_opacity``: Inverse Compton opacity
- ``ph_spec_lower/upper``: Photon spectrum bounds
- ``emit_photons``: Whether to emit photons
- ``produce_pairs``: Whether to produce pairs

Notes
-----

1. All numerical values can be specified in scientific notation (e.g., 1.0e-7)
2. Arrays are specified using square brackets (e.g., [4096, 4096])
3. Boolean values are specified as true/false
4. String values are specified in quotes (e.g., "black_body")
5. Comments can be added using # or //

The configuration file should be placed in the simulation directory and named according to your needs (e.g., config_kn.toml).
