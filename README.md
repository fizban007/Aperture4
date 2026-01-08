# Aperture

Aperture is a high-performance, parallel, Particle-in-Cell (PIC) code framework for plasma simulation. It is designed to be fast, flexible, and easy to extend, and it can run on both traditional CPUs and modern GPUs (NVIDIA and AMD).

For a more detailed introduction, please see the [full documentation](https://fizban007.github.io/Aperture4/).

## Features

*   **High-performance:** Designed for speed and scalability on large core counts and GPUs.
*   **Parallel:** Supports both distributed memory parallelism with MPI and shared memory parallelism with OpenMP.
*   **GPU-accelerated:** Supports both CUDA for NVIDIA GPUs and HIP for AMD GPUs.
*   **Modular and Extensible:** A flexible architecture allows for easy addition of new physics modules, numerical methods, and diagnostics.

## Getting Started

### Prerequisites

*   A C++17 compliant compiler (e.g., GCC, Clang, Intel)
*   CMake (version 3.8 or later)
*   MPI
*   HDF5

### Building Aperture

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fizban007/Aperture4.git
    cd Aperture4
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Run CMake and build:**
    ```bash
    cmake ..
    make -j
    ```

### Running Aperture

The executables for the different simulation problems are located in the `bin` directory. To run a simulation, simply execute the corresponding binary. For example:

```bash
./bin/two_stream
```

## Documentation

The full documentation can be built from the `docs` directory.

```bash
mkdir build_docs
cd build_docs
cmake ../docs
make
```

The documentation will be generated in the `html` directory.
