
# Contributing to Aperture

First off, thank you for considering contributing to Aperture! It's people like you that make Aperture such a great tool.

## Getting Started

- Fork the repository on GitHub.
- Clone your fork locally:

  ```bash
  git clone https://github.com/YOUR_USERNAME/Aperture4.git
  ```

- Set up the development environment as described in the `README.md` and the [documentation](https://fizban007.github.io/Aperture4/).

## Coding Style

To ensure consistency throughout the codebase, we use `clang-format` to format our C++ code. The configuration is defined in the `.clang-format` file at the root of the repository.

Before submitting a pull request, please format your code by running:

```bash
clang-format -i path/to/your/file.cpp
```

This will format the file in place. Key aspects of the style include:

- Indentation: 2 spaces
- Column Limit: 80 characters
- Braces: Attached (K&R style)
- Pointer Alignment: Left

## Testing

All contributions should include tests that cover the new functionality or bug fixes. The tests are located in the `tests/` directory.

To run the tests, build the code with the `build_tests=1` CMake option, and then run `make check` from the build directory:

```bash
cd build
make check
```

## Pull Request Process

1. Ensure that your code adheres to the coding style.
2. Make sure all tests pass.
3. Create a pull request from your fork to the `develop` branch of the main repository.
4. Provide a clear and descriptive title and description for your pull request.
5. Be prepared to address any feedback or requested changes from the maintainers.
