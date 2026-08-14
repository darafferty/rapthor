
# Devcontainer for rapthor

This folder contains configuration for developing rapthor in a [VS Code devcontainer](https://code.visualstudio.com/docs/remote/containers).

## Prerequisites
- VS Code with the "Dev Containers" extension installed
- Docker (or Podman)

## Usage

1. Open the project in VS Code.
2. When prompted, reopen in the devcontainer.
   - Or use Command Palette (`F1` or `Ctrl+Shift+P`): "Dev Containers: Reopen in Container"
3. The container will build the `dev` target from `ci/ubuntu_24_04-base`, matching the CI OS,
   compiled tools, and runtime dependencies.
4. After the container is created, Rapthor and its development dependencies are installed in
   `/opt/rapthor-venv`.
5. You can now develop, run, and test rapthor inside the container.

> **Note:** The initial build may take up to an hour, depending on your system and network speed.

## Running Tests

- To run tests, use `tox` (preferred) or `pytest` inside the container:
  - `tox` will set up the environments defined in `pyproject.toml` and run their tests.
  - You can also run `pytest` directly if you want to run tests in the current environment.
