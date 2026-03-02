
# Devcontainer for rapthor

This folder contains configuration for developing rapthor in a [VS Code devcontainer](https://code.visualstudio.com/docs/remote/containers).

## Prerequisites
- VS Code with the "Dev Containers" extension installed
- Docker (or Podman)

## Usage

1. Open the project in VS Code.
2. When prompted, reopen in the devcontainer.
   - Or use Command Palette (`F1` or `Ctrl+Shift+P`): "Dev Containers: Reopen in Container"
3. The container will build using the `Docker/Dockerfile` and install all runtime dependencies.
4. After the container is created, all testing dependencies are installed automatically via `tox`.
5. You can now develop, run, and test rapthor inside the container.

> **Note:** The initial build may take up to an hour, depending on your system and network speed.

## Running Tests

- To run tests, use `tox` (preferred) or `pytest` inside the container:
  - `tox` will set up the environment and run all test environments as defined in `tox.ini`.
  - You can also run `pytest` directly if you want to run tests in the current environment.

