# Dev Container for Rapthor

This directory contains the configuration for a VS Code development container that provides a complete development environment for Rapthor.

## Features

- **Complete Environment**: Built using the same Dockerfile as production (`Docker/Dockerfile`)
- **Pre-installed Dependencies**: All required tools (DP3, WSClean, EveryBeam, IDG, etc.)
- **Python Development Tools**: Configured with Pylance, Black formatter, Flake8 linter, and pytest
- **Auto-installation**: Rapthor is automatically installed in editable mode on container creation

## Usage

### Prerequisites
- VS Code with the "Dev Containers" extension installed
- Docker or Podman

### Opening the Dev Container

1. Open this folder in VS Code
2. Click the notification "Reopen in Container" or:
   - Press `F1` or `Cmd+Shift+P` (macOS) / `Ctrl+Shift+P` (Linux/Windows)
   - Type "Dev Containers: Reopen in Container"
   - Press Enter

VS Code will build the container (first time may take 30+ minutes) and reopen the workspace inside it.

### Working in the Container

Once inside the container:
- All commands run in the containerized environment
- The workspace is mounted at `/workspace`
- Rapthor is installed in editable mode
- Run tests with `python -m pytest tests -v`
- Run Rapthor with `rapthor examples/rapthor.parset`

### Rebuilding the Container

If you modify the Dockerfile or need to update dependencies:
- Press `F1` / `Cmd+Shift+P` / `Ctrl+Shift+P`
- Type "Dev Containers: Rebuild Container"
- Press Enter

## Configuration

The dev container configuration is in `.devcontainer/devcontainer.json` and includes:
- Build arguments for dependency versions
- Python interpreter and tool settings
- Recommended VS Code extensions
- Post-creation commands

## Notes

- The container uses the same Dockerfile as production for consistency
- Build args can be modified in `devcontainer.json` to pin specific dependency versions
- The workspace is mounted with cached consistency for better performance
