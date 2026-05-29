"""Command construction helpers shared by Prefect tasks."""

import shlex
from typing import Sequence, Union

CommandInput = Union[str, Sequence[object]]


def normalize_command(command: CommandInput) -> list[str]:
    """Return command tokens suitable for golden parity comparisons."""
    if isinstance(command, str):
        return shlex.split(command)
    return [str(token) for token in command]


def command_to_string(command: CommandInput) -> str:
    """Return a shell-safe string representation of command tokens."""
    return shlex.join(normalize_command(command))


def command_matches_fixture(command: CommandInput, expected_tokens: Sequence[object]) -> bool:
    """Compare a command to a normalized golden fixture."""
    return normalize_command(command) == normalize_command(expected_tokens)
