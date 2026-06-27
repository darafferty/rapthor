"""Command construction helpers shared by Prefect tasks."""

import shlex
from typing import Sequence, Union

CommandInput = Union[str, Sequence[object]]


def bool_token(value: bool) -> str:
    """Return the capitalized boolean spelling used by Rapthor command builders."""
    return "True" if value else "False"


def comma_join(values: Sequence[object]) -> str:
    """Return values as one comma-separated command token."""
    return ",".join(str(value) for value in values)


def bracketed_list_token(values: Sequence[object]) -> str:
    """Return values as a bracketed comma-separated command token."""
    return f"[{comma_join(values)}]"


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
