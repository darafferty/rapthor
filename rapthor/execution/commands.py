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


def append_prefixed_value(command: list[str], prefix: str, value: object) -> None:
    """Append one combined command token when the optional value is present."""
    if value is not None:
        command.append(f"{prefix}{value}")


def append_option_value(command: list[str], option: str, value: object) -> None:
    """Append a command option and value as separate tokens."""
    command.extend([option, str(value)])


def append_flag(command: list[str], option: str, enabled: bool) -> None:
    """Append a flag token when its boolean setting is enabled."""
    if enabled:
        command.append(option)


def append_option_values(command: list[str], options: Sequence[tuple[str, object]]) -> None:
    """Append option/value pairs, expanding list values into repeated value tokens."""
    for option, value in options:
        if value is None:
            continue
        if isinstance(value, list):
            command.append(option)
            command.extend(str(item) for item in value)
        else:
            append_option_value(command, option, value)


def append_key_value(command: list[str], key: str, value: object) -> None:
    """Append a `key=value` token, using DP3-style bool and list values."""
    if value is None:
        return
    if isinstance(value, bool):
        value = bool_token(value)
    elif isinstance(value, list):
        if any(item is None for item in value):
            return
        value = bracketed_list_token(value)
    command.append(f"{key}={value}")


def normalize_command(command: CommandInput) -> list[str]:
    """Return command tokens suitable for golden parity comparisons."""
    if isinstance(command, str):
        return shlex.split(command)
    return [str(token) for token in command]


def command_to_string(command: CommandInput) -> str:
    """Return a shell-safe string representation of command tokens."""
    return shlex.join(normalize_command(command))
