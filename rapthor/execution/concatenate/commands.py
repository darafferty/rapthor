"""Concatenate command builders."""


def build_concatenate_command(
    input_filenames: list[str],
    output_filename: str,
    data_colname: str,
) -> list[str]:
    """Build the `concat_ms.py` command for one epoch."""
    return [
        "concat_ms.py",
        *input_filenames,
        f"--msout={output_filename}",
        "--concat_property=frequency",
        f"--data_colname={data_colname}",
    ]
