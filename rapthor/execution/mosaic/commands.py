"""Mosaic command builders."""


def build_compress_mosaic_command(mosaic_filename: str) -> list[str]:
    """Build the `fpack` command for one mosaic image."""
    return ["fpack", mosaic_filename]
