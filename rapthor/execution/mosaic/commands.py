"""Mosaic command builders."""

from rapthor.execution.commands import bool_token, comma_join


def build_make_mosaic_template_command(
    input_image_filenames: list[str],
    sector_vertices_filenames: list[str],
    template_image_filename: str,
    skip_processing: bool = False,
) -> list[str]:
    """Build the `make_mosaic_template.py` command for one image type."""
    return [
        "make_mosaic_template.py",
        comma_join(input_image_filenames),
        comma_join(sector_vertices_filenames),
        template_image_filename,
        f"--skip={bool_token(skip_processing)}",
    ]


def build_regrid_image_command(
    input_image_filename: str,
    template_image_filename: str,
    sector_vertices_filename: str,
    regridded_image_filename: str,
    skip_processing: bool = False,
) -> list[str]:
    """Build the `regrid_image.py` command for one sector image."""
    return [
        "regrid_image.py",
        input_image_filename,
        template_image_filename,
        sector_vertices_filename,
        regridded_image_filename,
        f"--skip={bool_token(skip_processing)}",
    ]


def build_make_mosaic_command(
    regridded_image_filenames: list[str],
    template_image_filename: str,
    mosaic_filename: str,
    skip_processing: bool = False,
) -> list[str]:
    """Build the `make_mosaic.py` command for one image type."""
    return [
        "make_mosaic.py",
        comma_join(regridded_image_filenames),
        template_image_filename,
        mosaic_filename,
        f"--skip={bool_token(skip_processing)}",
    ]


def build_compress_mosaic_command(mosaic_filename: str) -> list[str]:
    """Build the `fpack` command for one mosaic image."""
    return ["fpack", mosaic_filename]
