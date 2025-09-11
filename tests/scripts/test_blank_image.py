"""
Tests for the blank_image script.
"""

from rapthor.scripts.blank_image import main


def test_main(tmp_path):
    """
    Test the main function of the blank_image script.
    """
    # Define test parameters
    output_image_file = tmp_path / "test_blank_image.fits"
    input_image_file = None  # No input image, we will create a blank one
    vertices_file = None  # No vertices file for this test
    reference_ra_deg = 10.684  # Example RA in degrees
    reference_dec_deg = 41.269  # Example Dec in degrees
    cellsize_deg = 0.001  # Example cell size in degrees
    imsize = "100,100"  # Example image size
    region_file = "[]"  # No region file for this test
    main(
        output_image_file,
        input_image_file,
        vertices_file=vertices_file,
        reference_ra_deg=reference_ra_deg,
        reference_dec_deg=reference_dec_deg,
        cellsize_deg=cellsize_deg,
        imsize=imsize,
        region_file=region_file,
    )
