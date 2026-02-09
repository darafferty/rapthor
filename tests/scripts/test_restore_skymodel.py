import pytest


import astropy.coordinates as coords
from astropy.io.fits import CompImageHDU, PrimaryHDU
import astropy.io.fits as fits
import numpy as np
from rapthor.scripts.restore_skymodel import (
    main as restore_main,
    log_fits_info,
    get_primary_hdu_or_compressed,
    make_zero_image,
    compress_image_if_needed
)

def get_first_source(sky_model_path)-> list:
    """
    Fixture to get the first source from the apparent skymodel for testing.
    """
    for line in sky_model_path.read_text().splitlines():
        if (line := line.strip()) and not line.startswith(("#", "FORMAT", ",")):
            return [value.strip() for value in line.split(",")]

    raise ValueError("No valid source found in the skymodel.")


def make_reference_image(image_path, sky_model_path):
    """
    Fixture to create a reference image for testing.
    """
    _, _, _, ra_str, dec_str, *_ = get_first_source(sky_model_path)
    if ":" not in dec_str:
        dec_str = dec_str.replace(".", ":", 2)
    reference_coord = coords.SkyCoord(ra=ra_str, dec=dec_str, unit=('hourangle', 'deg'))
    
    header = fits.Header()
    n_pixels = 100

    header['NAXIS'] = 2
    header['NAXIS1'] = n_pixels
    header['NAXIS2'] = n_pixels
    header['CTYPE1'] = 'RA---SIN'
    header['CTYPE2'] = 'DEC--SIN'
    header['CRVAL1'] = reference_coord.ra.deg 
    header['CRVAL2'] = reference_coord.dec.deg
    header['CRPIX1'] = 0
    header['CRPIX2'] = 0
    header['CDELT1'] = -0.01
    header['CDELT2'] = 0.01
    generator = np.random.default_rng(42)
    data = generator.random((n_pixels, n_pixels)).astype(np.float32)
    
    if ".fz" in image_path.suffixes:    
        compressed_hdu = CompImageHDU(data=data, header=header, compression_type='RICE_1')
        compressed_hdu.writeto(image_path)
    else:
        primary_hdu = PrimaryHDU(data=data, header=header)
        primary_hdu.writeto(image_path)
    return image_path

@pytest.fixture
def reference_image(tmp_path, sky_model_path):
    """
    Fixture to get the reference image path for testing.
    """
    image_path = tmp_path / "reference_image.fits"
    return make_reference_image(image_path, sky_model_path)


@pytest.fixture
def reference_image_compressed(tmp_path, sky_model_path):
    """
    Fixture to get the compressed reference image path for testing.
    """
    image_path = tmp_path / "reference_image.fits.fz"
    return make_reference_image(image_path, sky_model_path)

    

def test_integration_restore_skymodel(reference_image, sky_model_path, tmp_path):
    """
    Integration test for the restore_skymodel script.
    This test checks if the restored image is created correctly from the apparent skymodel.
    """
    output_image = tmp_path / "restored_image.fits"

    # Run the restore_skymodel script
    restore_main(
        source_catalog=sky_model_path,
        reference_image=reference_image,
        output_image=output_image
    )

    # Check if the output image was created
    assert output_image.exists(), "Restored image was not created."


def test_integration_restore_skymodel_compressed(reference_image_compressed, sky_model_path, tmp_path):
    """
    Integration test for the restore_skymodel script with compressed reference image.
    This test checks if the restored image is created correctly from the apparent skymodel.
    """
    output_image = tmp_path / "restored_image_compressed.fits.fz"

    # Run the restore_skymodel script
    restore_main(
        source_catalog=sky_model_path,
        reference_image=reference_image_compressed,
        output_image=output_image
    )

    # Check if the output image was created
    assert output_image.exists(), "Restored compressed image was not created."


def test_log_fits_info(reference_image, caplog):
    """
    Test the log_fits_info function to ensure it logs FITS file information correctly.
    """
    with fits.open(reference_image) as fits_obj:
        log_fits_info(fits_obj)
    
    # Check that the log output contains expected HDU information
    assert "HDU" in caplog.text, "Log should contain HDU information"
    assert "name=" in caplog.text, "Log should contain HDU name information"


def test_get_primary_hdu_or_compressed_with_primary_with_uncompressed_returns_primary(reference_image):
    """
    Test get_primary_hdu_or_compressed with a standard FITS file (PrimaryHDU).
    """
    with fits.open(reference_image) as fits_obj:
        hdu = get_primary_hdu_or_compressed(fits_obj)
        
        assert isinstance(hdu, PrimaryHDU), "Should return PrimaryHDU for standard FITS file"
        assert hdu.header['NAXIS'] == 2, "HDU should have correct NAXIS"


def test_get_primary_hdu_or_compressed_with_compressed_returns_compressed(reference_image_compressed):
    """
    Test get_primary_hdu_or_compressed with a compressed FITS file (CompImageHDU).
    """
    with fits.open(reference_image_compressed) as fits_obj:
        fits_obj.info()
        hdu = get_primary_hdu_or_compressed(fits_obj)
        
        assert isinstance(hdu, CompImageHDU), "Should return CompImageHDU or PrimaryHDU"
        assert hdu.header['NAXIS'] == 2, "HDU should have correct NAXIS"


def test_make_zero_image(reference_image):
    """
    Test make_zero_image to ensure it creates a zero-valued image with the correct header.
    """
    zero_image_path, pixel_scale = make_zero_image(reference_image)
    
    try:
        assert zero_image_path.exists(), "Zero image should be created"
        
        # Open and verify the zero image
        with fits.open(zero_image_path) as zero_fits, \
             fits.open(reference_image) as ref_fits:
            zero_hdu = zero_fits[0]
            ref_hdu = ref_fits[0]
            assert pixel_scale > 0, "Pixel scale should be positive"
            assert pixel_scale == min(abs(ref_hdu.header['CDELT1']) * 3600.0,
                                      abs(ref_hdu.header['CDELT2']) * 3600.0), "Pixel scale should match derived minimum scale"    
            # Check that data is all zeros
            assert np.all(zero_hdu.data == 0), "Image data should be all zeros"
            
            # Check that dimensions match
            assert zero_hdu.data.shape == ref_hdu.data.shape, "Dimensions should match reference"
            
            # Check that key header values are preserved
            assert zero_hdu.header['NAXIS1'] == ref_hdu.header['NAXIS1'], "NAXIS1 should match"
            assert zero_hdu.header['NAXIS2'] == ref_hdu.header['NAXIS2'], "NAXIS2 should match"
    finally:
        # Clean up temporary file
        zero_image_path.unlink(missing_ok=True)


def test_make_zero_image_with_compressed(reference_image_compressed):
    """
    Test make_zero_image with compressed reference image.
    """
    zero_image_path, _ = make_zero_image(reference_image_compressed)
    
    try:
        assert zero_image_path.exists(), "Zero image should be created from compressed reference"
        
        # Open and verify the zero image
        with fits.open(zero_image_path) as zero_fits:
            zero_hdu = zero_fits[0]
            
            # Check that data is all zeros
            assert np.all(zero_hdu.data == 0), "Image data should be all zeros"
            
            # Check basic structure
            assert zero_hdu.header['NAXIS'] == 2, "Should have 2 axes"
    finally:
        # Clean up temporary file
        zero_image_path.unlink(missing_ok=True)


def test_compress_image_if_needed_no_compression(tmp_path):
    """
    Test compress_image_if_needed when output has .fits extension (no compression needed).
    """
    # Create a simple test FITS file
    input_image = tmp_path / "input.fits"
    output_image = tmp_path / "output.fits"
    
    data = np.ones((10, 10))
    header = fits.PrimaryHDU().header
    header['SIMPLE'] = True
    fits.writeto(input_image, data=data, header=header)
    
    result = compress_image_if_needed(input_image, output_image)
    
    assert result == output_image, "Should return output path"
    assert output_image.exists(), "Output file should exist"
    
    # Verify content was copied
    with fits.open(output_image) as fits_obj:
        assert fits_obj[0].data.shape == (10, 10), "Data should be preserved"


def test_compress_image_if_needed_with_compression(tmp_path):
    """
    Test compress_image_if_needed when output has compressed extension.
    """
    # Create a simple test FITS file
    input_image = tmp_path / "input.fits"
    output_image = tmp_path / "output.fits.fz"
    
    data = np.ones((10, 10))
    header = fits.PrimaryHDU().header
    header['SIMPLE'] = True
    fits.writeto(input_image, data=data, header=header)
    
    result = compress_image_if_needed(input_image, output_image)
    
    assert result == output_image, "Should return output path"
    assert output_image.exists(), "Compressed output file should exist"
