
import pytest


@pytest.fixture
def expected_image_output():
    """
    Fixture which provides the expected output structure for CWL execution
    of an image step in the non-last cycle.
    """
    return {
        "sector_I_images": [["sector0-MFS-I-image-pb.fits", "sector0-MFS-I-image.fits"]],
        "sector_extra_images": [["sector0-MFS-I-residual.fits", "sector0-MFS-I-model-pb.fits", "sector0-MFS-I-dirty.fits"]],
        "filtered_skymodel_true_sky": ["sector0.true_sky.txt"],
        "filtered_skymodel_apparent_sky": ["sector0.apparent_sky.txt"],
        "pybdsf_catalog": ["sector0.source_catalog.fits"],
        "sector_diagnostics": ["sector0_diagnostics.json"],
        "sector_offsets": ["sector0_offsets.txt"],
        "source_filtering_mask": ["sector0_mask.fits"],
    }


@pytest.fixture
def expected_image_output_last_cycle():
    """
    Fixture which provides the expected output structure for CWL execution
    of the image step in the last cycle.
    """
    return {
        "sector_I_images": [["sector0-MFS-I-image-pb.fits", "sector0-MFS-I-image.fits"]],
        "filtered_skymodel_true_sky": ["sector0.true_sky.txt"],
        "filtered_skymodel_apparent_sky": ["sector0.apparent_sky.txt"],
        "pybdsf_catalog": ["sector0.source_catalog.fits"],
        "sector_diagnostics": ["sector0_diagnostics.json"],
        "sector_offsets": ["sector0_offsets.txt"],
        "source_filtering_mask": ["sector0_mask.fits"],
        "sector_extra_images": [[
            'sector_1-MFS-I-image-pb.fits',
            'sector_1-MFS-I-image-pb.fits',
            'sector_1-MFS-I-image.fits',
            'sector_1-MFS-Q-image.fits',
            'sector_1-MFS-U-image.fits',
            'sector_1-MFS-V-image.fits',
            'sector_1-MFS-Q-image-pb.fits',
            'sector_1-MFS-U-image-pb.fits',
            'sector_1-MFS-V-image-pb.fits',
            'sector_1-MFS-I-residual.fits',
            'sector_1-MFS-Q-residual.fits',
            'sector_1-MFS-U-residual.fits',
            'sector_1-MFS-V-residual.fits',
            'sector_1-MFS-I-model-pb.fits',
            'sector_1-MFS-Q-model-pb.fits',
            'sector_1-MFS-U-model-pb.fits',
            'sector_1-MFS-V-model-pb.fits',
            'sector_1-MFS-I-dirty.fits',
            'sector_1-MFS-Q-dirty.fits',
            'sector_1-MFS-U-dirty.fits',
            'sector_1-MFS-V-dirty.fits'
        ]]
    }
