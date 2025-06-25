"""
Test suite for rapthor.scripts.calculate_image_diagnostics.
"""

from rapthor.scripts.calculate_image_diagnostics import (
    check_astrometry, check_photometry, fits_to_makesourcedb, main,
    plot_astrometry_offsets)


def test_plot_astrometry_offsets():
    """
    Test the plot_astrometry_offsets function.
    """
    # Define test parameters
    # facets = None  # Replace with actual test data
    # field_ra = 10.684  # Example RA in degrees
    # field_dec = 41.269  # Example Dec in degrees
    # output_file = "test_astrometry_offsets.png"  # Output file for the plot
    # plot_astrometry_offsets(facets, field_ra, field_dec, output_file, plot_labels=False)
    pass

def test_fits_to_makesourcedb():
    """
    Test the fits_to_makesourcedb function.
    """
    # catalog = "input_catalog.fits"
    # reference_freq = 1.4e9  # Example reference frequency in Hz
    # fits_to_makesourcedb(catalog, reference_freq, flux_colname='Isl_Total_flux')
    pass

def test_check_photometry():
    """
    Test the check_photometry function.
    """
    # Define test parameters
    # obs = "obs.ms"
    # input_catalog = "input_catalog.fits"
    # freq = 1.4e9  # Example frequency in Hz
    # min_number = 5
    # comparison_skymodel = "comparison_skymodel.fits"
    # comparison_surveys = ['TGSS', 'LOTSS']
    # backup_survey = 'NVSS'
    # check_photometry(obs, input_catalog, freq, min_number, comparison_skymodel=comparison_skymodel,
    #                  comparison_surveys=comparison_surveys, backup_survey=backup_survey)
    pass

def test_check_astrometry():
    """
    Test the check_astrometry function.
    """
    # Define test parameters
    # obs = "obs.ms"
    # input_catalog = "input_catalog.fits"
    # image = "image.fits"
    # facet_region_file = "facet_region.reg"
    # min_number = 5
    # output_root = "output_root"
    # comparison_skymodel = "comparison_skymodel.fits"
    # check_astrometry(obs, input_catalog, image, facet_region_file, min_number,
    #                  output_root, comparison_skymodel=None)
    pass


def test_main():
    """
    Test the main function.
    """
    # # Define test parameters
    # flat_noise_image = "flat_noise_image.fits"
    # flat_noise_rms_image = "flat_noise_rms_image.fits"
    # true_sky_image = "true_sky_image.fits"
    # true_sky_rms_image = "true_sky_rms_image.fits"
    # input_catalog = "input_catalog.fits"
    # input_skymodel = "input_skymodel.fits"
    # obs_ms = "obs.ms"
    # diagnostics_file = "diagnostics.txt"
    # output_root = "output_root"
    # facet_region_file = "facet_region.reg"
    # photometry_comparison_skymodel = "photometry_comparison_skymodel.fits"
    # photometry_comparison_surveys = ["TGSS", "NVSS"]
    # photometry_backup_survey = "NVSS"
    # astrometry_comparison_skymodel = "astrometry_comparison_skymodel.fits"
    # min_number = 5
    # main(flat_noise_image, flat_noise_rms_image, true_sky_image, true_sky_rms_image,
    #      input_catalog, input_skymodel, obs_ms, diagnostics_file, output_root,
    #      facet_region_file=facet_region_file, photometry_comparison_skymodel=photometry_comparison_skymodel,
    #      photometry_comparison_surveys=photometry_comparison_surveys, photometry_backup_survey=photometry_backup_survey,
    #      astrometry_comparison_skymodel=astrometry_comparison_skymodel, min_number=min_number)
    pass