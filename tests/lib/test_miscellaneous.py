"""
Tests for the `rapthor.lib.miscellaneous` module.
"""

import numpy
import pytest
from rapthor.lib.miscellaneous import *


@pytest.mark.parametrize("ra", (10.75,))
@pytest.mark.parametrize("dec", (5.34,))
@pytest.mark.parametrize("skymodel_path", ("/tmp/sky.model",))
@pytest.mark.parametrize("radius", (5.0,))
@pytest.mark.parametrize("overwrite", (False,))
@pytest.mark.parametrize("source", ("TGSS",))
@pytest.mark.parametrize("targetname", ("Patch",))
def test_download_skymodel(
    ra, dec, skymodel_path, radius, overwrite, source, targetname
):
    download_skymodel(ra, dec, skymodel_path, radius, overwrite, source, targetname)


@pytest.mark.parametrize("ra", (190.75,))
@pytest.mark.parametrize("dec", (-115.34,))
def test_normalize_ra_dec(ra, dec):
    normalize_ra_dec(ra, dec)


@pytest.mark.parametrize("wcs", (None,))  # maybe use make_wcs to create a WCS object?
@pytest.mark.parametrize("ra", (10.75,))
@pytest.mark.parametrize("dec", (5.34,))
def test_radec2xy(wcs, ra, dec):
    # radec2xy(wcs, ra, dec)
    pass


@pytest.mark.parametrize("ra", (None,))
@pytest.mark.parametrize("dec", (None,))
@pytest.mark.parametrize("wcs_pixel_scale", (10.0 / 3600.0,))
def test_make_wcs(ra, dec, wcs_pixel_scale):
    make_wcs(ra, dec, wcs_pixel_scale)


@pytest.mark.parametrize("filename", ("/path/to/vertices.file",))
def test_read_vertices(filename):
    wcs = make_wcs(0, 0)
    with pytest.raises(FileNotFoundError):
        read_vertices(filename, wcs)


@pytest.mark.parametrize("image_name", (None,))
@pytest.mark.parametrize("reference_ra_deg", (None,))
@pytest.mark.parametrize("reference_dec_deg", (None,))
@pytest.mark.parametrize("ximsize", (512,))
@pytest.mark.parametrize("yimsize", (512,))
@pytest.mark.parametrize("cellsize_deg", (0.000417,))
@pytest.mark.parametrize("freqs", (None,))
@pytest.mark.parametrize("times", (None,))
@pytest.mark.parametrize("antennas", (None,))
@pytest.mark.parametrize("aterm_type", ("tec",))
@pytest.mark.parametrize("fill_val", (0,))
def test_make_template_image(
    image_name,
    reference_ra_deg,
    reference_dec_deg,
    ximsize,
    yimsize,
    cellsize_deg,
    freqs,
    times,
    antennas,
    aterm_type,
    fill_val,
):
    make_template_image(
        image_name,
        reference_ra_deg,
        reference_dec_deg,
        ximsize,
        yimsize,
        cellsize_deg,
        freqs,
        times,
        antennas,
        aterm_type,
        fill_val,
    )


@pytest.mark.parametrize("verts", ([(2.7, 3.1), (3.2, 4.5), (5.1, 6.3)],))
@pytest.mark.parametrize("data", (numpy.zeros((5, 5)),))
@pytest.mark.parametrize("blank_value", (0,))
def test_rasterize(verts, data, blank_value):
    rasterize(verts, data, blank_value)


@pytest.mark.parametrize("invar", (None,))
def test_string2bool(invar):
    string2bool(invar)


@pytest.mark.parametrize("invar", (None,))
def test_string2list(invar):
    string2list(invar)


@pytest.mark.parametrize("x", (1.23456789,))
@pytest.mark.parametrize("y", (1.23457890,))
@pytest.mark.parametrize("args", ([1e-5],))
@pytest.mark.parametrize("kwargs", ({"rel": 1e-5},))
def test_approx_equal(x, y, args, kwargs):
    assert approx_equal(x, y, *args, **kwargs)


@pytest.mark.parametrize("deg", (34.56,))
@pytest.mark.parametrize("as_string", (False,))
def test_ra2hhmmss(deg, as_string):
    ra2hhmmss(deg, as_string)


@pytest.mark.parametrize("deg", (34.56,))
@pytest.mark.parametrize("as_string", (False,))
def test_dec2ddmmss(deg, as_string):
    dec2ddmmss(deg, as_string)


@pytest.mark.parametrize("mjd_sec", (4567890123,))
def test_convert_mjd2mvt(mjd_sec):
    assert convert_mvt2mjd(convert_mjd2mvt(mjd_sec)) == mjd_sec


@pytest.mark.parametrize("mvt_str", ("18Aug2003/02:22:03.000",))
def test_convert_mvt2mjd(mvt_str):
    assert convert_mjd2mvt(convert_mvt2mjd(mvt_str)) == mvt_str


@pytest.mark.parametrize(
    "position1, position2, separation",
    [((0.0, 0.0), (45.0, 45.0), 60.0)],
)
def test_angular_separation(position1, position2, separation):
    assert angular_separation(position1, position2).value == pytest.approx(separation)


@pytest.mark.parametrize("soltab", (None,))  # we may want to use a fixture here
@pytest.mark.parametrize("max_ind", (None,))
def test_get_reference_station(soltab, max_ind):
    # get_reference_station(soltab, max_ind)
    pass


@pytest.mark.parametrize("solset", (None,))  # we may want to use a fixture here
@pytest.mark.parametrize("soltabnames", (None,))
def test_remove_soltabs(solset, soltabnames):
    # remove_soltabs(solset, soltabnames)
    pass


@pytest.mark.parametrize("obs_list", ([],))
@pytest.mark.parametrize("w_factor", (1.5,))
def test_calc_theoretical_noise(obs_list, w_factor):
    calc_theoretical_noise(obs_list, w_factor)


@pytest.mark.parametrize("ms_file", ("ms_file",))  # we may want to use a fixture here
@pytest.mark.parametrize("start_time", (4567890123,))
@pytest.mark.parametrize("end_time", (4567890134,))
def test_find_unflagged_fraction(ms_file, start_time, end_time):
    # find_unflagged_fraction(ms_file, start_time, end_time)
    pass


@pytest.mark.parametrize("h5file", ("h5file",))  # we may want to use a fixture here
@pytest.mark.parametrize("solsetname", ("sol000",))
def test_get_flagged_solution_fraction(h5file, solsetname):
    # get_flagged_solution_fraction(h5file, solsetname)
    pass


@pytest.mark.parametrize("from_skymodel", (None,))  # we may want to use a fixture here
@pytest.mark.parametrize("to_skymodel", (None,))
@pytest.mark.parametrize("patch_dict", (None,))
def test_transfer_patches(from_skymodel, to_skymodel, patch_dict):
    # transfer_patches(from_skymodel, to_skymodel, patch_dict)
    pass


@pytest.mark.parametrize("skymodel", (None,))  # we may want to use a fixture here
@pytest.mark.parametrize("order_dec", (None,))
@pytest.mark.parametrize("order_ra", (None,))
@pytest.mark.parametrize("dec_bin_width", (2.0,))
def test_rename_skymodel_patches(skymodel, order_dec, order_ra, dec_bin_width):
    # rename_skymodel_patches(skymodel, order_dec, order_ra, dec_bin_width)
    pass


@pytest.mark.parametrize("skymodel_file", ("skymodel_file",))
def test_get_max_spectral_terms(skymodel_file):
    with pytest.raises(FileNotFoundError):
        get_max_spectral_terms(skymodel_file)


def test_nproc():
    assert nproc() == int(subprocess.run(["nproc"], capture_output=True).stdout)
