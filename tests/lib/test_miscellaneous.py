"""
Tests for the `rapthor.lib.miscellaneous` module.
"""


def test_download_skymodel(
    ra=None,
    dec=None,
    skymodel_path=None,
    radius=5.0,
    overwrite=False,
    source="TGSS",
    targetname="Patch",
):
    pass


def test_normalize_ra_dec(ra=None, dec=None):
    pass


def test_radec2xy(wcs=None, ra=None, dec=None):
    pass


def test_xy2radec(wcs=None, x=None, y=None):
    pass


def test_make_wcs(ra=None, dec=None, wcs_pixel_scale=10.0 / 3600.0):
    pass


def test_read_vertices(filename=None):
    pass


def test_make_template_image(
    image_name=None,
    reference_ra_deg=None,
    reference_dec_deg=None,
    ximsize=512,
    yimsize=512,
    cellsize_deg=0.000417,
    freqs=None,
    times=None,
    antennas=None,
    aterm_type="tec",
    fill_val=0,
):
    pass


def test_rasterize(verts=None, data=None, blank_value=0):
    pass


def test_string2bool(invar=None):
    pass


def test_string2list(invar=None):
    pass


def test__float_approx_equal(x=None, y=None, tol=None, rel=None):
    pass


def test_approx_equal(x=None, y=None, *args, **kwargs):
    pass


def test_ra2hhmmss(deg=None, as_string=False):
    pass


def test_dec2ddmmss(deg=None, as_string=False):
    pass


def test_convert_mjd2mvt(mjd_sec=None):
    pass


def test_convert_mvt2mjd(mvt_str=None):
    pass


def test_angular_separation(position1=None, position2=None):
    pass


def test_get_reference_station(soltab=None, max_ind=None):
    pass


def test_remove_soltabs(solset=None, soltabnames=None):
    pass


def test_calc_theoretical_noise(obs_list=None, w_factor=1.5):
    pass


def test_find_unflagged_fraction(ms_file=None, start_time=None, end_time=None):
    pass


def test_get_flagged_solution_fraction(h5file=None, solsetname="sol000"):
    pass


def test_transfer_patches(from_skymodel=None, to_skymodel=None, patch_dict=None):
    pass


def test_rename_skymodel_patches(
    skymodel=None, order_dec="high_to_low", order_ra="high_to_low", dec_bin_width=2.0
):
    pass


def test_get_max_spectral_terms(skymodel_file=None):
    pass


def test_nproc():
    pass
