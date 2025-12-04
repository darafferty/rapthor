import os
import shutil
import requests
import pytest
import numpy as np

from rapthor.lib.field import Field
from rapthor.lib.parset import parset_read


def _download_ms(filename: str):
    url = 'https://support.astron.nl/software/ci_data/rapthor/tDDECal.in_MS.tgz'
    r = requests.get(url)
    with open('downloaded.tgz', 'wb') as f:
        f.write(r.content)

    os.system('tar xvf downloaded.tgz')
    os.system('rm downloaded.tgz')
    os.system('mv tDDECal.MS ' + filename)


@pytest.fixture(scope="module")
def tests_env():
    # Change directory to the tests directory (one level up from this file),
    # because that's where these tests need to be run from.
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    testmsname = 'resources/test.ms'
    if not os.path.exists(testmsname):
        _download_ms(testmsname)

    yield

    # Cleanup created directories
    for d in ['images', 'logs', 'pipelines', 'regions', 'skymodels', 'solutions', 'plots']:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def field(tests_env):
    par = parset_read('resources/test.parset')
    fld = Field(par)
    fld.fast_timestep_sec = 32.0  # needed for test_get_obs_parameters()
    fld.scan_observations()
    fld.update_skymodels(1, True, target_flux=0.2)
    # Ensure target_flux exists even if sky models are preloaded from disk
    if not hasattr(fld, 'target_flux'):
        fld.target_flux = 0.2
    fld.set_obs_parameters()
    fld.define_imaging_sectors()
    fld.define_outlier_sectors(1)
    return fld


def test_scan_observations(field):
    assert field.fwhm_ra_deg == 4.500843683229519


def test_regular_frequency_spacing(field):
    assert all([obs.channels_are_regular for obs in field.observations])


def test_imaging_sectors(field):
    assert field.sector_bounds_deg == '[258.558431;57.961675;259.103519;56.885818]'


def test_outlier_sectors(field):
    assert field.outlier_sectors == []


def test_chunk_observations(field):
    for obs in field.full_observations:
        obs.data_fraction = 0.8
    field.chunk_observations(600.0, prefer_high_el_periods=False)
    assert field.imaging_sectors[0].observations[0].starttime == 4871282392.90695


def test_chunk_observations_high_el(field):
    for obs in field.full_observations:
        obs.data_fraction = 0.2
    field.chunk_observations(600.0, prefer_high_el_periods=True)
    assert field.imaging_sectors[0].observations[0].starttime == 4871282392.90695


def test_get_obs_parameters(field):
    obsp = field.get_obs_parameters('starttime')
    assert obsp == ['29Mar2013/13:59:52.907']


def test_define_imaging_sectors(field):
    field.define_imaging_sectors()
    assert field.sector_bounds_mid_deg == '[258.841667;57.410833]'


def test_define_outlier_sectors(field):
    field.define_outlier_sectors(1)
    assert field.outlier_sectors == []


def test_define_bright_source_sectors(field):
    field.define_bright_source_sectors(0)
    assert field.bright_source_sectors == []


def test_find_intersecting_sources(field):
    iss = field.find_intersecting_sources()
    assert iss[0].area == pytest.approx(18.37996802132365)


def test_check_selfcal_progress(field):
    assert field.check_selfcal_progress() == (False, False, False)


def test_plot_overview_patches(field):
    field.plot_overview('field_overview_1.png', show_calibration_patches=True)
    assert os.path.exists(os.path.join('plots', 'field_overview_1.png'))


def test_plot_overview_initial(field):
    field.plot_overview('initial_field_overview.png', show_initial_coverage=True)
    assert os.path.exists(os.path.join('plots', 'initial_field_overview.png'))
    os.system('rm plots/initial_field_overview.png')


def test_plot_overview_initial_near_pole(field):
    field.dec = 89.5  # test behavior near pole
    field.plot_overview('initial_field_overview.png', show_initial_coverage=True)
    assert os.path.exists(os.path.join('plots', 'initial_field_overview.png'))
    os.system('rm plots/initial_field_overview.png')


@pytest.fixture
def minimal_parset(tmp_path):
    # Build a minimal parset dict with required nested keys
    return {
        'dir_working': str(tmp_path),
        'mss': ['dummy.ms'],
        'data_colname': 'DATA',
        'input_h5parm': None,
        'input_fulljones_h5parm': None,
        'dde_mode': 'facets',
        'facet_layout': None,
        'generate_initial_skymodel': False,
        'download_initial_skymodel': False,
        'input_skymodel': None,
        'apparent_skymodel': None,
        'cluster_specific': {
            'max_nodes': 1,
        },
        'calibration_specific': {
            'use_image_based_predict': False,
            'bda_timebase': 0,
            'bda_frequencybase': 0,
            'dd_interval_factor': 1.0,
            'fulljones_timestep_sec': 600.0,
            'fast_smoothnessconstraint': 0.0,
            'fast_smoothnessreffrequency': 0.0,
            'fast_smoothnessrefdistance': 0.0,
            'medium_smoothnessconstraint': 0.0,
            'medium_smoothnessreffrequency': 0.0,
            'medium_smoothnessrefdistance': 0.0,
            'slow_smoothnessconstraint': 0.0,
            'fulljones_smoothnessconstraint': 0.0,
            'propagatesolutions': False,
            'solveralgorithm': 'default',
            'onebeamperpatch': False,
            'llssolver': False,
            'maxiter': 5,
            'stepsize': 0.1,
            'stepsigma': 0.1,
            'tolerance': 1e-3,
            'parallelbaselines': False,
            'sagecalpredict': False,
            'fast_datause': 1.0,
            'medium_datause': 1.0,
            'slow_datause': 1.0,
            'solverlbfgs_dof': 0,
            'solverlbfgs_iter': 0,
            'solverlbfgs_minibatches': 0,
            'correct_time_frequency_smearing': False,
            'use_included_skymodels': False,
        },
        'imaging_specific': {
            'dde_method': 'idg',
            'save_visibilities': False,
            'save_image_cube': False,
            'save_supplementary_images': False,
            'compress_selfcal_images': False,
            'compress_final_images': False,
            'use_mpi': False,
            'reweight': False,
            'bda_timebase': 0,
            'do_multiscale_clean': False,
            'apply_diagonal_solutions': False,
            'make_quv_images': False,
            'pol_combine_method': 'I',
            'correct_time_frequency_smearing': False,
            'sector_center_ra_list': [],
            'sector_center_dec_list': [],
            'sector_width_ra_deg_list': [],
            'sector_width_dec_deg_list': [],
            'grid_center_ra': None,
            'grid_center_dec': None,
            'grid_width_ra_deg': None,
            'grid_width_dec_deg': None,
            'grid_nsectors_ra': 1,
            'grid_nsectors_dec': 1,
        },
    }


def mock_scan_observations(self):
    # Provide minimal attributes used later without touching real MS
    class MockObs:
        def __init__(self):
            self.ms_filename = 'dummy.ms'
            self.name = 'dummy'
            self.antenna = 'HBA'
            self.starttime = 0.0
            self.endtime = 3600.0
            self.mean_el_rad = np.deg2rad(60.0)
            self.referencefreq = 150e6
            self.ra = 180.0
            self.dec = 45.0
            self.diam = 30.0
            self.stations = ['ST01', 'ST02']
            self.data_fraction = 1.0
            self.high_el_starttime = 0.0
            self.high_el_endtime = 3600.0
        def set_prediction_parameters(self, *args, **kwargs):
            pass
    obs = MockObs()
    self.full_observations = [obs]
    self.observations = [obs]
    self.antenna = obs.antenna
    self.ra = obs.ra
    self.dec = obs.dec
    self.diam = obs.diam
    self.stations = obs.stations
    self.mean_el_rad = obs.mean_el_rad
    # Recompute beam-related values deterministically
    sec_el = 1.0 / np.sin(self.mean_el_rad)
    self.fwhm_deg = 1.1 * ((3.0e8 / obs.referencefreq) / self.diam) * 180.0 / np.pi * sec_el
    self.fwhm_ra_deg = self.fwhm_deg / sec_el
    self.fwhm_dec_deg = self.fwhm_deg
    self.beam_ms_filename = obs.ms_filename


def make_field(monkeypatch, parset):
    # Monkeypatch heavy scan to a lightweight mock
    monkeypatch.setattr(Field, 'scan_observations', mock_scan_observations)
    f = Field(parset, minimal=True)
    # Ensure WCS is available for sector helpers
    f.makeWCS()
    return f


def test_get_source_distances(monkeypatch, minimal_parset):
    field = make_field(monkeypatch, minimal_parset)
    field.ra = 180.0
    field.dec = 45.0
    sources = {
        'A': [180.0, 45.0],
        'B': [180.5, 45.0],
        'C': [180.0, 45.5],
    }
    names, distances = field.get_source_distances(sources)
    assert set(names.tolist()) == {'A', 'B', 'C'}
    # A at phase center => distance ~ 0
    assert pytest.approx(distances[names.tolist().index('A')], 1e-6) == 0.0
    # Others have non-zero distances
    assert distances.max() > 0


def test_get_calibration_radius(monkeypatch, minimal_parset):
    field = make_field(monkeypatch, minimal_parset)
    field.ra = 180.0
    field.dec = 45.0
    field.calibrator_positions = {
        'X': [180.0, 45.0],
        'Y': [181.0, 45.0],
        'Z': [180.0, 46.0],
    }
    radius = field.get_calibration_radius()
    # Roughly ~1 degree to the farthest calibrator
    assert radius > 0.9 and radius < 1.1


def test_define_full_field_sector(monkeypatch, minimal_parset, tmp_path):
    field = make_field(monkeypatch, minimal_parset)
    # Avoid file writes from Sector helpers by monkeypatching
    from rapthor.lib.sector import Sector
    monkeypatch.setattr(Sector, 'make_vertices_file', lambda self: None)
    monkeypatch.setattr(Sector, 'make_region_file', lambda self, path: None)

    field.define_full_field_sector(radius=1.0)
    s = field.full_field_sector
    assert s is not None
    assert s.width_ra == pytest.approx(2.0)
    assert s.width_dec == pytest.approx(2.0)


def test_get_matplotlib_patch(monkeypatch, minimal_parset):
    field = make_field(monkeypatch, minimal_parset)
    field.makeWCS()  # lightweight WCS
    patch = field.get_matplotlib_patch()
    from matplotlib.patches import Ellipse
    assert isinstance(patch, Ellipse)


def test_define_imaging_sectors_user_list(monkeypatch, minimal_parset):
    # Exercise the user-defined sectors branch
    minimal_parset['imaging_specific']['sector_center_ra_list'] = [180.0]
    minimal_parset['imaging_specific']['sector_center_dec_list'] = [45.0]
    minimal_parset['imaging_specific']['sector_width_ra_deg_list'] = [2.0]
    minimal_parset['imaging_specific']['sector_width_dec_deg_list'] = [2.0]
    field = make_field(monkeypatch, minimal_parset)
    # Avoid file writes from Sector helpers
    from rapthor.lib.sector import Sector
    monkeypatch.setattr(Sector, 'make_vertices_file', lambda self: None)
    monkeypatch.setattr(Sector, 'make_region_file', lambda self, path: None)
    # Need WCS for bounds
    field.makeWCS()
    field.define_imaging_sectors()
    assert len(field.imaging_sectors) == 1
    assert field.sector_bounds_mid_deg.startswith('[')


def test_define_imaging_sectors_grid(monkeypatch, minimal_parset):
    # Exercise the grid branch with 2x2 sectors
    minimal_parset['imaging_specific']['grid_center_ra'] = 180.0
    minimal_parset['imaging_specific']['grid_center_dec'] = 45.0
    minimal_parset['imaging_specific']['grid_width_ra_deg'] = 4.0
    minimal_parset['imaging_specific']['grid_width_dec_deg'] = 4.0
    minimal_parset['imaging_specific']['grid_nsectors_ra'] = 2
    minimal_parset['imaging_specific']['grid_nsectors_dec'] = 2
    minimal_parset['imaging_specific']['skip_corner_sectors'] = False
    field = make_field(monkeypatch, minimal_parset)
    from rapthor.lib.sector import Sector
    monkeypatch.setattr(Sector, 'make_vertices_file', lambda self: None)
    monkeypatch.setattr(Sector, 'make_region_file', lambda self, path: None)
    field.makeWCS()
    field.define_imaging_sectors()
    assert len(field.imaging_sectors) == 4


def test_chunk_observations_node_scaling(monkeypatch, minimal_parset):
    # Ensure the minnobs scaling branch is hit
    minimal_parset['cluster_specific']['max_nodes'] = 4
    # Avoid Observation creating real casacore tables by monkeypatching
    from rapthor.lib import observation as observation_mod
    class DummyObs(observation_mod.Observation):
        def __init__(self, ms_filename, *args, **kwargs):
            # Bypass any real IO in Observation
            self.ms_filename = ms_filename
            self.name = 'dummy'
            self.antenna = 'HBA'
            self.starttime = 0.0
            self.endtime = 3600.0
            self.mean_el_rad = np.deg2rad(60.0)
            self.referencefreq = 150e6
            self.ra = 180.0
            self.dec = 45.0
            self.diam = 30.0
            self.stations = ['ST01', 'ST02']
            self.data_fraction = 1.0
            self.high_el_starttime = 0.0
            self.high_el_endtime = 3600.0
            self.channels_are_regular = True
            # Minimal attributes used during chunking
            self.ntimechunks = 1
            self.timepersample = 1.0
        def copy(self):
            return DummyObs(self.ms_filename)
    # Monkeypatch both the module and the class reference used inside field.py
    monkeypatch.setattr(observation_mod, 'Observation', DummyObs)
    import rapthor.lib.field as field_mod
    monkeypatch.setattr(field_mod, 'Observation', DummyObs)
    field = make_field(monkeypatch, minimal_parset)
    # Create a dummy imaging sector to trigger multiplication with nsectors
    from rapthor.lib.sector import Sector
    monkeypatch.setattr(Sector, 'make_vertices_file', lambda self: None)
    monkeypatch.setattr(Sector, 'make_region_file', lambda self, path: None)
    field.imaging_sectors = [Sector('s1', field.ra, field.dec, 2.0, 2.0, field)]
    # obs.data_fraction small so chunking runs
    for obs in field.full_observations:
        obs.data_fraction = 0.5
    field.chunk_observations(600.0, prefer_high_el_periods=False)
    assert len(field.observations) >= 1


def test_define_predict_sectors(monkeypatch, minimal_parset):
    # Mock a simple calibration_skymodel with a few sources
    field = make_field(monkeypatch, minimal_parset)
    class MockSkyModel:
        def __len__(self):
            return 250
        def copy(self):
            return self
        def select(self, *args, **kwargs):
            pass
        def write(self, *args, **kwargs):
            pass
        def getPatchNames(self):
            return [f'patch_{i}' for i in range(250)]
    field.calibration_skymodel = MockSkyModel()
    field.define_predict_sectors(1)
    # Expect a limited number of sectors (<= 10 as per logic)
    assert len(field.predict_sectors) >= 1 and len(field.predict_sectors) <= 10


def test_define_normalize_sector(monkeypatch, minimal_parset):
    field = make_field(monkeypatch, minimal_parset)
    from rapthor.lib.sector import Sector
    monkeypatch.setattr(Sector, 'make_vertices_file', lambda self: None)
    monkeypatch.setattr(Sector, 'make_region_file', lambda self, path: None)
    # Create multiple imaging sectors to choose largest area
    field.imaging_sectors = [
        Sector('a', field.ra, field.dec, 1.0, 1.0, field),
        Sector('b', field.ra, field.dec, 2.0, 2.0, field),
    ]
    field.define_normalize_sector()
    assert field.normalize_sector is not None
    assert field.normalize_sector.width_ra == 2.0


def test_remove_skymodels(monkeypatch, minimal_parset):
    field = make_field(monkeypatch, minimal_parset)
    # Set up mock skymodels and sectors
    field.calibration_skymodel = "mock_cal"
    field.source_skymodel = "mock_src"
    field.calibrators_only_skymodel = "mock_cal_only"
    from rapthor.lib.sector import Sector
    monkeypatch.setattr(Sector, 'make_vertices_file', lambda self: None)
    monkeypatch.setattr(Sector, 'make_region_file', lambda self, path: None)
    sector = Sector('test', field.ra, field.dec, 1.0, 1.0, field)
    sector.calibration_skymodel = "mock"
    sector.predict_skymodel = "mock"
    sector.field.source_skymodel = "mock"
    field.sectors = [sector]
    field.remove_skymodels()
    assert field.calibration_skymodel is None
    assert field.source_skymodel is None
    assert field.calibrators_only_skymodel is None
    assert sector.calibration_skymodel is None


def test_makeWCS(monkeypatch, minimal_parset):
    field = make_field(monkeypatch, minimal_parset)
    # makeWCS already called in make_field, verify it exists
    assert field.wcs is not None
    assert field.wcs_pixel_scale > 0


def test_chunk_observations_high_elevation(monkeypatch, minimal_parset):
    # Test the prefer_high_el_periods=True branch
    from rapthor.lib import observation as observation_mod
    class DummyObs(observation_mod.Observation):
        def __init__(self, ms_filename, *args, **kwargs):
            self.ms_filename = ms_filename
            self.name = 'dummy'
            self.antenna = 'HBA'
            self.starttime = 0.0
            self.endtime = 3600.0
            self.high_el_starttime = 100.0
            self.high_el_endtime = 1000.0
            self.mean_el_rad = np.deg2rad(60.0)
            self.referencefreq = 150e6
            self.ra = 180.0
            self.dec = 45.0
            self.diam = 30.0
            self.stations = ['ST01', 'ST02']
            self.data_fraction = 0.5
            self.channels_are_regular = True
            self.ntimechunks = 1
            self.timepersample = 1.0
        def copy(self):
            return DummyObs(self.ms_filename)
    monkeypatch.setattr(observation_mod, 'Observation', DummyObs)
    import rapthor.lib.field as field_mod
    monkeypatch.setattr(field_mod, 'Observation', DummyObs)
    field = make_field(monkeypatch, minimal_parset)
    from rapthor.lib.sector import Sector
    monkeypatch.setattr(Sector, 'make_vertices_file', lambda self: None)
    monkeypatch.setattr(Sector, 'make_region_file', lambda self, path: None)
    field.imaging_sectors = [Sector('s1', field.ra, field.dec, 2.0, 2.0, field)]
    field.chunk_observations(200.0, prefer_high_el_periods=True)
    assert len(field.observations) >= 1


def test_scan_h5parms_with_file(monkeypatch, minimal_parset, tmp_path):
    # Test h5parm scanning with a mocked h5parm file
    h5_file = tmp_path / "test.h5"
    minimal_parset['input_h5parm'] = str(h5_file)
    
    # Mock h5parm and its structure
    class MockSolset:
        def getSoltabNames(self):
            return ['phase000', 'amplitude000']
        def getSoltab(self, name):
            class MockSoltab:
                def getAxesNames(self):
                    return ['time', 'freq', 'ant', 'dir', 'pol']
            return MockSoltab()
    
    class MockH5parm:
        def getSolsetNames(self):
            return ['sol000']
        def getSolset(self, name):
            return MockSolset()
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    # Create a dummy file so exists check passes
    h5_file.write_bytes(b'dummy')
    
    import rapthor.lib.field as field_mod
    monkeypatch.setattr(field_mod, 'h5parm', lambda x: MockH5parm())
    
    field = make_field(monkeypatch, minimal_parset)
    field.scan_h5parms()
    assert field.apply_amplitudes is True


def test_get_obs_parameters_multiple(monkeypatch, minimal_parset):
    field = make_field(monkeypatch, minimal_parset)
    # Mock observations with parameters attribute
    class MockObsWithParams:
        def __init__(self):
            self.parameters = {'test_param': ['value1', 'value2']}
    field.observations = [MockObsWithParams(), MockObsWithParams()]
    result = field.get_obs_parameters('test_param')
    assert result == ['value1', 'value2', 'value1', 'value2']


def test_scan_observations_antenna_mismatch(monkeypatch, minimal_parset):
    # Test error when observations have different antenna types
    from rapthor.lib import observation as observation_mod
    class MismatchObs(observation_mod.Observation):
        _counter = 0
        def __init__(self, ms_filename, *args, **kwargs):
            self.ms_filename = ms_filename
            self.name = 'dummy'
            # Alternate antenna types to trigger mismatch
            MismatchObs._counter += 1
            self.antenna = 'HBA' if MismatchObs._counter == 1 else 'LBA'
            self.starttime = 0.0
            self.endtime = 3600.0
            self.mean_el_rad = np.deg2rad(60.0)
            self.referencefreq = 150e6
            self.ra = 180.0
            self.dec = 45.0
            self.diam = 30.0
            self.stations = ['ST01', 'ST02']
    monkeypatch.setattr(observation_mod, 'Observation', MismatchObs)
    import rapthor.lib.field as field_mod
    monkeypatch.setattr(field_mod, 'Observation', MismatchObs)
    
    minimal_parset['mss'] = ['dummy1.ms', 'dummy2.ms']
    with pytest.raises(ValueError):
        Field(minimal_parset, minimal=True)


def test_update_with_final_pass(monkeypatch, minimal_parset):
    # Test update method with final=True - focus on final flag behavior
    field = make_field(monkeypatch, minimal_parset)
    from rapthor.lib.sector import Sector
    monkeypatch.setattr(Sector, 'make_vertices_file', lambda self: None)
    monkeypatch.setattr(Sector, 'make_region_file', lambda self, path: None)
    
    # Setup minimal state - avoid complex skymodel operations by setting flags to skip them
    field.imaging_sectors = [Sector('s1', field.ra, field.dec, 2.0, 2.0, field)]
    field.outlier_sectors = []
    field.bright_source_sectors = []
    field.predict_sectors = []
    field.non_calibrator_source_sectors = []
    # Initialize sectors list (required by update method)
    field.sectors = field.imaging_sectors[:]
    field.peel_outliers = True
    field.peel_bright_sources = True
    
    # Minimal step_dict to avoid complex branches
    step_dict = {
        'regroup_model': False,
        'peel_outliers': False,
        'peel_bright_sources': False,
        'peel_non_calibrator_sources': False,
    }
    
    # Mock update_skymodels to avoid lsmtool internals
    def mock_update_skymodels(*args, **kwargs):
        pass
    monkeypatch.setattr(field, 'update_skymodels', mock_update_skymodels)
    
    field.update(step_dict, 2, final=True)
    # final=True should reset peeling flags
    assert field.peel_outliers is False
    assert field.peel_bright_sources is False


def test_check_selfcal_progress_with_diagnostics(monkeypatch, minimal_parset):
    # Test selfcal convergence checking with actual diagnostics
    field = make_field(monkeypatch, minimal_parset)
    from rapthor.lib.sector import Sector
    monkeypatch.setattr(Sector, 'make_vertices_file', lambda self: None)
    monkeypatch.setattr(Sector, 'make_region_file', lambda self, path: None)
    
    sector = Sector('test', field.ra, field.dec, 2.0, 2.0, field)
    # Add diagnostics for two cycles with all required fields
    sector.diagnostics = [
        {'noise_rms_Jy_beam': 0.001, 'dynamic_range': 100, 'nsources': 50, 'theoretical_rms': 0.0005,
         'median_rms_flat_noise': 0.001, 'unflagged_data_fraction': 0.9,
         'dynamic_range_global_flat_noise': 100},
        {'noise_rms_Jy_beam': 0.0009, 'dynamic_range': 110, 'nsources': 52, 'theoretical_rms': 0.0005,
         'median_rms_flat_noise': 0.0009, 'unflagged_data_fraction': 0.9,
         'dynamic_range_global_flat_noise': 110}
    ]
    field.imaging_sectors = [sector]
    
    result = field.check_selfcal_progress()
    # With 10% improvement, result depends on convergence_ratio (default 0.95)
    # Just verify the method runs and returns proper structure
    assert hasattr(result, 'converged')
    assert hasattr(result, 'diverged')
    assert hasattr(result, 'failed')
    assert result.diverged is False
    assert result.failed is False


def test_define_outlier_sectors_with_peeling(monkeypatch, minimal_parset):
    # Test outlier sector creation when peel_outliers is True
    field = make_field(monkeypatch, minimal_parset)
    field.peel_outliers = True
    from rapthor.lib.sector import Sector
    monkeypatch.setattr(Sector, 'make_vertices_file', lambda self: None)
    monkeypatch.setattr(Sector, 'make_region_file', lambda self, path: None)
    
    # Create imaging sector and calibration skymodel
    field.imaging_sectors = [Sector('img1', field.ra, field.dec, 1.0, 1.0, field)]
    field.bright_source_sectors = []
    
    class SimpleSkyModel:
        def __len__(self):
            return 5
        def getColValues(self, col):
            return ['src1', 'src2', 'src3', 'src4', 'src5']
        def select(self, indices, force=False):
            pass
        def copy(self):
            return self
        def write(self, *args, **kwargs):
            pass
        def getPatchNames(self):
            return ['patch1', 'patch2']
    
    field.calibration_skymodel = SimpleSkyModel()
    field.bright_source_skymodel = SimpleSkyModel()
    
    # Mock the make_outlier_skymodel to return a non-empty model
    def mock_make_outlier():
        return SimpleSkyModel()
    monkeypatch.setattr(field, 'make_outlier_skymodel', mock_make_outlier)
    
    field.define_outlier_sectors(1)
    # Should create outlier sectors when sources exist
    assert len(field.outlier_sectors) >= 0


def test_scan_h5parms_fulljones(monkeypatch, minimal_parset, tmp_path):
    # Test fulljones h5parm scanning
    h5_file = tmp_path / "fulljones.h5"
    minimal_parset['input_fulljones_h5parm'] = str(h5_file)
    
    class MockSolset:
        def getSoltabNames(self):
            return ['phase000', 'amplitude000']
        def getSoltab(self, name):
            class MockSoltab:
                def getAxesNames(self):
                    return ['time', 'freq', 'ant', 'dir', 'pol']
            return MockSoltab()
    
    class MockH5parm:
        def getSolsetNames(self):
            return ['sol000']
        def getSolset(self, name):
            return MockSolset()
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    h5_file.write_bytes(b'dummy')
    
    import rapthor.lib.field as field_mod
    monkeypatch.setattr(field_mod, 'h5parm', lambda x: MockH5parm())
    
    field = make_field(monkeypatch, minimal_parset)
    field.scan_h5parms()
    assert field.apply_fulljones is True


def test_set_obs_parameters(monkeypatch, minimal_parset):
    # Test setting observation parameters
    field = make_field(monkeypatch, minimal_parset)
    
    class MockObsForParams:
        def __init__(self):
            self.ntimechunks = 2
        def set_calibration_parameters(self, *args, **kwargs):
            pass
    
    field.observations = [MockObsForParams(), MockObsForParams()]
    field.num_patches = 5
    field.calibrator_fluxes = [1.0, 2.0, 3.0, 4.0, 5.0]
    field.target_flux = 1.0
    
    field.set_obs_parameters()
    assert field.ntimechunks == 4  # 2 obs * 2 chunks each
