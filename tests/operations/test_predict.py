"""
Test cases for the `rapthor.operations.predict` module.
"""
from pathlib import Path

import pytest

import rapthor
from rapthor.operations.predict import PredictDD, PredictDI
from tests.operations.conftest import get_cwl_input_ids


@pytest.fixture
def predict_field(operation_parset, mocker, single_source_sky_model):
    class Field:
        def __init__(self, parset):
            self.parset = parset

            self.imaging_sectors = []
            self.bright_source_sectors = []
            self.outlier_sectors = []
            self.predict_sectors = []
            self.sectors = []

            self.observations = []

            self.data_colname = "DATA"
            self.h5parm_filename = "h5.parm"
            self.normalize_h5parm = "norm.h5"

            self.apply_amplitudes = False
            self.apply_normalizations = False

            self.onebeamperpatch = True
            self.sagecalpredict = False
            self.correct_smearing_in_calibration = False

    return Field(operation_parset)

class TestPredictDD:
    @pytest.mark.parametrize(
    "peel_outliers, has_outlier_sector, expected_sectors, expect_outlier_removed",
    [
        (True, True, 1, True),    # outlier removed
        (True, False, 1, False),  # nothing removed
        (False, True, 2, False),  # finalize does NOT filter unless enabled
    ],
)
    def test_predict_dd_finalize(
        self, predict_field,
        observation, 
        mocker,
        peel_outliers,
        has_outlier_sector,
        expected_sectors,
        expect_outlier_removed,
    ):
        field = predict_field
        field.peel_outliers = peel_outliers

        observation.ms_field = "field1"
        field.observations = [observation]

        good_sector = mocker.Mock(is_outlier=False, observations=[observation])

        sectors = [good_sector]

        if has_outlier_sector:
            bad_sector = mocker.Mock(is_outlier=True, observations=[observation])
            sectors.append(bad_sector)
            field.outlier_sectors = [bad_sector]
        else:
            field.outlier_sectors = []

        field.sectors = sectors

        field.bright_source_sectors = []
        field.imaging_sectors = []
        field.reweight = False

        predict = PredictDD(field, index=1)

        mocker.patch(
            "rapthor.lib.operation.Operation.finalize",
            return_value=None
        )

        # Act
        predict.finalize()

        # Assert: sector filtering
        assert len(field.sectors) == expected_sectors

        if expect_outlier_removed:
            assert all(not s.is_outlier for s in field.sectors)

        # Always expected side effects
        if peel_outliers:
            assert field.outlier_sectors == []
            assert field.imaged_sources_only is True


        # Observation updates always happen when peel_outliers=True
        if peel_outliers:
            assert observation.infix == ""
            assert observation.ms_filename.endswith(observation.ms_field)
            assert observation.ms_imaging_filename == observation.ms_filename

        assert Path(predict.done_file).exists()


class TestPredictDI:

    @pytest.mark.parametrize(
        "obs_name_matches, obs_starttime_matches, expect_filename_set",
        [
            (True, True, True),
            (False, True, False),
            (True, False, False),
        ],
    )
    def test_predict_di_finalize(
        self,
        predict_field,
        observation,
        sector,
        obs_name_matches,
        obs_starttime_matches,
        expect_filename_set,
    ):
        field = predict_field

        # field observation
        observation.name = "obs1"
        observation.starttime = 100
        field.observations = [observation]

        # sector observation (this is what DI iterates over)
        sector_obs = sector.observations[0]
        sector_obs.ms_predict_di = "predict_di.ms"

        sector_obs.name = observation.name if obs_name_matches else "other_obs"
        sector_obs.starttime = observation.starttime if obs_starttime_matches else 999

        field.predict_sectors = [sector]
        field.sectors = [sector]

        predict = PredictDI(field, index=1)

        # Act
        predict.finalize()

        # Assert
        if expect_filename_set:
            assert observation.ms_predict_di_filename is not None
            assert observation.ms_predict_di_filename.endswith("predict_di.ms")
        else:
            assert getattr(observation, "ms_predict_di_filename", None) is None

        assert Path(predict.done_file).exists()


class TestPredict:
    @pytest.mark.parametrize(
        "mode, batch_system, max_cores, expected_cores",
        [
            ("dd", "some_other_system", 42, 42),
            ("dd", "slurm", 42, None),
            ("di", "some_other_system", 42, 42),
            ("di", "slurm", 42, None),
            
        ],
)
    def test_set_parset_parameters(
        self, predict_field, mode, batch_system, max_cores, expected_cores
    ):
        # Arrange
        predict_field.parset["cluster_specific"]["batch_system"] = batch_system
        predict_field.parset["cluster_specific"]["max_cores"] = max_cores
        predict_field.rapthor_pipeline_dir = "/tmp/pipeline"

        predict = PredictDD(predict_field, index = 1) if mode == "dd" else PredictDI(predict_field, index = 1)
        # Act
        predict.set_parset_parameters()
    
        # Assert
        rapthor_pipeline_path = Path(rapthor.__file__).parent / "pipeline"

        assert predict.parset_parms["rapthor_pipeline_dir"] == str(rapthor_pipeline_path)
        assert predict.parset_parms["max_cores"] == expected_cores

        
    @pytest.mark.parametrize(
        "mode, reweight, peel_outliers, peel_bright_sources",
        [
            ("dd", False, False, False),
            ("dd", True, False, False),
            ("dd", False, True, False),
            ("dd", False, False, True),
            ("di", False, False, False),
            ("di", True, False, False),
            ("di", False, True, False),
            ("di", False, False, True),
        ],
    )
    def test_set_input_parameters(
        self,
        predict_field,
        mode,
        reweight,
        peel_outliers,
        peel_bright_sources,
    ):
        field = predict_field

        field.reweight = reweight
        field.peel_outliers = peel_outliers
        field.peel_bright_sources = peel_bright_sources

        predict = PredictDD(field, index=1) if mode == "dd" else PredictDI(field, index=1)
        predict.set_input_parameters()

        
        rapthor_pipeline_dir = str(Path(rapthor.__file__).parent / "pipeline")
        template_parset_parms = {
                "reweight": reweight,
                "peel_outliers": peel_outliers,
                "peel_bright_sources": peel_bright_sources,
                "max_cores": None,
                "rapthor_pipeline_dir": rapthor_pipeline_dir
        }
        expected_cwl_ids = get_cwl_input_ids("predict_pipeline.cwl", template_parset_parms) if mode == "dd" else get_cwl_input_ids("predict_di_pipeline.cwl", template_parset_parms)
        input_parms_keys = set(predict.input_parms.keys())
        assert expected_cwl_ids.issubset(input_parms_keys), f"input_parms is missing CWL inputs: {expected_cwl_ids - input_parms_keys}"