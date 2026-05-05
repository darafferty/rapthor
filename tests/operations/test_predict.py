"""
Test cases for the `rapthor.operations.predict` module.
"""
from pathlib import Path

import pytest

import rapthor
from rapthor.operations.predict import PredictDD, PredictDI
from tests.conftest import observation, outlier_sector, sector
from tests.operations.conftest import get_cwl_input_ids


@pytest.fixture
def predict_field(operation_parset, mocker, single_source_sky_model):
    class Field:
        def __init__(self, parset):
            self.parset = parset

            self.sectors = [sector]
            self.bright_source_sectors = []
            self.outlier_sectors = [outlier_sector]
            self.predict_sectors = []

            self.observations = [observation]

            self.data_colname = "DATA"
            self.h5parm_filename = "h5.parm"
            self.normalize_h5parm = "norm.h5"

            self.apply_amplitudes = False
            self.apply_normalizations = False

            self.onebeamperpatch = True
            self.sagecalpredict = False
            self.correct_smearing_in_calibration = False

            self.reweight = False

    return Field(operation_parset)

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

    @pytest.mark.parametrize(
    "mode, peel_outliers, has_outlier_sector, expected_sectors, expect_outlier_removed",
    [
        ("dd", True, True, 1, True),    # outlier removed
        ("dd", True, False, 1, False),  # nothing removed
        ("dd", False, True, 2, False),  # finalize does NOT filter unless enabled
        ("di", False, False, 1, False), # di case with dummy values
    ],
)
    def test_predict_dd_finalize(
        self, predict_field,
        sector, 
        observation, 
        mode,
        peel_outliers,
        has_outlier_sector,
        expected_sectors,
        expect_outlier_removed,
    ):
        field = predict_field
        field.peel_outliers = peel_outliers
        observation.ms_field = "field1"
        field.observations = [observation]
        
        sector_obs = sector.observations[0]
        sector_obs.name = observation.name
        sector_obs.starttime = observation.starttime
        sector_obs.ms_predict_di = "predict_di.ms"

        if has_outlier_sector:
            field.sectors.append(outlier_sector)
            field.outlier_sectors = [outlier_sector]

        predict = PredictDD(field, index=1) if mode == "dd" else PredictDI(field, index=1)

        predict.finalize()

        if mode == "dd":
            assert len(field.sectors) == expected_sectors

            if expect_outlier_removed:
                assert all(not sector.is_outlier for sector in field.sectors)

            # Always expected side effects
            if peel_outliers:
                assert field.outlier_sectors == []
                assert field.imaged_sources_only is True

            # Observation updates always happen when peel_outliers=True
            if peel_outliers:
                assert observation.infix == ""
                assert observation.ms_filename.endswith(observation.ms_field)
                assert observation.ms_imaging_filename == observation.ms_filename
        if mode == "di":
            assert observation.ms_predict_di_filename.endswith("predict_di.ms")                      
        assert Path(predict.done_file).exists()