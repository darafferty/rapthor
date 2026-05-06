"""
Test cases for the `rapthor.operations.predict` module.
"""
from pathlib import Path
import pytest
import rapthor
from rapthor.operations.predict import Predict
from tests.operations.conftest import get_cwl_input_ids


@pytest.fixture
def predict_field(operation_parset):
    class Field:
        def __init__(self, parset):
            self.parset = parset
            self.sectors = []
            self.imaging_sectors = []
            self.bright_source_sectors = []
            self.outlier_sectors = []
            self.predict_sectors = []
            self.observations = []
            self.data_colname = "DATA"
            self.h5parm_filename = "h5.parm"
            self.normalize_h5parm = "norm.h5"
            self.reweight = False
            self.apply_amplitudes = False
            self.apply_normalizations = False
            self.onebeamperpatch = True
            self.sagecalpredict = False
            self.correct_smearing_in_calibration = False
            self.peel_outliers = False
            self.peel_bright_sources = False

    return Field(operation_parset)


class TestPredict:
    @pytest.mark.parametrize(
        "mode, batch_system, max_cores, expected_cores",
        [
            ("dd", "some_other_system", 42, 42),
            ("dd", "slurm", 42, None),
            ("di", "some_other_system", 42, 42),
            ("di", "slurm", 42, None),
            ("dd", "single_machine", 42, 42),
            ("dd", "slurm_static", 42, None),
            ("di", "single_machine", 42, 42),
            ("di", "slurm_static", 42, None),
        ],
    )
    def test_set_parset_parameters(
        self, predict_field, mode, batch_system, max_cores, expected_cores
    ):
        predict_field.parset["cluster_specific"]["batch_system"] = batch_system
        predict_field.parset["cluster_specific"]["max_cores"] = max_cores

        predict = Predict(mode=mode, field=predict_field, index=1)
        predict.set_parset_parameters()

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
        self, predict_field, mode, reweight, peel_outliers, peel_bright_sources
    ):
        predict_field.reweight = reweight
        predict_field.peel_outliers = peel_outliers
        predict_field.peel_bright_sources = peel_bright_sources

        predict = Predict(mode=mode, field=predict_field, index=1)
        predict.set_input_parameters()

        rapthor_pipeline_dir = str(Path(rapthor.__file__).parent / "pipeline")
        template_parset_parms = {
            "reweight": reweight,
            "peel_outliers": peel_outliers,
            "peel_bright_sources": peel_bright_sources,
            "max_cores": None,
            "rapthor_pipeline_dir": rapthor_pipeline_dir,
        }
        cwl_file = "predict_pipeline.cwl" if mode == "dd" else "predict_di_pipeline.cwl"
        expected_cwl_ids = get_cwl_input_ids(cwl_file, template_parset_parms)
        input_parms_keys = set(predict.input_parms.keys())
        assert expected_cwl_ids.issubset(input_parms_keys), (
            f"input_parms is missing CWL inputs: {expected_cwl_ids - input_parms_keys}"
        )

    @pytest.mark.parametrize(
    "apply_amplitudes, apply_normalizations, expected_steps",
    [
        (False, False, "[fastphase]"),
        (True, False, "[fastphase,slowgain]"),
        (False, True, "[fastphase,normalization]"),
        (True, True, "[fastphase,slowgain,normalization]"),
    ],
)
    def test_set_input_parameters_dp3_applycal_steps(
        self,
        predict_field,
        apply_amplitudes,
        apply_normalizations,
        expected_steps,
    ):
        predict_field.apply_amplitudes = apply_amplitudes
        predict_field.apply_normalizations = apply_normalizations

        predict = PredictDD(predict_field, index=1)
        predict.set_input_parameters()

        assert predict.input_parms["dp3_applycal_steps"] == expected_steps

    @pytest.mark.parametrize(
        "mode, peel_outliers, has_outlier_sector, expected_sectors, expect_outlier_removed",
        [
            ("dd", True, True, 1, True),    # outlier removed when peel_outliers=True
            ("dd", True, False, 1, False),  # no outlier to remove
            ("dd", False, True, 2, False),  # outlier kept when peel_outliers=False
            ("di", False, False, 1, False), # DI: only checks ms_predict_di_filename
        ],
    )
    def test_finalize(
        self,
        predict_field,
        sector,
        outlier_sector,
        observation,
        mode,
        peel_outliers,
        has_outlier_sector,
        expected_sectors,
        expect_outlier_removed,
    ):
        field = predict_field
        field.peel_outliers = peel_outliers
        field.observations = [observation]
        field.sectors = [sector]
        field.predict_sectors = [sector]

        # Align the sector's copied observation with the field observation
        # so finalize() can match them by name/starttime
        sector_obs = sector.observations[0]
        sector_obs.name = observation.name
        sector_obs.starttime = observation.starttime
        sector_obs.ms_field = "sector_field1"
        sector_obs.ms_predict_di = "predict_di.ms"
        observation.ms_field = "field1"

        if has_outlier_sector:
            field.sectors.append(outlier_sector)
            field.outlier_sectors = [outlier_sector]

        predict = Predict(mode=mode, field=predict_field, index=1)
        predict.finalize()

        if mode == "dd":
            assert len(field.sectors) == expected_sectors
            if expect_outlier_removed:
                assert all(not s.is_outlier for s in field.sectors)
            if peel_outliers:
                assert field.outlier_sectors == []
                assert field.imaged_sources_only is True
                assert observation.infix == ""
                assert observation.ms_filename.endswith(observation.ms_field)
                assert sector_obs.ms_imaging_filename == sector_obs.ms_filename

        if mode == "di":
            assert observation.ms_predict_di_filename.endswith("predict_di.ms")

        assert Path(predict.done_file).exists()
