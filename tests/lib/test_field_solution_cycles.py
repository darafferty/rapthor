"""
Test solution-cycle helpers on the Field model.
"""

from rapthor.lib.field import Field


def _field():
    return object.__new__(Field)


def test_solution_cycle_number_prefers_field_cycle_attribute():
    field = _field()
    field.h5parm_cycle_number = "3"

    assert field.solution_cycle_number("/work/calibrate_1/field.h5", "h5parm_cycle_number") == 3


def test_solution_cycle_number_reads_cycle_from_calibrate_path():
    field = _field()

    assert field.solution_cycle_number("/work/calibrate_4/field.h5", "h5parm_cycle_number") == 4


def test_solution_cycle_number_ignores_di_calibration_paths_by_default():
    field = _field()

    assert (
        field.solution_cycle_number("/work/calibrate_di_4/field.h5", "h5parm_cycle_number") is None
    )


def test_solution_cycle_number_can_read_di_calibration_paths():
    field = _field()

    assert (
        field.solution_cycle_number(
            "/work/calibrate_di_4/field.h5",
            "h5parm_cycle_number",
            include_di_calibration=True,
        )
        == 4
    )
