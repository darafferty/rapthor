"""Shared scientific metadata for supported calibration solve types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationSolveMetadata:
    """Scientific and configuration metadata for one strategy solve type."""

    solution_interval_family: str
    dp3_mode: str
    timestep_key: str
    frequency_step_key: str
    target_timestep_key: str


CALIBRATION_SOLVE_METADATA = {
    "fast_phase": CalibrationSolveMetadata(
        solution_interval_family="fast",
        dp3_mode="scalarphase",
        timestep_key="solint_fast_timestep",
        frequency_step_key="solint_fast_freqstep",
        target_timestep_key="fast_timestep_sec",
    ),
    "medium_phase": CalibrationSolveMetadata(
        solution_interval_family="medium",
        dp3_mode="scalarphase",
        timestep_key="solint_medium_timestep",
        frequency_step_key="solint_medium_freqstep",
        target_timestep_key="medium_timestep_sec",
    ),
    "slow_gains": CalibrationSolveMetadata(
        solution_interval_family="slow",
        dp3_mode="diagonal",
        timestep_key="solint_slow_timestep",
        frequency_step_key="solint_slow_freqstep",
        target_timestep_key="slow_timestep_sec",
    ),
    "full_jones": CalibrationSolveMetadata(
        solution_interval_family="fulljones",
        dp3_mode="fulljones",
        timestep_key="solint_fulljones_timestep",
        frequency_step_key="solint_fulljones_freqstep",
        target_timestep_key="fulljones_timestep_sec",
    ),
}
