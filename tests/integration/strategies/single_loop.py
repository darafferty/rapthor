"""Strategy for testing a single self-calibration loop."""

from tests.integration.strategies.common import make_step

strategy_steps = [make_step(do_calibrate=True, do_image=True)]
