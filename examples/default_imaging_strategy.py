"""
Script that defines the default user processing strategy for imaging only.
Specifying this file as the strategy in the Rapthor parset causes default
Rapthor to use the default behaviour, which is equal to specifying "strategy =
image".

This file is provided to base custom strategies from. See the documentation for
detailed information on each parameter.
"""
# Here we disable calibration, flux-scale normalization, and peeling of
# outliers and bright sources
strategy_steps = [{}]
strategy_steps[0]['do_calibrate'] = False
strategy_steps[0]['do_normalize'] = False
strategy_steps[0]['peel_outliers'] = False
strategy_steps[0]['peel_bright_sources'] = False

# Here we activate imaging and set the imaging strategy parameters
strategy_steps[0]['do_image'] = True
strategy_steps[0]['auto_mask'] = 3.0
strategy_steps[0]['auto_mask_nmiter'] = 2
strategy_steps[0]['channel_width_hz'] = 4e6
strategy_steps[0]['threshisl'] = 3.0
strategy_steps[0]['threshpix'] = 5.0
strategy_steps[0]['max_nmiter'] = 12

# Here we disable the self-calbration convergence check and the
# sky model regrouping (both useful only when there is more
# than one cycle)
strategy_steps[0]['do_check'] = False
strategy_steps[0]['regroup_model'] = False
