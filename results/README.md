# Getting Results

## Before running anything...
Before running any scripts, set the proper absolute paths to `data` and `figures` in `scripts\consts\paths.py`.


## Setting simulation trial parameters
To set the number of desired simulation trials, edit `NUM_TRIALS` in `scripts\consts\trials.py`. Other simulation trial, controller, and estimator parameters can be set in the other modules in `scripts\consts`.


## Running simulation trials
Run either `scripts\run_crazyflie_trials.py` or `scripts\run_fusion_one_trials.py`.


## Getting plots and results
Run `scripts\get_plots.py`.
