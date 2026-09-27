# morris_robust

A small NaN-aware wrapper around SALib's Morris screening analyzer. 
It makes failed model evaluations visible and lets you choose whether to abort, 
omit their complete trajectories, 
or retain unaffected elementary effects.

## Install
This project is an installable Python package:
```sh
pip install .
```

## Environment
This developer's preferred Python environment manager is `pixi`.
`pixi` will locally install `morris_robust` with the necessary dependencies as follows:

```sh
pixi install
```

Tests can be ran using:

```sh
pixi run tests
```

See Prefix documentation[^pixidocs] if you don't already have `pixi` installed.

[^pixidocs]: https://pixi.prefix.dev/latest/installation/

## Background
`SALib`[^salib] is a popular open-source library for conducting sensitivity analysis (SA) in Python.
SA involves systematically changing model inputs and observing resultant changes in model outputs.
This can be done as a pre-calibration step to identify sensitive inputs on which to focus model calibration;
to identify a parsimonious set of model inputs from which to build a ML surrogate/emulator of the model;
or as a post-calibration robustness check to identify the impacts of input uncertainty on model/emulator output.

Morris screening (MS)[^morris] is a simple and interpretable SA technique which involves traversing a trajectory through input-space, changing the values of the model inputs, $\bm{x}$, one-at-a-time (OAT), and recording the resulting change in a model output, $y$.
Each trajectory consists of one change for each input.
Multiple trajectories are taken at different points in the parameter space, with the aim to traverse a representative sample of the input-space.
The effect of each input change is aggregated across trajectories, with the mean "effect", $\mu_i$, mean effect magnitude, $\mu^*_i$, and standard deviation of the effect, $\sigma_i$, reported for each input, $x_i$.

[^salib]: https://salib.readthedocs.io/
[^morris]: https://en.wikipedia.org/wiki/Morris_method

## Original behaviour
The current `SALib` implementation silently fails when any model evaluation returns a non-finite value (`NaN`, `NA`, `inf`, etc.).

```python
from SALib.sample import morris as ms_sampler
from SALib.analyze import morris as ms_analyser
from tests.conftest import ishigami as model

problem = {
		"num_vars": 3,
		"names": ["x1", "x2", "x3"],
		"bounds": [[0.0, 1.0]] * 3,
}
X = ms_sampler.sample(problem, N=32, seed=42)
Y = model(X)

result = ms_analyser.analyze(problem, X, Y, seed=24)
result.to_df()
#           mu   mu_star     sigma  mu_star_conf
# x1  0.888340  0.888340  0.087606      0.030507
# x2  5.091101  5.091101  1.163925      0.383722
# x3  0.041730  0.041730  0.046623      0.014944

Y[10] = float("nan")  # Inject one failed model evaluation

result_nan = ms_analyser.analyze(problem, X, Y, seed=24)
result_nan.to_df()
#     mu  mu_star  sigma  mu_star_conf
# x0 NaN      NaN    NaN           NaN
# x1 NaN      NaN    NaN           NaN
# x2 NaN      NaN    NaN           NaN
```

## Wrapper behaviour
This wrapper enables users to use an explicit policy for handling non-finite values:
1. `raise`: Raise an error and abort analysis if any non-finite values detected. Non-finite model inputs are reported with the error.
2. `drop-trajectories`: Exclude any trajectories from the analysis that contain one or more non-finite values, and raise a warning. Diagnostics are tracked to highlight effects of exclusion on statistics.
3. `drop-samples`: Exclude non-finite values from analysis. Raise a warning and capture effects on analysis statistics in diagnostics.

The wrapper returns `MorrisResult` from an analysis, which contains 

```python
from SALib.sample import morris as ms_sampler
from morris_robust import analyze
from tests.conftest import ishigami as model

problem = {
		"num_vars": 3,
		"names": ["x1", "x2", "x3"],
		"bounds": [[0.0, 1.0]] * 3,
}
X = ms_sampler.sample(problem, N=32, seed=42)
Y = model(X)

result = analyze(problem, X, Y, seed=24)
result.salib_result.to_df()
#           mu   mu_star     sigma  mu_star_conf
# x1  0.888340  0.888340  0.087606      0.030507
# x2  5.091101  5.091101  1.163925      0.383722
# x3  0.041730  0.041730  0.046623      0.014944
result.summary
#           mu   mu_star     sigma  mu_star_conf  n_samples  validity_ratio
# x1  0.888340  0.888340  0.087606      0.030507         32             1.0
# x2  5.091101  5.091101  1.163925      0.383722         32             1.0
# x3  0.041730  0.041730  0.046623      0.014944         32             1.0

Y[10] = float("nan")  # Inject one failed model evaluation

result_raise = analyze(problem, X, Y, seed=24, nan_policy="raise")
# NanOutputError: Non-finite model outputs at X[10].

result_dropT = analyze(problem, X, Y, seed=24, nan_policy="drop_trajectories")
# UserWarning: Excluded 1 trajectories containing non-finite outputs.
result_dropT.summary
#           mu   mu_star     sigma  mu_star_conf  n_samples  validity_ratio
# x1  0.884083  0.884083  0.085624      0.025178         31         0.96875
# x2  5.051760  5.051760  1.161335      0.394614         31         0.96875
# x3  0.043077  0.043077  0.046757      0.015137         31         0.96875

result_dropS = analyze(problem, X, Y, seed=24, nan_policy="drop_samples")
# UserWarning: Excluded 1 samples with non-finite outputs.
result_dropS.summary
#           mu   mu_star     sigma  mu_star_conf  n_samples  validity_ratio
# x1  0.884083  0.884083  0.085624      0.025178         31         0.96875
# x2  5.091101  5.091101  1.163925      0.387574         32         1.00000
# x3  0.043077  0.043077  0.046757      0.017719         31         0.96875
```
