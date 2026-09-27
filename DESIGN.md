---
created: 25 Sep 2026
author: kaedonkers
modified: 27 Sep 2026
---

# Design: Graceful handling of failed model runs in Morris screening with `SALib` 

## Original behaviour
As demonstrated in the `README.md`, the original `SALib` Morris screening analysis silently returns `NaN` results.
The authors' likely identified that any statistics with missing values can bias results, and thus `NaN` returns are mathematically consistent.
See [Background](README.md#background) in the `README.md` for more information on `SALib` and Morris screening.

## Proposed alternative behaviour
A model returning `NaN` values for all analysis results unfortunately discards a lot of information.
There may be structure to the missingness e.g. regions of model input-space which return null values.
There may also be some recoverable statistics from only a small number of invalid model outputs, with the caveat that there may be a resulting bias in the results.

Thus, I have proposed a wrapper that requires a user-stated policy about how to deal with non-finite model outputs.
These are:
1. `raise`: Raise an error and abort analysis if any non-finite values detected. Non-finite model inputs are reported with the error.
2. `drop-trajectories`: Exclude any trajectories from the analysis that contain one or more non-finite values, and raise a warning. Diagnostics are tracked to highlight effects of exclusion on statistics.
3. `drop-samples`: Exclude non-finite values from analysis. Raise a warning and capture effects on analysis statistics in diagnostics.

This is achieved by providing an alternative to `SALib.analyse.morris`, which takes the same arguments, filters the resulting data structures, raises any warnings (according to the policy) and conducts Morris analysis using `SALib.analyse.morris` on the remaining samples (if appropriate).
A `MorrisResult` object is returned and contains the original `SALib` results, both as a property (`MorrisResult.salib_result`) and a function (`MorrisResult.to_salib()`).
`MorrisResult.summary` is equivalent to `SALib.analyze.morris.to_df()`, with additional statistics on the number of samples used to calculate statistics.
`MorrisResult.diagnostics` is an object with additional detailed information on statistics and samples leading to non-finite model outputs.


## Shortcomings of new design
For users who rely on `SALib` returning NaN results when any NaN is present, or are not familiar with the underlying statistics that are being calculated, this new approach may be misleading, where results are seen as "oh, it worked - they must be Morris screening results as I expect them".
Warnings are used to highlight this possibility to the user, but users often ignore warnings so this may not be an effective remedy.

The extra step of extracting the original results using `MorrisResult.to_salib()/salib_result` means that this wrapper cannot be hot-swapped for existing uses of `SALib.analyse.morris`.
The change of method from `to_df()` to `summary` breaks expected behaviour.

The code is also poorly commented - a result of the tight timeline, sorry...

## Further work
With more time, I would slim down and simplify the partially vibe-coded content.
From my reading of the maths for Morris screening, I don't see any issues with calculating statistics which account for missingness, but it may be the case that this approach illegitimates the underlying statistics.
I would also test this approach of `NaN`-handling on the other sensitivity analysis methods in `SALib`, where the same problem arises.
I would also harmonise the design of the wrapper to be closer to the original API.

The `sampler/analyser` design in `SALib` useful in its statelessness (e.g. running non-Python models), but brittly relies on array positions for identification of trajectories and inputs.
This could be made more explicit and robust by using a consistent data-model (e.g. dataframe) throughout.
However, this would require a significant redesign of `SALib`.

## Use of AI
Two privacy-focussed AI chatbots (`Lumo`, `Rool`) were used to ideate on approaches to developing an existing solution to this problem from my PhD.
Once a rough plan was hashed out, I used GitHub Copilot with GPT6.0 to plan further, generate code and tests, and review documentation.


