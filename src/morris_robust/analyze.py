# ---
# created: 27 Sep 2026
# author: kaedonkers, GPT6.0
# modified: 27 Sep 2026
# ---

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from SALib.analyze import morris as salib_morris
from SALib.util import ResultDict, _compute_delta, handle_seed


class NanPolicy(str, Enum):
    RAISE = "raise"
    DROP_TRAJECTORIES = "drop_trajectories"
    DROP_SAMPLES = "drop_samples"


class NanOutputError(ValueError):
    def __init__(self, failed_samples: list[dict[str, Any]]) -> None:
        self.failed_samples = failed_samples
        rows = [sample["row_index"] for sample in failed_samples]
        super().__init__(f"Non-finite model outputs at X{rows}.")

@dataclass
class MorrisResult:
    salib_result: ResultDict
    summary: pd.DataFrame
    diagnostics: Diagnostics

    def to_salib(self) -> ResultDict:
        return self.salib_result

@dataclass
class Diagnostics:
    nan_policy: NanPolicy
    n_outputs: int
    n_finite_outputs: int
    output_validity_ratio: float
    n_trajectories: int
    failed_samples: list[dict[str, Any]]
    n_retained_trajectories: int | None = None
    n_dropped_trajectories: int | None = None
    retained_trajectory_ids: list[int] | None = None
    dropped_trajectory_ids: list[int] | None = None
    per_input_sample_sizes: dict[str, int] | None = None
    per_input_validity_ratio: dict[str, float] | None = None

    def print(self) -> None:
        print(f"NaN policy: {self.nan_policy}")
        print(f"Number of outputs: {self.n_outputs}")
        print(f"Number of finite outputs: {self.n_finite_outputs}")
        print(f"Output validity ratio: {self.output_validity_ratio:.2f}")
        print(f"Number of trajectories: {self.n_trajectories}")
        print(f"Number of retained trajectories: {self.n_retained_trajectories}")
        print(f"Number of dropped trajectories: {self.n_dropped_trajectories}")
        print(f"Retained trajectory IDs: {self.retained_trajectory_ids}")
        print(f"Dropped trajectory IDs: {self.dropped_trajectory_ids}")
        print(f"Failed samples: {self.failed_samples}")
        print(f"Per-input sample sizes: {self.per_input_sample_sizes}")
        print(f"Per-input validity ratio: {self.per_input_validity_ratio}")


def analyze(
    problem: dict,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    nan_policy: NanPolicy | str = NanPolicy.RAISE,
    num_resamples: int = 100,
    conf_level: float = 0.95,
    scaled: bool = False,
    print_to_console: bool = False,
    num_levels: int = 4,
    seed: int | np.random.Generator | None = None,
    **kwargs: Any,
) -> MorrisResult:
    """
    Run SALib Morris analysis with explicit handling of failed outputs.
    
    NaN-containing analyses support ungrouped, scalar-output designs. 
    With ``drop_samples``, an elementary effect is retained only when both output
    endpoints are finite. 
    Scaled analysis with failed outputs is not supported.

    Parameters
    ----------
    problem : dict
        The SALib problem definition.
    X : np.ndarray
        The input samples.
    Y : np.ndarray
        The model outputs corresponding to the input samples.
    nan_policy : NanPolicy or str, optional
        Policy for handling NaN outputs. Options are 'raise', 'drop_trajectories', or 'drop_samples'.
    num_resamples : int, optional
        Number of resamples for bootstrapping.
    conf_level : float, optional
        Confidence level for the analysis.
    scaled : bool, optional
        Whether to perform a scaled analysis.
    print_to_console : bool, optional
        Whether to print results to the console.
    num_levels : int, optional
        Number of levels for the Morris design.
    seed : int or np.random.Generator, optional
        Random seed for reproducibility.
    **kwargs : Any
        Additional keyword arguments passed to the SALib Morris analyzer.

    Returns
    -------
    MorrisResult
        The result of the Morris analysis, including the SALib result, summary, and diagnostics.
    """
    # Check argument validity
    try:
        policy = NanPolicy(nan_policy)
    except ValueError as error:
        choices = ", ".join(policy.value for policy in NanPolicy)
        raise ValueError(f"nan_policy must be one of: {choices}.") from error
    if num_resamples < 1:
        raise ValueError("num_resamples must be at least 1.")
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be between 0 and 1.")

    # Check data validity
    X = np.asarray(X)
    Y = np.asarray(Y)
    num_vars = problem.get("num_vars")
    if not isinstance(num_vars, int) or num_vars < 1:
        raise ValueError("problem['num_vars'] must be a positive integer.")
    if X.shape[0] != Y.size:
        raise ValueError("X and Y must contain the same number of samples.")
    if X.ndim != 2 or X.shape[1] != num_vars:
        raise ValueError("X must be a 2-D array with problem['num_vars'] columns.")
    if Y.ndim != 1:
        raise ValueError("Y must be a 1-D array containing one output per sample.")
    
    trajectory_size = num_vars + 1
    if Y.size == 0 or Y.size % trajectory_size:
        raise ValueError(
            f"The number of rows in X and Y must be a positive multiple of {trajectory_size}."
        )

    # Abort if `groups` is specified (only ungrouped problems are supported)
    groups = problem.get("groups")
    if groups is not None and (
        len(groups) != num_vars or len(set(groups)) != num_vars
    ):
        raise NotImplementedError("This wrapper currently supports ungrouped Morris problems only.")

    # Identify failed samples and handle according to the nan_policy
    n_trajectories = Y.size // trajectory_size
    finite_output = np.isfinite(Y)
    failed_samples = _failed_sample_records(X, Y, finite_output, trajectory_size)
    if failed_samples and policy is NanPolicy.RAISE:
        raise NanOutputError(failed_samples)
    if failed_samples and policy is NanPolicy.DROP_SAMPLES and scaled:
        raise ValueError("scaled=True is not supported with drop_samples and non-finite outputs.")

    # Compute diagnostics for samples in trajectories
    diagnostics = _diagnostics(policy, finite_output, n_trajectories, failed_samples)

    # Prepare keyword arguments for SALib's Morris analyzer
    salib_kwargs = {
        "num_resamples": num_resamples,
        "conf_level": conf_level,
        "scaled": scaled,
        "print_to_console": print_to_console,
        "num_levels": num_levels,
        "seed": seed,
        **kwargs
    }

    # Perform analysis based on the nan_policy
    # TODO: Fold diagnostic handling in each if-block, instead of catching at the end.
    
    # No failed samples, proceed with standard SALib Morris analysis
    if not failed_samples:
        result = salib_morris.analyze(problem, X, Y, **salib_kwargs)
        counts = np.full(num_vars, n_trajectories, dtype=int)
    
    elif policy is NanPolicy.DROP_TRAJECTORIES:
        # Identify valid trajectories where all outputs are finite.
        valid_trajectories = finite_output.reshape(-1, trajectory_size).all(axis=1)
        
        # Repeat the validity of each trajectory for all its constituent samples.
        kept_rows = np.repeat(valid_trajectories, trajectory_size)
        
        # Keep only the rows corresponding to valid trajectories.
        if not valid_trajectories.any():
            raise ValueError("No complete Morris trajectories remain after excluding failed runs.")
        warnings.warn(
            f"Excluded {(~valid_trajectories).sum()} trajectories containing non-finite outputs. "
            f"This may affect any space-filling properties of the sampling design. "
            f"For more details, check the diagnostics.",
            UserWarning,
            stacklevel=2,
        )
        
        # Perform the analysis on valid trajectories using original SALib Morris analyzer
        result = salib_morris.analyze(problem, X[kept_rows], Y[kept_rows], **salib_kwargs)
        
        # Update diagnostics with the number of retained and dropped trajectories.
        counts = np.full(num_vars, int(valid_trajectories.sum()), dtype=int)
        diagnostics.dropped_trajectory_ids = np.flatnonzero(~valid_trajectories).tolist()
        diagnostics.retained_trajectory_ids = np.flatnonzero(valid_trajectories).tolist()
     
    # nan_policy: DROP_SAMPLES
    # Handle the case where the nan_policy is not DROP_TRAJECTORIES and failed samples exist.
    else:
        # Identify and summarize the elementary effects for valid samples only.
        elementary_effects, valid_effects = _sample_effects(
            problem, X, Y, finite_output, trajectory_size, num_levels
        )
        counts = valid_effects.sum(axis=1)

        # Summarize the elementary effects for the valid samples using custom summarization function.
        result = _summarize_effects(
            elementary_effects,
            valid_effects,
            problem["names"],
            num_resamples,
            conf_level,
            seed,
        )

        # Update diagnostics.
        diagnostics.retained_trajectory_ids = np.flatnonzero(valid_effects.any(axis=0)).tolist()
        diagnostics.dropped_trajectory_ids = sorted(
            set(range(n_trajectories)) - set(diagnostics.retained_trajectory_ids)
        )
        diagnostics.per_input_sample_sizes = dict(zip(problem["names"], counts.tolist()))
        
        # Warn the user about the excluded samples with non-finite outputs.
        warnings.warn(
            f"Excluded {len(failed_samples)} samples with non-finite outputs. "
            f"This may bias analysis results. "
            f"For more details, check the diagnostics.",
            UserWarning,
            stacklevel=2,
        )

        # Warn the user if any inputs have fewer than two valid elementary effects.
        if np.any(counts < 2):
            warnings.warn(
                "Inputs with fewer than two valid elementary effects have undefined "
                "sigma and mu_star_conf statistics.",
                UserWarning,
                stacklevel=2,
            )
        if print_to_console:
            print(result.to_df())

    # Ensure diagnostics have default values if not already set.
    if diagnostics.retained_trajectory_ids is None:
        diagnostics.retained_trajectory_ids = list(range(n_trajectories))
    if diagnostics.dropped_trajectory_ids is None:
        diagnostics.dropped_trajectory_ids = []
    diagnostics.n_retained_trajectories = len(diagnostics.retained_trajectory_ids)
    diagnostics.n_dropped_trajectories = len(diagnostics.dropped_trajectory_ids)
    diagnostics.per_input_sample_sizes = dict(zip(problem["names"], counts.tolist()))
    diagnostics.per_input_validity_ratio = dict(
        zip(problem["names"], (counts / n_trajectories).tolist())
    )
    
    # Generate the summary frame for the final results.
    summary = _summary_frame(result, counts, n_trajectories)

    # Return the final MorrisResult object containing the result, summary, and diagnostics.
    return MorrisResult(result, summary, diagnostics)


def _failed_sample_records(
    X: np.ndarray, Y: np.ndarray, finite_output: np.ndarray, trajectory_size: int
) -> list[dict[str, Any]]:
    """
    Get records of failed samples with non-finite outputs.

    Parameters
    ----------
    X : np.ndarray
        Input samples array.
    Y : np.ndarray
        Model output array.
    finite_output : np.ndarray
        Boolean array indicating which outputs are finite.
    trajectory_size : int
        Number of samples per trajectory.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries containing information about failed samples.
    """
    records = []
    for row_index in np.flatnonzero(~finite_output):
        records.append(
            {
                "row_index": int(row_index),
                "trajectory_id": int(row_index // trajectory_size),
                "step_index": int(row_index % trajectory_size),
                "X": X[row_index].tolist(),
                "Y": float(Y[row_index]),
            }
        )
    return records


def _diagnostics(
    policy: NanPolicy,
    finite_output: np.ndarray,
    n_trajectories: int,
    failed_samples: list[dict[str, Any]],
) -> Diagnostics:
    """
    Generate Diagnostics object.

    Parameters
    ----------
    policy : NanPolicy
        The policy for handling NaN values in the output.
    finite_output : np.ndarray
        Boolean array indicating which outputs are finite.
    n_trajectories : int
        Number of trajectories in the analysis.
    failed_samples : list[dict[str, Any]]
        List of records for failed samples.

    Returns
    -------
    Diagnostics
        An object containing diagnostic information about the analysis.
    """
    return Diagnostics(
        nan_policy=policy,
        n_outputs=int(finite_output.size),
        n_finite_outputs=int(finite_output.sum()),
        output_validity_ratio=float(finite_output.mean()),
        n_trajectories=n_trajectories,
        failed_samples=failed_samples,
    )


def _sample_effects(
    problem: dict,
    X: np.ndarray,
    Y: np.ndarray,
    finite_output: np.ndarray,
    trajectory_size: int,
    num_levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute elementary effects of finite samples.

    Parameters
    ----------
    problem : dict
        The Morris sensitivity analysis problem definition.
    X : np.ndarray
        Input samples array.
    Y : np.ndarray
        Model output array.
    finite_output : np.ndarray
        Boolean array indicating which outputs are finite.
    trajectory_size : int
        Number of samples per trajectory.
    num_levels : int
        Number of levels used in the Morris sampling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the elementary effects array and the valid effects boolean array.
    """
    num_vars = problem["num_vars"]
    n_trajectories = Y.size // trajectory_size
    x_by_trajectory = X.reshape(n_trajectories, trajectory_size, num_vars)
    y_by_trajectory = Y.reshape(n_trajectories, trajectory_size)
    finite_by_trajectory = finite_output.reshape(n_trajectories, trajectory_size)

    # Compute differences between consecutive samples in each trajectory to identify changed inputs.
    differences = np.diff(x_by_trajectory, axis=1)
    changed = differences != 0
    if not np.all(changed.sum(axis=2) == 1):
        raise ValueError("Each Morris transition must change exactly one input.")
    changed_inputs = np.argmax(changed, axis=2)
    if not np.all(np.sort(changed_inputs, axis=1) == np.arange(num_vars)):
        raise ValueError("Each ungrouped Morris trajectory must change every input once.")
    
    # Identify valid transitions where both the current and next outputs are finite.
    # TODO: Could this be determined with np.isnan(differences)?
    valid_transitions = finite_by_trajectory[:, :-1] & finite_by_trajectory[:, 1:]
    valid_effects = np.zeros((num_vars, n_trajectories), dtype=bool)
    for trajectory_id in range(n_trajectories):
        valid_effects[changed_inputs[trajectory_id], trajectory_id] = valid_transitions[
            trajectory_id
        ]
    
    # Replace non-finite outputs with zeros to avoid propagating NaNs in the elementary effects calculation.
    finite_Y = np.where(finite_output, Y, 0.0)
    elementary_effects = salib_morris._compute_elementary_effects(
        X,
        finite_Y,
        trajectory_size,
        _compute_delta(num_levels),
        scaling=False,
    )
    
    # Discard elementary effects with invalid transitions
    elementary_effects = np.where(valid_effects, elementary_effects, np.nan)
    return elementary_effects, valid_effects


def _summarize_effects(
    elementary_effects: np.ndarray,
    valid_effects: np.ndarray,
    names: list[str],
    num_resamples: int,
    conf_level: float,
    seed: int | np.random.Generator | None,
) -> ResultDict:
    """
    Populate a ResultDict with manually calculated summary statistics of finite samples.

    Parameters
    ----------
    elementary_effects : np.ndarray
        Array of elementary effects for each parameter and trajectory.
    valid_effects : np.ndarray
        Boolean array indicating which elementary effects are valid.
    names : list[str]
        List of parameter names.
    num_resamples : int
        Number of resamples to use for estimating the confidence interval of mu_star.
    conf_level : float
        Confidence level for the mu_star confidence interval.
    seed : int | np.random.Generator | None
        Random seed or generator for resampling.

    Returns
    -------
    ResultDict
        Dictionary containing the summarized Morris sensitivity analysis results.
    """
    rng = handle_seed(seed)
    n_parameters = elementary_effects.shape[0]
    output = ResultDict((key, [None] * n_parameters) for key in ("names", "mu", "mu_star", "sigma", "mu_star_conf"))
    output["names"] = names
    for parameter in range(n_parameters):
        effects = elementary_effects[parameter, valid_effects[parameter]]
        n_effects = effects.size
        if n_effects == 0:
            output["mu"][parameter] = np.nan
            output["mu_star"][parameter] = np.nan
            output["sigma"][parameter] = np.nan
            output["mu_star_conf"][parameter] = np.nan
            continue

        output["mu"][parameter] = float(np.mean(effects))
        output["mu_star"][parameter] = float(np.mean(np.abs(effects)))
        output["sigma"][parameter] = (
            float(np.std(effects, ddof=1)) if n_effects > 1 else np.nan
        )
        if n_effects < 2:
            output["mu_star_conf"][parameter] = np.nan
            continue

        resample_indices = rng.integers(n_effects, size=(num_resamples, n_effects))
        resampled_mu_star = np.mean(np.abs(effects[resample_indices]), axis=1)
        output["mu_star_conf"][parameter] = float(
            norm.ppf(0.5 + conf_level / 2) * np.std(resampled_mu_star, ddof=1)
        )
    return output


def _summary_frame(
    result: ResultDict, counts: np.ndarray, n_trajectories: int
) -> pd.DataFrame:
    """
    Augment the Morris screening results dataframe with sample counts and validity ratios.

    Parameters
    ----------
    result : ResultDict
        The result of the Morris sensitivity analysis.
    counts : np.ndarray
        Array containing the number of valid samples for each parameter.
    n_trajectories : int
        Total number of trajectories used in the analysis.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the original Morris results augmented with sample counts and validity ratios.
    """
    frame = result.to_df().copy()
    frame["n_samples"] = counts
    frame["validity_ratio"] = counts / n_trajectories
    return frame