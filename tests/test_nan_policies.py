# ---
# created: 25 Sep 2026
# author: kaedonkers, GPT6.0
# modified: 27 Sep 2026
# ---

import numpy as np
import pytest
from SALib.analyze import morris
from SALib.util import _compute_delta

from morris_robust import Diagnostics, NanOutputError, analyze

# Analyser tests
@pytest.mark.parametrize("nan_policy", ["raise", "drop_trajectories", "drop_samples"])
def test_noop_on_clean_data(problem, clean_design, nan_policy):
    """
    Test that the analyze function behaves identically to SALib's morris.analyze
    when there are no NaN values in the input data.
    """
    X, Y = clean_design
    expected = morris.analyze(
        problem.copy(), X, Y, num_levels=4, num_resamples=100, seed=42
    )
    wrapped = analyze(
        problem.copy(),
        X,
        Y,
        nan_policy=nan_policy,
        num_levels=4,
        num_resamples=100,
        seed=42,
    )
    assert isinstance(wrapped.diagnostics, Diagnostics)
    actual = wrapped.to_salib()

    np.testing.assert_array_equal(actual["names"], expected["names"])
    for key in ("mu", "mu_star", "sigma", "mu_star_conf"):
        np.testing.assert_array_equal(actual[key], expected[key])

def test_rejects_malformed_output_shapes(problem, clean_design):
    """
    Test that the analyzer rejects malformed output shapes for Y.
    """
    X, Y = clean_design
    with pytest.raises(ValueError, match="1-D"):
        analyze(problem.copy(), X, Y[:, None])
    with pytest.raises(ValueError, match="same number"):
        analyze(problem.copy(), X, Y[:-1])

# Raise policy tests
def test_raise_policy_includes_failed_sample_details(problem, clean_design):
    """
    Test that the 'raise' nan_policy includes detailed information about failed samples.
    """
    X, Y = clean_design
    Y = Y.copy()
    Y[1] = np.nan

    with pytest.raises(NanOutputError) as exc_info:
        analyze(problem.copy(), X, Y, nan_policy="raise")

    failure = exc_info.value.failed_samples[0]
    assert failure["row_index"] == 1
    assert failure["trajectory_id"] == 0
    assert failure["step_index"] == 1
    assert failure["X"] == X[1].tolist()
    assert np.isnan(failure["Y"])

# Drop-trajectories policy tests
def test_drop_trajectories_matches_salib_on_complete_trajectories(
    problem, clean_design
):
    """
    Test that the 'drop_trajectories' nan_policy correctly drops entire trajectories
    containing NaN values and matches the expected results from SALib's morris.analyze.
    """
    X, Y = clean_design
    Y = Y.copy()
    Y[1] = np.nan
    trajectory_size = problem["num_vars"] + 1
    valid_trajectories = np.isfinite(Y).reshape(-1, trajectory_size).all(axis=1)
    valid_rows = np.repeat(valid_trajectories, trajectory_size)

    with pytest.warns(UserWarning, match="Excluded"):
        actual = analyze(
            problem.copy(), X, Y, nan_policy="drop_trajectories", num_levels=4, seed=13
        )
    expected = morris.analyze(
        problem.copy(), X[valid_rows], Y[valid_rows], num_levels=4, seed=13
    )

    for key in ("mu", "mu_star", "sigma", "mu_star_conf"):
        np.testing.assert_array_equal(actual.to_salib()[key], expected[key])
    assert actual.diagnostics.dropped_trajectory_ids == [0]
    assert actual.summary["n_samples"].tolist() == [valid_trajectories.sum()] * 3


def test_drop_trajectories_errors_when_all_trajectories_fail(problem, clean_design):
    """
    Test that the 'drop_trajectories' nan_policy raises an error when all trajectories fail due to NaN values.
    """
    X, Y = clean_design
    Y = np.full_like(Y, np.nan)

    with pytest.raises(ValueError, match="No complete Morris trajectories"):
        analyze(problem.copy(), X, Y, nan_policy="drop_trajectories", num_levels=4)


# Drop-samples policy tests
def test_drop_samples_counts_effects_by_changed_input(problem, clean_design):
    """
    Test that the 'drop_samples' nan_policy correctly drops individual samples
    containing NaN values and counts the effects based on changed inputs.
    """
    X, Y = clean_design
    Y = Y.copy()
    failed_row = 1
    Y[failed_row] = np.nan
    trajectory_size = problem["num_vars"] + 1
    X_by_trajectory = X.reshape(-1, trajectory_size, problem["num_vars"])
    changed = np.diff(X_by_trajectory[0], axis=0) != 0
    affected_inputs = set(np.argmax(changed, axis=1)[
        np.maximum(0, failed_row % trajectory_size - 1):
        min(problem["num_vars"], failed_row % trajectory_size + 1)
    ])

    with pytest.warns(UserWarning, match=r"Excluded \d+ samples"):
        result = analyze(
            problem.copy(), X, Y, nan_policy="drop_samples", num_levels=4, seed=13
        )

    counts = result.summary["n_samples"]
    expected_effects = [[] for _ in problem["names"]]
    for trajectory_id in range(X.shape[0] // trajectory_size):
        start = trajectory_id * trajectory_size
        for step_index in range(problem["num_vars"]):
            left = start + step_index
            right = left + 1
            changed_input = np.flatnonzero(X[right] != X[left])
            assert changed_input.size == 1
            parameter = int(changed_input[0])
            if np.isfinite(Y[left]) and np.isfinite(Y[right]):
                expected_effects[parameter].append(
                    np.sign(X[right, parameter] - X[left, parameter])
                    * (Y[right] - Y[left])
                    / _compute_delta(4)
                )

    for index, name in enumerate(problem["names"]):
        assert counts[name] == 100 - (index in affected_inputs)
        effects = np.asarray(expected_effects[index])
        assert result.summary.loc[name, "mu"] == pytest.approx(np.mean(effects))
        assert result.summary.loc[name, "mu_star"] == pytest.approx(np.mean(np.abs(effects)))
        assert result.summary.loc[name, "sigma"] == pytest.approx(np.std(effects, ddof=1))
    assert result.diagnostics.failed_samples[0]["row_index"] == failed_row
    assert result.diagnostics.per_input_sample_sizes == counts.to_dict()

def test_drop_samples_bootstrap_is_reproducible(problem, clean_design):
    """
    Test that the 'drop_samples' nan_policy produces reproducible bootstrap results when the same seed is used.
    """
    X, Y = clean_design
    Y = Y.copy()
    Y[1] = np.nan
    with pytest.warns(UserWarning):
        first = analyze(
            problem.copy(), X, Y, nan_policy="drop_samples", num_levels=4, seed=31
        ).to_salib()
    with pytest.warns(UserWarning):
        second = analyze(
            problem.copy(), X, Y, nan_policy="drop_samples", num_levels=4, seed=31
        ).to_salib()

    for key in ("mu", "mu_star", "sigma", "mu_star_conf"):
        np.testing.assert_array_equal(first[key], second[key])

