# ---
# created: 25 Sep 2026
# author: kaedonkers, GPT6.0
# modified: 27 Sep 2026
# ---

import numpy as np
import pytest

def ishigami(X):
    """
    Ishigami function, a common test function for sensitivity analysis.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n_samples, 3), where each row represents a sample with three input variables.

    Returns
    -------
    np.ndarray
        Output array of shape (n_samples,), containing the function values for each input sample.
    """
    if X.shape[1] < 3:
        raise ValueError("Input X must have at least 3 columns for the Ishigami function.")
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + 7.0 * np.sin(x2) ** 2 + 0.1 * x3**4 * np.sin(x1)

@pytest.fixture
def problem():
    return {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[0.0, np.pi]] * 3,
    }

@pytest.fixture
def clean_design(problem):
    """
    Sample matrix and outputs with no failures, fixed seed.
    """
    from SALib.sample import morris
    X = morris.sample(problem, N=100, num_levels=4, seed=42)
    return X, ishigami(X)

