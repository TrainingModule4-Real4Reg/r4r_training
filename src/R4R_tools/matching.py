import numpy as np
import pandas as pd
from causalml.match import NearestNeighborMatch


def control_matching(
    data,
    treatment_col="treatment",
    score_cols=None,
    ratio=2,
    caliper=0.05,
    replace=False,
    random_state=None,
):
    """Match treated participants to control participants."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must contain at least one row.")

    if not isinstance(treatment_col, str) or not treatment_col:
        raise TypeError("treatment_col must be a non-empty string.")

    if treatment_col not in data.columns:
        raise ValueError(f"Treatment column {treatment_col!r} was not found.")

    if score_cols is None:
        score_cols = ["propensity_score"]
    elif isinstance(score_cols, str):
        score_cols = [score_cols]
    elif not isinstance(score_cols, (list, tuple)):
        raise TypeError("score_cols must be a string, list, or tuple.")

    if not score_cols:
        raise ValueError("At least one score column is required.")

    if any(not isinstance(column, str) or not column for column in score_cols):
        raise TypeError("score_cols must contain non-empty strings.")

    missing_score_cols = [column for column in score_cols if column not in data.columns]
    if missing_score_cols:
        raise ValueError(
            f"Score column(s) not found: {', '.join(missing_score_cols)}."
        )

    if isinstance(ratio, bool) or not isinstance(ratio, (int, np.integer)):
        raise TypeError("ratio must be an integer.")

    if ratio < 1:
        raise ValueError("ratio must be at least 1.")

    if isinstance(caliper, bool) or not isinstance(caliper, (int, float, np.number)):
        raise TypeError("caliper must be a number.")

    if not np.isfinite(caliper) or caliper < 0:
        raise ValueError("caliper must be finite and non-negative.")

    if not isinstance(replace, (bool, np.bool_)):
        raise TypeError("replace must be a boolean.")

    treatment = data[treatment_col]

    if treatment.isna().any():
        raise ValueError(f"Treatment column {treatment_col!r} contains missing values.")

    if set(treatment.unique()) != {0, 1}:
        raise ValueError(
            f"Treatment column {treatment_col!r} must contain both 0 and 1."
        )

    scores = data[list(score_cols)].apply(pd.to_numeric, errors="coerce")

    if scores.isna().any().any():
        raise ValueError("Score columns must contain numeric, non-missing values.")

    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Score columns must not contain infinite values.")

    matcher = NearestNeighborMatch(
        replace=bool(replace),
        ratio=int(ratio),
        caliper=float(caliper),
        random_state=random_state,
    )

    return matcher.match(
        data=data,
        treatment_col=treatment_col,
        score_cols=list(score_cols),
    )