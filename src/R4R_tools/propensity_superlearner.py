import numpy as np
import pandas as pd

from pytmle.get_initial_estimates import fit_propensity_super_learner


def estimate_propensity(
    df,
    col_group="group",
    calibration_method="isotonic",
    base_learners=None,
    cv_folds=8,
):
    """Estimate treatment propensity scores with a pyTMLE super learner."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("df must contain at least one row.")

    if not isinstance(col_group, str) or not col_group:
        raise TypeError("col_group must be a non-empty string.")

    if col_group not in df.columns:
        raise ValueError(f"Treatment column {col_group!r} was not found in df.")

    if calibration_method not in {"isotonic", "sigmoid", None}:
        raise ValueError(
            "calibration_method must be 'isotonic', 'sigmoid', or None."
        )

    if isinstance(cv_folds, bool) or not isinstance(cv_folds, (int, np.integer)):
        raise TypeError("cv_folds must be an integer.")

    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    if base_learners is not None:
        if not isinstance(base_learners, (list, tuple)):
            raise TypeError("base_learners must be a list, tuple, or None.")
        if not base_learners:
            raise ValueError("base_learners must not be empty.")

    group = df[col_group]

    if group.isna().any():
        raise ValueError(f"Treatment column {col_group!r} contains missing values.")

    if set(group.unique()) != {0, 1}:
        raise ValueError(
            f"Treatment column {col_group!r} must contain both 0 and 1."
        )

    group_counts = group.value_counts()
    if cv_folds > group_counts.min():
        raise ValueError(
            f"cv_folds={cv_folds} exceeds the smallest treatment-group size "
            f"({group_counts.min()})."
        )

    try:
        X = df.drop(columns=[col_group]).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All covariates must be numeric.") from exc

    if X.shape[1] == 0:
        raise ValueError("At least one covariate is required.")

    if not np.isfinite(X).all():
        raise ValueError("Covariates must not contain missing or infinite values.")

    propensity_scores, _, _ = fit_propensity_super_learner(
        X,
        group.to_numpy(),
        base_learners=base_learners,
        verbose=0,
        cv_folds=cv_folds,
        return_model=False,
        calibration_method=calibration_method,
    )

    propensity_scores = np.asarray(propensity_scores, dtype=float).reshape(-1)

    if propensity_scores.shape != (len(df),):
        raise RuntimeError(
            "The propensity learner returned an unexpected number of scores."
        )

    if not np.isfinite(propensity_scores).all():
        raise RuntimeError(
            "The propensity learner returned missing or infinite scores."
        )

    if ((propensity_scores < 0) | (propensity_scores > 1)).any():
        raise RuntimeError(
            "The propensity learner returned probabilities outside [0, 1]."
        )

    return propensity_scores