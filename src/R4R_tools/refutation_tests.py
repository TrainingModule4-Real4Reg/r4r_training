"""Simple refutation tests for treatment-effect estimators."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from pytmle import PyTMLE


def random_treatment_refutation(
    data: pd.DataFrame,
    effect_estimator: Callable[[pd.DataFrame], float],
    treatment_col: str = "treatment",
    n_simulations: int = 100,
    random_state: int | None = None,
) -> dict[str, object]:
    """Test an estimated effect after randomly reassigning treatment.

    The observed effect is computed on the original data. The treatment
    column is then permuted repeatedly, the effect is re-estimated, and the
    resulting placebo effects form an empirical null distribution. A useful
    estimator should generally produce an effect more extreme than effects
    obtained after random treatment assignment.

    Parameters
    ----------
    data : pandas.DataFrame
        Data passed to ``effect_estimator``. It is never modified in place.
    effect_estimator : callable
        Function accepting a dataframe and returning one numeric treatment
        effect estimate.
    treatment_col : str, default="treatment"
        Name of the binary treatment column to permute.
    n_simulations : int, default=100
        Number of random treatment assignments used to create the placebo
        distribution. Must be a positive integer.
    random_state : int or None, default=None
        Seed for the random treatment assignments.

    Returns
    -------
    dict
        Dictionary with the observed effect, placebo effects, empirical
        two-sided p-value, and the absolute placebo-effect quantiles. The
        p-value is retained for programmatic use; the notebook focuses on
        visual comparison of the distributions.

    """
    
    if treatment_col not in data.columns:
        raise ValueError(f"Treatment column not found: {treatment_col}")
    if not isinstance(n_simulations, int) or isinstance(n_simulations, bool):
        raise TypeError("n_simulations must be a positive integer")
    if n_simulations < 1:
        raise ValueError("n_simulations must be a positive integer")

    observed_effect = float(effect_estimator(data.copy()))
    rng = np.random.default_rng(random_state)
    placebo_effects = np.empty(n_simulations, dtype=float)

    for simulation in range(n_simulations):
        placebo_data = data.copy()
        placebo_data[treatment_col] = rng.permutation(
            data[treatment_col].to_numpy()
        )
        placebo_effects[simulation] = float(effect_estimator(placebo_data))

    p_value = float(
        (np.count_nonzero(np.abs(placebo_effects) >= abs(observed_effect)) + 1)
        / (n_simulations + 1)
    )

    return {
        "observed_effect": observed_effect,
        "placebo_effects": placebo_effects,
        "p_value": p_value,
        "placebo_abs_quantiles": {
            str(q): float(np.quantile(np.abs(placebo_effects), q))
            for q in (0.5, 0.9, 0.95, 0.99)
        },
    }


def random_treatment_refutation_pytmle(
    data: pd.DataFrame,
    target_times: list[float],
    refutation_time: float,
    col_event_times: str = "time",
    col_event_indicator: str = "status",
    treatment_col: str = "chemo",
    n_simulations: int = 100,
    random_state: int | None = None,
    fit_kwargs: dict | None = None,
) -> dict[str, object]:
    """Run a random-treatment refutation test for a pyTMLE risk difference.

    The pyTMLE model is refit after each random permutation of the treatment
    column. The returned effect is the risk difference at ``refutation_time``.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing treatment, event-time, and event-indicator
        columns. The input dataframe is not modified.
    target_times : list of float
        Times at which pyTMLE estimates counterfactual risks.
    refutation_time : float
        Target time whose risk difference is returned. It must be present in
        ``target_times``.
    col_event_times : str, default="time"
        Name of the event-time column.
    col_event_indicator : str, default="status"
        Name of the event-indicator column.
    treatment_col : str, default="chemo"
        Name of the binary treatment column to permute.
    n_simulations : int, default=100
        Number of random treatment assignments.
    random_state : int or None, default=None
        Seed for the random treatment assignments.
    fit_kwargs : dict or None, default=None
        Additional keyword arguments passed to ``PyTMLE.fit``.

    Returns
    -------
    dict
        Dictionary containing the observed risk difference, placebo effects,
        empirical two-sided p-value, and placebo-effect quantiles.
    """
    if refutation_time not in target_times:
        raise ValueError("refutation_time must be included in target_times")

    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)

    def estimate_risk_difference(refutation_data: pd.DataFrame) -> float:
        tmle = PyTMLE(
            refutation_data,
            col_event_times=col_event_times,
            col_event_indicator=col_event_indicator,
            col_group=treatment_col,
            target_times=target_times,
            g_comp=True,
            evalues_benchmark=False,
        )
        tmle.fit(**fit_kwargs)
        risk_differences = tmle.predict(type="rd", g_comp=True)
        return float(
            risk_differences.loc[
                risk_differences["Time"] == refutation_time, "Pt Est"
            ].iloc[0]
        )

    return random_treatment_refutation(
        data=data,
        effect_estimator=estimate_risk_difference,
        treatment_col=treatment_col,
        n_simulations=n_simulations,
        random_state=random_state,
    )


def random_common_cause_refutation_pytmle(
    data: pd.DataFrame,
    target_times: list[float],
    refutation_time: float,
    col_event_times: str = "time",
    col_event_indicator: str = "status",
    treatment_col: str = "chemo",
    n_simulations: int = 100,
    random_state: int | None = None,
    fit_kwargs: dict | None = None,
) -> dict[str, object]:
    """Run a random-common-cause refutation test for a pyTMLE risk difference.

    A new independent random covariate is added to the data for each
    simulation. pyTMLE is refitted after each addition. The estimated effect
    should remain broadly similar if adding an irrelevant covariate does not
    materially change the analysis.

    Parameters are the same as for :func:`random_treatment_refutation_pytmle`.
    The returned ``refuted_effects`` contain the risk difference after adding
    a different random covariate in each simulation.

    Returns
    -------
    dict
        Dictionary containing the original risk difference and the risk
        differences after adding random common causes.
    """
    if refutation_time not in target_times:
        raise ValueError("refutation_time must be included in target_times")
    if not isinstance(n_simulations, int) or isinstance(n_simulations, bool):
        raise TypeError("n_simulations must be a positive integer")
    if n_simulations < 1:
        raise ValueError("n_simulations must be a positive integer")

    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    rng = np.random.default_rng(random_state)

    def estimate_risk_difference(refutation_data: pd.DataFrame) -> float:
        tmle = PyTMLE(
            refutation_data,
            col_event_times=col_event_times,
            col_event_indicator=col_event_indicator,
            col_group=treatment_col,
            target_times=target_times,
            g_comp=True,
            evalues_benchmark=False,
        )
        tmle.fit(**fit_kwargs)
        risk_differences = tmle.predict(type="rd", g_comp=True)
        return float(
            risk_differences.loc[
                risk_differences["Time"] == refutation_time, "Pt Est"
            ].iloc[0]
        )

    observed_effect = estimate_risk_difference(data.copy())
    refuted_effects = np.empty(n_simulations, dtype=float)

    for simulation in range(n_simulations):
        refutation_data = data.copy()
        refutation_data["random_common_cause"] = rng.normal(size=len(data))
        refuted_effects[simulation] = estimate_risk_difference(refutation_data)

    return {
        "observed_effect": observed_effect,
        "refuted_effects": refuted_effects,
    }
