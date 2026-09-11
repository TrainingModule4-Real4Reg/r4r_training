from pytmle import PyTMLE
from pytmle.get_initial_estimates import fit_propensity_super_learner


def estimate_propensity(
    df,
    col_group='group',
    calibration_method='isotonic',
    base_learners=None,
    cv_folds=8,
):
    """Estimate treatment propensity scores with a pyTMLE super learner.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the binary treatment indicator and numeric covariates.
        The treatment column is removed before fitting the propensity model.
    col_group : str, default="group"
        Name of the binary treatment column. Values should identify treated
        participants with 1 and control participants with 0.
    calibration_method : {"isotonic", "sigmoid", None}, default="isotonic"
        Probability calibration method applied to the super learner. Use
        ``None`` to estimate uncalibrated probabilities.
    base_learners : list or None, default=None
        Base learner estimators passed to the super learner. If ``None``,
        pyTMLE uses its default base learners.
    cv_folds : int, default=8
        Number of cross-validation folds used by the super learner and the
        optional calibration step.

    Returns
    -------
    numpy.ndarray
        Estimated probability of treatment for each row in ``df``, in the
        same order as the input dataframe.

    Examples
    --------
    Use a random forest as the only base learner:

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> random_forest = RandomForestClassifier(
    ...     n_estimators=200,
    ...     random_state=42,
    ... )
    >>> propensity_scores = estimate_propensity(
    ...     df,
    ...     col_group="chemo",
    ...     base_learners=[random_forest],
    ... )
    """
    group = df[col_group].to_numpy()
    X = df.drop(columns=[col_group]).to_numpy(dtype=float)
    
    propensity_scores_1,_,_ = fit_propensity_super_learner(
        X, 
        group, 
        base_learners=base_learners, 
        verbose=0, 
        cv_folds=cv_folds,
        return_model=False, 
        calibration_method=calibration_method
        )
    
    return propensity_scores_1
    