from causalml.match import NearestNeighborMatch


def control_matching(data, treatment_col="treatment", score_cols=["propensity_score"], ratio=2, caliper=0.05, replace = False, random_state=None):
    """Match treated participants to control participants.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the treatment indicator and matching variables.
    treatment_col : str, default="treatment"
        Name of the binary treatment column. Values must identify treated
        participants with 1 and control participants with 0.
    score_cols : list of str, default=["propensity_score"]
        Columns used to calculate matching distances, typically one or more
        propensity scores.
    ratio : int, default=2
        Number of control participants matched to each treated participant.
    caliper : float, default=0.05
        Maximum allowed distance between matched participants. Pairs outside
        this caliper are not matched.
    replace : bool, default=False
        Whether a control participant may be matched more than once.
    random_state : int or None, default=None
        Seed used by the matching procedure when random choices are needed.

    Returns
    -------
    pandas.DataFrame
        The matched data containing the selected treated and control
        participants.
    """

    psm = NearestNeighborMatch(replace=replace, ratio=ratio, caliper=caliper, random_state=random_state)
    df_matched = psm.match(data=data, treatment_col=treatment_col, score_cols=score_cols)
    
    return df_matched