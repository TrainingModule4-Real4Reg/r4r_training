from causalml.match import NearestNeighborMatch


def control_matching(data, treatment_col="treatment", score_cols=["propensity_score"], ratio=2, caliper=0.05):

    psm = NearestNeighborMatch(replace=False, ratio=ratio, caliper=caliper)
    df_matched = psm.match(data=data, treatment_col=treatment_col, score_cols=score_cols)
    
    return df_matched