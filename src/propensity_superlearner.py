from pytmle import PyTMLE
from pytmle.get_initial_estimates import fit_propensity_super_learner


def estimate_propensity(df, col_group = 'group', calibration_method = 'isotonic'):
    
    group = df["col_group"].to_numpy()
    X = df.drop(columuns=[col_group]).to_numpy(dtype=float)
    
    propensity_scores_1,_,_ = fit_propensity_super_learner(
        X, 
        group, 
        base_learners=None, 
        verbose=0, 
        cv_folds=8, 
        return_model=False, 
        calibration_method=calibration_method
        )
    
    return propensity_scores_1
    