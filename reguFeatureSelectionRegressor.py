# standart imports
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def run_Lasso_single(cv_slice, config):
    """Runs Lasso on the given data testing on the given protein(must be included in the data),
    """
    
    # create Lasso Pipline
    alpha = 10**(-config.reg_alpha_exp)

    clf = Pipeline(
        steps=[('regression', Lasso(alpha=alpha, max_iter=config.maxIter))])

    # train model
    clf.fit(cv_slice.train_feature_matrix, cv_slice.train_targets)

    return clf

def wrapper(cv_slice, config, print_out = False, pre_filter = None, debug = False):
    """Runs Lasso with numeric and categorical features in a given alpha and iterations intervall
        returns data for the single runs
    """
    if pre_filter is not None:
        cv_slice.filterFeatures(pre_filter)
        filtered = pre_filter
    else:
        cv_slice.filterFeatures([])
        filtered = []
    if (pre_filter is not None) or (not config.reg_alpha_exp in cv_slice.alpha_map):
        if debug:
            print('=========== start LASSO ===================')
        if len(cv_slice.features) == 0:
            if debug:
                print('skipped LASSO, all features got already filtered')
            return filtered
        clf = run_Lasso_single(cv_slice, config)

        if clf is None:
            print('============================= LASSO failed ================================')
            return filtered
        coefs = clf.named_steps.regression.coef_
        cv_slice.alpha_map[config.reg_alpha_exp] = coefs
    else:
        coefs = cv_slice.alpha_map[config.reg_alpha_exp]
    if config.reg_thresh_exp < config.maximal_exp:
        thresh = 10**(-config.reg_thresh_exp)
    else:
        thresh = 0.

    for i in range(0, len(coefs)):
        if abs(coefs[i]) <= thresh:
            filtered.append(cv_slice.feature_names[i])
            if debug:
                print(cv_slice.name,'Filtered by reguFS:',cv_slice.feature_names[i],'Lasso coef:',coefs[i])

    return filtered
