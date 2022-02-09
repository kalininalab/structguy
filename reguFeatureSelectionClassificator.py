# standart imports
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def run_linarSVC(cv_slice, config):
    """Runs Lasso on the given data testing on the given protein(must be included in the data),
    """
    if config.reg_c == 0.:
        return None

    reg_c = 1/(10**(-config.reg_c_exp))

    '''Runs a LinearSVC classification dor the given parameter'''
    clf = Pipeline(steps=[('classification', svm.LinearSVC(
        penalty='l1',
        loss="squared_hinge",
        C=reg_c,
        multi_class='ovr',
        max_iter=config.max_iter,
        dual=False,
        class_weight ='balanced'
    ))])

    # train model
    clf.fit(cv_slice.train_feature_matrix, cv_slice.train_targets)

    return clf

def wrapper(cv_slice, config, print_out = False):
    clf = run_linarSVC(cv_slice, config)

    if clf is None:
        return []

    thresh = 1/(10**(-config.reg_thresh_exp))

    filtered = []
    for i in range(0, len(clf.named_steps.regression.coef_)):
        if clf.named_steps.regression.coef_[i] >= thresh:
            filtered.append(cv_slice.feature_names[i])

    return filtered
