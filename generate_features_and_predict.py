import sys
import os
import learn
import util
import sampleSpace
import trainForest
import featureAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from scipy import stats

if __name__ == "__main__":
    dataset_config = util.Config(sys.argv[1])

    forest_file = sys.argv[2]

    forest, feature_names, forest_config = learn.loadModel(forest_file)

    outfile = sys.argv[3]

    if os.path.isfile(outfile):
        samples = learn.createTrainingSet(dataset_config,dataset_config.session,infile = outfile)
    else:
        samples = learn.createTrainingSet(dataset_config,dataset_config.session,outfile = outfile)

    samples.oneHotifyAll()

    full_slice = sampleSpace.FullSlice(samples, dataset_config)

    cv_slice = full_slice.slices[0]

    cv_slice.reorderByNames(feature_names)

    y_pred = forest.predict(cv_slice.train_feature_matrix)

    weighted_r2 = r2_score(cv_slice.train_targets,y_pred,sample_weight = cv_slice.train_class_weight_vector)
    weighted_mse = mean_squared_error(cv_slice.train_targets,y_pred,sample_weight = cv_slice.train_class_weight_vector)

    r2 = r2_score(cv_slice.train_targets,y_pred)
    mse = mean_squared_error(cv_slice.train_targets,y_pred)

    #observed_value_threshold = config.binary_thresh

    tv_median = util.median(cv_slice.train_targets)

    test_targets_b = trainForest.makeBinaryClassifier(cv_slice.train_targets,tv_median)

    #middle_value = (max(y_pred) + min(y_pred))/2.
    y_pred_median = util.median(y_pred)

    y_pred_b = trainForest.makeBinaryClassifier(y_pred,y_pred_median)
    mcc = matthews_corrcoef(test_targets_b,y_pred_b)

    exp_score = mcc/(1.+mse)
    mcc = exp_score #JUST FOR TESTING

    corr,p_value = stats.spearmanr(cv_slice.train_targets,y_pred)
    pearson,pearson_p = stats.pearsonr(cv_slice.train_targets,y_pred)

    scores = util.Scores(mse = mse,r2 = r2,corr = corr,mcc = mcc,pearson_r=pearson, wmse = weighted_mse, wr2 = weighted_r2)

    scores.printOut()

    base_name,f_type = outfile.rsplit('.',1)
    scatterfile = '%s.png' % (base_name)
    hexbinfile = '%s_hexbin.png' % (base_name)

    y_pred_median = util.median(y_pred)
    tv_median = util.median(cv_slice.train_targets)
    util.scatterplot(y_pred, cv_slice.train_feature_matrix, cv_slice.feature_names,
                     cv_slice.train_targets, dataset_config.target_values, y_pred_median, tv_median, scatterfile)
    util.hexbinplot(y_pred, cv_slice.train_targets, dataset_config.target_values, hexbinfile)

    cv_slice.mirrorTrainSamples()

    featureAnalysis.findAndAnalyseInterestingSample(forest, cv_slice, dataset_config)
