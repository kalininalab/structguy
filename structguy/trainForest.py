from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef

import time
import sys
import os
import traceback
import ray
import contextlib
from scipy import stats
import xgboost as xgb
from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback
#import dask

#from dask import array as da
#from dask import dataframe as dd
#from dask.distributed import Client
#from dask_cuda import LocalCUDACluster

#from xgboost import dask as dxgb
#from xgboost.dask import DaskDMatrix

from filelock import FileLock
from structguy import featureSelection, util
from structman.base_utils.base_utils import pack, unpack, add_to_times, print_times, aggregate_times
from structguy.support_classes import CrossValidationSlice
from structguy.sampleSpace import DataSAIL_cv
import numpy
from ray.util.queue import Queue
from ray.train import RunConfig
import shap
from numba import njit

def makeBinaryClassifier(data, thresh, flip_sign=False):
    binary = []
    for value in data:
        if not flip_sign:
            if value < thresh:
                binary.append(0)
            else:
                binary.append(1)
        else:
            if value > thresh:
                binary.append(0)
            else:
                binary.append(1)
    return binary


@ray.remote(max_calls=1)
def trainClassificationForest(
    config,
    samples,
    cv_slice,
    print_out=True,
    skip_scoring=False,
    remote=False,
    cv_counter=None,
):
    depth = config.tree_depth
    min_sample_split = config.min_sample_split
    proc = config.proc_n
    leaf_samples = config.tree_min_leaf_samples
    n_of_trees = int(config.num_of_trees)
    max_leaf_nodes = config.max_leaf_nodes
    class_weight = config.class_weight
    max_features = config.max_feature_parameter
    bootstrap = config.bootstrap_parameter
    min_impurity_decrease = 10 ** (-config.min_impurity_decrease_exp)
    oob_score = config.oob_score
    ccp_alpha = 10 ** (-config.ccp_alpha_exp)
    max_sample_parameter = config.max_sample_parameter
    criterion = config.criterion

    zero_scores = util.Scores(zero=True, n_of_features=len(cv_slice.feature_names))

    if remote:
        zero_return = zero_scores, None, cv_counter
    else:
        zero_return = None, zero_scores, cv_counter

    if depth < 1:
        return zero_return
    if leaf_samples < 1:
        return zero_return
    if min_sample_split < 2:
        return zero_return
    if n_of_trees < 1:
        return zero_return
    if min_impurity_decrease < 0.0 or min_impurity_decrease > 1.0:
        return zero_return
    if ccp_alpha < 0.0:
        return zero_return
    if max_sample_parameter <= 0.0 or max_sample_parameter > 1.0:
        return zero_return

    slice_updated = featureSelection.select_features(config, cv_slice, samples, print_out=print_out)

    if slice_updated is None:
        return zero_return

    if print_out:
        config.printParameter()

    forest = RandomForestClassifier(
        n_estimators=n_of_trees,
        max_depth=depth,
        min_samples_leaf=leaf_samples,
        max_features=max_features,
        n_jobs=proc,
        min_samples_split=min_sample_split,
        max_leaf_nodes=max_leaf_nodes,
        class_weight=class_weight,
        bootstrap=bootstrap,
        criterion=criterion,
        max_samples=max_sample_parameter,
    )

    if print_out:
        config.logger.info(
            f"Fit classification forest, {len(samples.raw_feature_matrix)=} {len(samples.raw_feature_matrix[0])=}"
        )

    forest.fit(cv_slice.train_feature_matrix, cv_slice.train_targets)

    if skip_scoring:
        return forest, None, cv_counter

    y_pred = forest.predict(cv_slice.test_feature_matrix)

    acc = accuracy_score(cv_slice.test_targets, y_pred)
    int_targets = cv_slice.classToInt(cv_slice.test_targets)
    int_preds = cv_slice.classToInt(y_pred)
    roc = roc_auc_score(int_targets, int_preds)

    f1 = f1_score(int_targets, int_preds)

    precision = precision_score(int_targets, int_preds)
    recall = recall_score(int_targets, int_preds)

    mcc = matthews_corrcoef(int_targets, int_preds)

    scores_obj = util.Scores(
        acc=acc,
        roc=roc,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=mcc,
        n_of_features=len(cv_slice.feature_names),
    )

    if print_out:
        scores_obj.printOut()

    if not slice_updated:
        cv_slice = None

    if remote:
        return scores_obj, pack(cv_slice), cv_counter
    return forest, scores_obj, cv_counter


def test_for_constant_array(a):
    if len(a) <= 1:
        return True
    first_value = a[0]
    for second_value in a[1:]:
        if first_value != second_value:
            return False
    return True


@ray.remote
def para_prediction(forest_dump_file, feat_matrix):
    forest = util.loadModel(forest_dump_file)[0]
    pred = forest.predict(feat_matrix)
    return pred

@ray.remote(max_calls=1)
def trainRegressionForestWrapper(
    config,
    cv_slice,
    samples_store_id=None,
    slice_slices=None,
    distance_map=None,
    print_out=True,
    skip_scoring=False,
    cv_counter=None,
    debug=False,
    overwrite_proc_n=None,
    skip_feature_selection=False,
    force_confusion=False,
    score_train=True,
):
    return trainRegressionForest(
        config,
        unpack(cv_slice),
        samples_store_id=samples_store_id[0],
        slice_slices=slice_slices,
        distance_map=distance_map,
        print_out=print_out,
        skip_scoring=skip_scoring,
        cv_counter=cv_counter,
        debug=debug,
        remote=True,
        skip_feature_selection=skip_feature_selection,
        overwrite_proc_n=overwrite_proc_n,
        force_confusion=force_confusion,
        score_train=score_train,
    )



@ray.remote
def para_perturb(com_queue, store):
    feat_names, feat_name_backmap, forest_dump_file, get_sample_wise_data, get_feat_impacts = store
    
    acc_feat_impacts = {}
    sample_wise_feature_influence = {}
    
    
    max_number_of_samples = 1_000
    if not com_queue.empty():
        feat_vecs, left, prediction_vector, target_vector = com_queue.get()
    else:
        return sample_wise_feature_influence, acc_feat_impacts
    while True:
        metadata_list = []
        perturbed_feat_matrices = [[]]
        matrix_id = 0
        for sub_sample_id, feat_vec in enumerate(feat_vecs):
            for feat_name in feat_names:
                #config.logger.info(f'{feat_name=} {feat_name not in feat_name_backmap=}')
                if feat_name not in feat_name_backmap:
                    continue
                    
                #pred = xgb_forest.predict([feat_vec])[0]
                feat_pos = feat_name_backmap[feat_name]
                val = feat_vec[feat_pos]
                #config.logger.info(f'{sample_id=} {feat_name=} {val=} {pred=}')

                if feat_name[:3] == 'oh_':
                    if val is not None:
                        """
                        alt_val = abs(val-1)
                        perturbed_vec = feat_vec[:]
                        perturbed_vec[feat_pos] = alt_val
                        perturbed_feat_matrix.append(perturbed_vec)
                        metadata_list.append((sample_id, feat_name, 'oh_alt'))
                        """

                        perturbed_vec = feat_vec[:]
                        perturbed_vec[feat_pos] = None
                        perturbed_feat_matrices[matrix_id].append(perturbed_vec)
                        metadata_list.append((sub_sample_id, feat_name))
                else:
                    perturbed_vec = feat_vec[:]
                    perturbed_vec[feat_pos] = None
                    perturbed_feat_matrices[matrix_id].append(perturbed_vec)
                    metadata_list.append((sub_sample_id, feat_name))
                if len(perturbed_feat_matrices[matrix_id]) >= max_number_of_samples:
                    perturbed_feat_matrices.append([])
                    matrix_id += 1

        forest = util.loadModel(forest_dump_file)[0]
        perturbed_pred = numpy.array([])
        for perturbed_feat_matrix in perturbed_feat_matrices:
            if len(perturbed_feat_matrix) == 0:
                continue
            perturbed_pred = numpy.concatenate((perturbed_pred, forest.predict(perturbed_feat_matrix)))


        pert_dicts: list[dict[str, float]] = [{}]
        curr_sample_id = 0
        for pert_id, (sub_sample_id, feat_name) in enumerate(metadata_list):

            pred = prediction_vector[sub_sample_id]
            pert_pred = perturbed_pred[pert_id]

            d = pred - pert_pred

            #config.logger.info(f'{sample_id=} {feat_name=} {d=} {alt_val_type=} {curr_sample_id}')

            if sub_sample_id != curr_sample_id:
                curr_sample_id = sub_sample_id
                pert_dicts.append({})

            if get_feat_impacts:
                true_val = target_vector[sub_sample_id]
                feature_impact = abs(true_val - pred) - abs(true_val - pert_pred)
            else:
                feature_impact = None
            pert_dicts[sub_sample_id][feat_name] = [d, feature_impact]

        for sub_sample_id, pert_dict in enumerate(pert_dicts):
            sample_id = sub_sample_id + left

            for feat_name in pert_dict:
                d = pert_dict[feat_name]
                if get_feat_impacts:
                    if feat_name not in acc_feat_impacts:
                        acc_feat_impacts[feat_name] = 0.

                    acc_feat_impacts[feat_name] += d[1]
                    
            if get_sample_wise_data:
                pert_list = [(k, pert_dict[k]) for k in pert_dict]
                sorted_pert_list = sorted(pert_list, key=lambda x:abs(x[1][0]), reverse=True)
                feat_data = []
                
                for feat_name, d in sorted_pert_list:
                    feat_pos = feat_name_backmap[feat_name]
                    val = feat_vecs[sub_sample_id][feat_pos]
                    feat_data.append((feat_name, d[0], val))
                sample_wise_feature_influence[sample_id] = feat_data

        if not com_queue.empty():
            try:
                feat_vecs, left, prediction_vector, target_vector = com_queue.get(timeout = 120.)
            except TimeoutError:
                break
            except ray.util.queue.Empty:
                break
        else:
            break
    return sample_wise_feature_influence, acc_feat_impacts



def perturb_xgb(config, forest_dump_file, feat_vecs, feature_names, prediction_vector, target_vector, get_sample_wise_data = False, get_feat_impacts=True):
    #total_ram = ray._private.utils.get_system_memory()
    model_filesize = os.path.getsize(forest_dump_file)
    #safe_memory_estimate_per_model = (model_filesize * 1_000) + 1024*1024*1024
    #half_ram = total_ram/2

    #max_proc_n_by_mem = int(half_ram // safe_memory_estimate_per_model)
    max_proc_n_by_mem = config.proc_n
    config.logger.info(f'{max_proc_n_by_mem=} {model_filesize=}')
    n_jobs = max([2, min([config.proc_n, max_proc_n_by_mem])])

    max_number_of_samples = len(feat_vecs) // n_jobs
    if len(feat_vecs) % n_jobs != 0:
        max_number_of_samples += 1

    max_number_of_samples = 1_000

    feat_name_backmap = {}
    for feat_number, feat_name in enumerate(feature_names):
        feat_name_backmap[feat_name] = feat_number

    config.logger.info(f'{len(feature_names)=} {len(feat_vecs)=} {get_sample_wise_data=} {get_feat_impacts=}')

    acc_feat_impacts = {}
    if get_sample_wise_data:
        sample_wise_feature_influence = [None] * len(feat_vecs)
    else:
        sample_wise_feature_influence = None

    not_done = True
    iter_number = 0
    para_perturb_process_ids = []
    store = ray.put((feature_names, feat_name_backmap, forest_dump_file, get_sample_wise_data, get_feat_impacts))

    com_queue = Queue()
    while not_done:
        left = iter_number*max_number_of_samples
        right = (iter_number+1)*max_number_of_samples
        if right >= len(feat_vecs):
            not_done = False

        submatrix = feat_vecs[left:right]
        com_queue.put((submatrix, left, prediction_vector[left:right], target_vector[left:right]))
        #config.logger.info(f'{left=} {right=} {len(submatrix)=} {len(prediction_vector[left:right])=} {len(target_vector[left:right])=}')
        iter_number += 1

    n_jobs = min([n_jobs, iter_number])

    for i in range(n_jobs):
        para_perturb_process_ids.append(para_perturb.remote(com_queue, store))


    config.logger.info(f'{len(para_perturb_process_ids)=}')

    t0 = time.time()
    results = ray.get(para_perturb_process_ids)
    t1 = time.time()

    config.logger.info(f'Time for the para processes: {t1-t0} {n_jobs=}')

    for sub_sample_wise_feature_influence, sub_acc_feat_impacts in results:
        if get_feat_impacts:
            for feat_name in sub_acc_feat_impacts:
                if feat_name not in acc_feat_impacts:
                    acc_feat_impacts[feat_name] = 0.

                acc_feat_impacts[feat_name] += sub_acc_feat_impacts[feat_name]

        if get_sample_wise_data:
            for sample_id in sub_sample_wise_feature_influence:
                sample_wise_feature_influence[sample_id] = sub_sample_wise_feature_influence[sample_id]


    if get_feat_impacts:
        acc_feat_impacts = [(k, acc_feat_impacts[k]/len(feat_vecs)) for k in acc_feat_impacts]
        acc_feat_impacts = sorted(acc_feat_impacts, key=lambda x:x[1], reverse=True)
        #config.logger.info(acc_feat_impacts)
    return sample_wise_feature_influence, acc_feat_impacts

@njit
def shap_internal_loop(
        N_feat_names: int,
        explanation: numpy.ndarray,
        prediction_vector: numpy.ndarray,
        target_vector: numpy.ndarray
        ) -> numpy.ndarray:
    acc_feat_impacts: numpy.ndarray = numpy.zeros(N_feat_names)
    for sample_pos, shap_values in enumerate(explanation):
        pred: float = prediction_vector[sample_pos]
        true_val: float = target_vector[sample_pos]
        loss: float = abs(true_val - pred)
        for feat_pos in range(N_feat_names):
            shap_val: float = shap_values[feat_pos]
            pert_pred: float = pred - shap_val
            feature_impact: float = loss - abs(true_val - pert_pred)
            acc_feat_impacts[feat_pos] += feature_impact

    return acc_feat_impacts

def shap_analysis(config, forest, feat_vecs, feature_names, prediction_vector, target_vector):
    if config.verbosity >= 2:
        config.logger.info(f'Call of shap_analysis: {config.gpu_mode=} {len(feat_vecs)=}')
    
    times = []
    ta = time.time()
    
    if config.gpu_mode:
        #explainer = shap.explainers.GPUTree(forest, np_feat_vecs)
        d_feat_vecs = xgb.DMatrix(feat_vecs, feature_names = feature_names)
        done = False
        n = 1
        while not done:
            try:
                if n > 1:
                    h = len(feat_vecs) // 2
                    d_feat_vecs_1 = xgb.DMatrix(feat_vecs[:h], feature_names = feature_names)
                    d_feat_vecs_2 = xgb.DMatrix(feat_vecs[h:], feature_names = feature_names)
                    explanation_1 = forest.get_booster().predict(d_feat_vecs_1, pred_contribs=True)
                    explanation_2 = forest.get_booster().predict(d_feat_vecs_2, pred_contribs=True)

                    explanation = numpy.concatenate(explanation_1, explanation_2)
                else:
                    explanation = forest.get_booster().predict(d_feat_vecs, pred_contribs=True)
                done = True
            except xgb.core.XGBoostError:
                done = False
                
                n+=1
                if n == 4:
                    return None, times
                config.logger.info(f'Catched XGBoost Error, try again {n}')
                time.sleep(n**2)

    else:
        np_feat_vecs = numpy.array(feat_vecs)
        explainer = shap.TreeExplainer(forest)
        ta = add_to_times(times, ta)
        explanation = explainer(np_feat_vecs)
        explanation = numpy.array([x.values for x in explanation])
    
    ta = add_to_times(times, ta)

    acc_feat_impacts = shap_internal_loop(
        len(feature_names),
        explanation,
        numpy.array(prediction_vector),
        numpy.array(target_vector)
        )
    
    ta = add_to_times(times, ta)

    acc_feat_impacts = sorted(zip(feature_names, [x/len(feat_vecs) for x in acc_feat_impacts]), key=lambda x:x[1], reverse=True)
    ta = add_to_times(times, ta)
    return acc_feat_impacts, times


def cut_and_predict(feat_matrix, model):
    h = len(feat_matrix)//2
    pred_1 = model.predict(feat_matrix[:h])
    pred_2 = model.predict(feat_matrix[h:])

    concatted = numpy.concatenate(pred_1, pred_2)
    return concatted


def return_zero(zero_return, remote, cv_slice):
    if remote:
        del cv_slice
    return zero_return


"""
def using_dask_matrix(client: Client, X: da.Array, y: da.Array, config: util.Config, es_list, train_weights, data_tuple_list) -> da.Array:
    # DaskDMatrix acts like normal DMatrix, works as a proxy for local DMatrix scatter
    # around workers.
    dtrain = DaskDMatrix(client, X, y)

    # Use train method from xgboost.dask instead of xgboost.  This distributed version
    # of train returns a dictionary containing the resulting booster and evaluation
    # history obtained from evaluation metrics.

    xgb_params = {
        "tree_method": "hist",
        "device": "cuda",
        "sample_weight": "train_weights",
        "n_estimators": "int(config.num_of_trees)",
        "max_depth": "config.tree_depth",
        "reg_alpha ": " config.xgb_alpha",
        "reg_lambda ": " config.xgb_lambda",
        "colsample_bytree ": " config.colsample_bytree",
        "max_delta_step ": " config.max_delta_step",
        "gamma": "config.xgb_gamma",
        "learning_rate": "config.learning_rate",
        "min_child_weight": "config.min_child_weight",
        "early_stopping_rounds": "config.early_stopping",
        "subsample": "config.max_sample_parameter",
        "callbacks": "es_list",
        "eval_metric": "util.rho_eval_for_xgboost",
        }
            
    output = dxgb.train(
        client,
        # Make sure the device is set to CUDA.
        xgb_params,
        dtrain,
        
        evals=data_tuple_list,
    )
    bst = output["booster"]
    #history = output["history"]

    # you can pass output directly into `predict` too.
    #prediction = dxgb.predict(client, bst, dtrain)
    #config.logger.info("Evaluation history:", history)
    return bst
"""
    
def ray_xgb_train_func(packed_params):
    dump_path = packed_params['dump_path']
    f = open(dump_path, 'rb')
    packed_par = f.read()
    f.close()
    params = unpack(packed_par)
    dtrain = xgb.DMatrix(params["train_feature_matrix"], params["train_targets"], weight = params["train_weights"])

    config = params["config"]

    es_list = []
    data_tuple_list: list[tuple[list[list[int | float | None]], list[float]]] = []
    for n, prot_id in enumerate(params["protwise_test_data_tuples"]):
        es = xgb.callback.EarlyStopping(
            rounds=config.early_stopping,
            min_delta=1e-3,
            save_best=True,
            maximize=False,
            data_name=f"validation_{n}",
            metric_name="irho"
        )
        es_list.append(es)
        data_tuple_list.append(xgb.DMatrix(numpy.array(params["protwise_test_data_tuples"][prot_id][0]), numpy.array(params["protwise_test_data_tuples"][prot_id][1])))

    xgb_params = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": config.tree_depth,
        "reg_alpha ": config.xgb_alpha,
        "reg_lambda ": config.xgb_lambda,
        "colsample_bytree ": config.colsample_bytree,
        "max_delta_step ": config.max_delta_step,
        "gamma": config.xgb_gamma,
        "learning_rate": config.learning_rate,
        "min_child_weight": config.min_child_weight,
        "early_stopping_rounds": config.early_stopping,
        "subsample": config.max_sample_parameter,
        "callbacks": es_list,
        "eval_metric": util.rho_eval_for_xgboost,
        }
    
    xgb.train(xgb_params, dtrain, num_boost_round=int(config.num_of_trees), evals=data_tuple_list, maximize=False, custom_metric=util.rho_eval_for_xgboost)

def xgb_train_wrapper(
        config: util.Config,
        forest: None | xgb.XGBRegressor,
        train_feature_matrix: list[list[int | float | None]],
        train_targets: list[float],
        data_tuple_list: list[tuple[list[list[int | float | None]], list[float]]],
        protwise_test_data_tuples: list[tuple[list[list[int | float | None]], list[float]]],
        train_weights: list[float],
        label: str = ''):
    if config.multi_gpu is None or config.multi_gpu < 2:
        try:
            if config.verbosity >= 3:
                forest.fit(
                    train_feature_matrix,
                    train_targets,
                    eval_set=data_tuple_list,
                    sample_weight=train_weights,
                )
            else:
            
                with contextlib.redirect_stdout(None):
                    forest.fit(
                        train_feature_matrix,
                        train_targets,
                        eval_set=data_tuple_list,
                        sample_weight=train_weights,
                    )
        except xgb.core.XGBoostError:
            return None
    else:
        """
        #import dask_cudf
        # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
        # `n_workers` represents the number of GPUs since we use one GPU per worker process.
        with LocalCUDACluster(n_workers=config.multi_gpu, threads_per_worker=config.proc_n) as cluster:
            # Create client from cluster, set the backend to GPU array (cupy).
            with Client(cluster) as client, dask.config.set({"array.backend": "cupy"}):
                # Generate some random data for demonstration
                

                X = dd.from_dask_array(train_feature_matrix)
                y = dd.from_dask_array(train_targets)
                # XGBoost can take arrays. This is to show that DataFrame uses the GPU
                # backend as well.
                #assert isinstance(X, dask_cudf.DataFrame)
                #assert isinstance(y, dask_cudf.Series)

                config.logger.info(f"Using dask to train multi gpu xgboost training: {config.multi_gpu=}")
                forest = using_dask_matrix(client, X, y, config, es_list, train_weights, data_tuple_list).compute()
        """
        storage = f'{config.outfolder}/ray_storage'
        if not os.path.isdir(storage):
            os.makedirs(storage)
        data_dump = f'{storage}/data.dump'
        params = {
            "config" : config,
            "train_feature_matrix" : train_feature_matrix,
            "train_targets" : train_targets,
            "train_weights" : train_weights,
            "protwise_test_data_tuples" : protwise_test_data_tuples,
        }
        packed_params = {'dump_path' : pack(params)}
        with open(data_dump, 'wb') as f:
            f.write(packed_params)
        
        run_config = RunConfig(storage_path=storage, name=f"run_name{label}")
        trainer = XGBoostTrainer(
            ray_xgb_train_func, scaling_config=config.scaling_config, run_config=run_config, train_loop_config=packed_params
        )
        result = trainer.fit()
        with result.checkpoint.as_directory() as checkpoint_dir:
            model_path = os.path.join(checkpoint_dir, RayTrainReportCallback.CHECKPOINT_NAME)
            forest = xgb.Booster()
            forest.load_model(model_path)
    return forest

def trainRegressionForest(
    config: util.Config,
    cv_slice: CrossValidationSlice,
    samples=None,
    samples_store_id=None,
    slice_slices=None,
    distance_map=None,
    print_out=True,
    skip_scoring=False,
    score_train=True,
    cv_counter=None,
    remote=False,
    overwrite_proc_n=None,
    debug=False,
    skip_feature_selection=False,
    force_confusion=False,
):
    times = []
    ta = time.time()


    depth = config.tree_depth
    min_sample_split = config.min_sample_split
    if overwrite_proc_n is None:
        proc = config.proc_n
    else:
        proc = int(overwrite_proc_n)
    leaf_samples = config.tree_min_leaf_samples
    n_of_trees = int(config.num_of_trees)
    rel_max_leaf_node_pruning = config.rel_max_leaf_node_pruning

    max_feature_parameter = config.max_feature_parameter
    if config.max_feature_cont_parameter is not None:
        max_feature_parameter = config.max_feature_cont_parameter

    if config.min_impurity_decrease_exp < config.maximal_exp:
        min_impurity_decrease = 10 ** (-config.min_impurity_decrease_exp)
    else:
        min_impurity_decrease = 0.0

    if config.fs_min_impurity_decrease_exp < config.maximal_exp:
        fs_min_impurity_decrease = 10**-(config.fs_min_impurity_decrease_exp)
    else:
        fs_min_impurity_decrease = 0.0

    if config.ccp_alpha_exp < config.maximal_exp:
        ccp_alpha = 10**-(config.ccp_alpha_exp)
    else:
        ccp_alpha = 0.0

    if config.fs_ccp_alpha_exp < config.maximal_exp:
        fs_ccp_alpha = 10**-(config.fs_cpp_alpha_exp)
    else:
        fs_ccp_alpha = 0.0

    max_sample_parameter = config.max_sample_parameter
    criterion = config.criterion
    number_of_bins = config.number_of_bins

    zero_scores = util.Scores(zero=True, n_of_features=len(cv_slice.feature_names))

    if remote:
        zero_return = zero_scores, None, cv_counter, times, None
    else:
        zero_return = (
            None,
            zero_scores,
            cv_counter,
            None,
            times,
            slice_slices,
        )

    if depth < 1:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {depth=}")
        return return_zero(zero_return, remote, cv_slice)
    if leaf_samples < 1:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {leaf_samples=}")
        return return_zero(zero_return, remote, cv_slice)
    if min_sample_split < 2:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {min_sample_split=}")
        return return_zero(zero_return, remote, cv_slice)
    if n_of_trees < 1:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {n_of_trees=}")
        return return_zero(zero_return, remote, cv_slice)
    if min_impurity_decrease < 0.0 or min_impurity_decrease > 1.0:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {min_impurity_decrease=}")
        return return_zero(zero_return, remote, cv_slice)
    if ccp_alpha < 0.0:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {ccp_alpha=}")
        return return_zero(zero_return, remote, cv_slice)

    if max_sample_parameter <= 0.0 or max_sample_parameter > 1.0:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {max_sample_parameter=}")
        return return_zero(zero_return, remote, cv_slice)

    if skip_feature_selection and (config.fs_max_sample_parameter <= 0.0 or config.fs_max_sample_parameter > 1.0):
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {config.fs_max_sample_parameter=}")
        return return_zero(zero_return, remote, cv_slice)

    if isinstance(max_feature_parameter, float):
        if max_feature_parameter <= 0.0 or max_feature_parameter > 1.0:
            if config.verbosity >= 3:
                config.logger.info(f"Return Zero: {max_feature_parameter=}")
            return return_zero(zero_return, remote, cv_slice)
    if number_of_bins < 1:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {number_of_bins=}")
        return return_zero(zero_return, remote, cv_slice)

    ta = add_to_times(times, ta) #0
    if config.verbosity >= 2:
        config.logger.info(f"Train regression forest part 1, Threads: {proc}, Feature selection: {not skip_feature_selection} {samples is None=} {config.forest_type=} {config.gpu_mode=} {config.multi_gpu=}")

    if not skip_feature_selection and not config.random_split:
        get_loss_map = False
        slice_updated, filtered_features, feat_select_times, slice_slices, loss_map = featureSelection.select_features(
            config,
            cv_slice,
            slice_slices,
            samples_store_id=samples_store_id,
            samples=samples,
            print_out=print_out,
            debug=debug,
            overwrite_proc_n=proc,
            force_confusion=force_confusion,
            get_loss_map=get_loss_map,
            remote=remote
        )
        if not config.random_split and get_loss_map:
            cv_slice.update_train_weight_vector(loss_map)
    else:
        slice_updated = False
        feat_select_times = []
        slice_slices = None
        loss_map = None
        filtered_features = []

    if config.auto_weighting:
        cv_slice.check_auto_weights()

    times.append(feat_select_times) #1

    if slice_updated is None:
        config.logger.error(f"ERROR =============== Feature selection failed: {cv_counter}")
        config.printParameter()
        config.logger.error("==============================================")
        return return_zero(zero_return, remote, cv_slice)

    if len(cv_slice.feature_names) == 0:
        config.logger.warning(f"Warning =============== Feature vector has len 0 {cv_counter}")
        # config.printParameter()
        # config.logger.info('==============================================')
        return return_zero(zero_return, remote, cv_slice)

    if max_sample_parameter == 1.0:
        max_sample_parameter = None

    if print_out:
        config.printParameter()

    ta = add_to_times(times, ta) #2
    if config.verbosity >= 2:
        config.logger.info(f"Train regression forest part 2, {proc=} {samples is None=} {score_train=} {len(filtered_features)=}")
    
    if config.forest_type == "gradient_boost" and not skip_feature_selection:
        forest = GradientBoostingRegressor(
            n_estimators=n_of_trees,
            max_depth=depth,
            min_samples_leaf=leaf_samples,
            max_features=max_feature_parameter,
            min_samples_split=min_sample_split,
            ccp_alpha=ccp_alpha,
            min_impurity_decrease=min_impurity_decrease,
            learning_rate=config.learning_rate,
            # no_iter_no_change = config.early_stopping,
            subsample=max_sample_parameter,
        )
    elif config.forest_type == "xgboost" and not skip_feature_selection:
        import xgboost as xgb

        packed_slice_slice = slice_slices[0]
        try:
            slice_slice = unpack(packed_slice_slice)
        except:
            slice_slice = packed_slice_slice
        slice_slice.filterFeatures(filtered_features)
        if debug:
            slice_slice.printBalance(config)
        if samples is None:
            samples = unpack(ray.get(samples_store_id))

        protwise_test_data_tuples = slice_slice.get_prot_wise_test_data_tuples(samples)
        
        if config.multi_gpu is None or config.multi_gpu < 2:
            if config.gpu_mode:
                n_jobs=config.proc_n
                device = 'cuda'
                tree_method = 'hist'
            else:
                n_jobs=config.proc_n
                device = 'cpu'
                tree_method = 'hist'

            es_list = []
            data_tuple_list: list[tuple[list[list[int | float | None]], list[float]]] = []
            for n, prot_id in enumerate(protwise_test_data_tuples):
                es = xgb.callback.EarlyStopping(
                    rounds=config.early_stopping,
                    min_delta=1e-3,
                    save_best=True,
                    maximize=False,
                    data_name=f"validation_{n}",
                    metric_name="irho"
                )
                es_list.append(es)
                data_tuple_list.append(protwise_test_data_tuples[prot_id])


            forest: xgb.XGBRegressor = xgb.XGBRegressor(
                n_jobs=n_jobs,
                device=device,
                n_estimators=n_of_trees,
                max_depth=depth,
                reg_alpha = config.xgb_alpha,
                reg_lambda = config.xgb_lambda,
                colsample_bytree = config.colsample_bytree,
                max_delta_step = config.max_delta_step,
                gamma=config.xgb_gamma,
                learning_rate=config.learning_rate,
                min_child_weight=config.min_child_weight,
                early_stopping_rounds=config.early_stopping,
                subsample=max_sample_parameter,
                verbosity=0,
                callbacks=es_list,
                eval_metric=util.rho_eval_for_xgboost,
                tree_method=tree_method,
            )
        else:
            forest = None
            data_tuple_list = None

    elif skip_feature_selection:
        forest = RandomForestRegressor(
            n_estimators=config.fs_num_of_trees,
            max_depth=config.fs_tree_depth,
            min_samples_leaf=config.fs_tree_min_leaf_samples,
            n_jobs=config.proc_n,
            min_samples_split=config.fs_min_sample_split,
            ccp_alpha=fs_ccp_alpha,
            min_impurity_decrease=fs_min_impurity_decrease,
            max_samples=config.fs_max_sample_parameter,
            criterion=criterion,
        )
    else:
        forest = RandomForestRegressor(
            n_estimators=n_of_trees,
            max_depth=depth,
            min_samples_leaf=leaf_samples,
            max_features=max_feature_parameter,
            n_jobs=config.proc_n,
            min_samples_split=min_sample_split,
            ccp_alpha=ccp_alpha,
            min_impurity_decrease=min_impurity_decrease,
            max_samples=max_sample_parameter,
            criterion=criterion,
        )

    
    ta = add_to_times(times, ta) #3

    if config.verbosity >= 5:
        util.sanity_check_value_list(
            cv_slice.train_targets,
            label_vector=cv_slice.train_sample_ids,
            datastructure_name=f"Train target of {cv_slice.name}",
        )

    ta = add_to_times(times, ta) #4
    if config.verbosity >= 3:
        config.logger.info(f"Train regression forest part 3, {len(cv_slice.feature_names)=} {samples is None=}")

    if print_out or config.verbosity >= 3:
        config.logger.info(
            f"Train regression {config.forest_type} forest, call of fit with # of features: {len(cv_slice.feature_names)}, skip feature selection {skip_feature_selection}, slice update {slice_updated}, skip scoring {skip_scoring}"
        )

    if samples is None:
        samples = unpack(ray.get(samples_store_id))

    ta = add_to_times(times, ta) #5

    weights_updated = False

    
    if config.forest_type == "xgboost" and not skip_feature_selection:
        if config.weighting == "geometric":
            slice_slice.calcSampleWeights(config, distance_map)
        elif config.weighting == "subsample_distance":
            weights_updated = slice_slice.calcSubsampleDistanceWeights(config, para_number=proc)
        train_feature_matrix: list[list[int | float | None]] = slice_slice.get_train_feature_matrix(samples, sub_sampling=config.sub_sample_factor)
        if config.sub_sample_factor == 1.0:
            train_targets = slice_slice.train_targets
            train_weights = slice_slice.train_class_weight_vector
        else:
            train_targets = slice_slice.sub_sampled_train_targets
            train_weights = slice_slice.sub_sampled_train_class_weight_vector

        ta = add_to_times(times, ta) #6

        forest = xgb_train_wrapper(config, forest, train_feature_matrix, train_targets, data_tuple_list, protwise_test_data_tuples, train_weights)
        if forest is None:
            return return_zero(zero_return, remote, cv_slice)

        """
        #with FileLock("rf_regressor.lock"):
        if config.verbosity >= 3:
            forest.fit(
                train_feature_matrix,
                train_targets,
                eval_set=data_tuple_list,
                sample_weight=train_weights,
            )
        else:
            try:
                with contextlib.redirect_stdout(None):
                    forest.fit(
                        train_feature_matrix,
                        train_targets,
                        eval_set=data_tuple_list,
                        sample_weight=train_weights,
                    )
            except xgb.core.XGBoostError:
                return return_zero(zero_return, remote, cv_slice)
        """
    else:
        if config.weighting == "geometric":
            cv_slice.calcSampleWeights(config, distance_map)
        elif config.weighting == "subsample_distance":
            weights_updated = cv_slice.calcSubsampleDistanceWeights(config, para_number=proc)
        train_feature_matrix: list[list[int | float | None]] = cv_slice.get_train_feature_matrix(samples, sub_sampling=config.sub_sample_factor)
        if config.sub_sample_factor == 1.0:
            train_targets = cv_slice.train_targets
            train_weights = cv_slice.train_class_weight_vector
        else:
            train_targets = cv_slice.sub_sampled_train_targets
            train_weights = cv_slice.sub_sampled_train_class_weight_vector
        #with FileLock("rf_regressor.lock"):
        ta = add_to_times(times, ta) #6
        forest.fit(
            train_feature_matrix,
            train_targets,
            sample_weight=train_weights,
        )

    slice_updated = slice_updated or weights_updated
    ta = add_to_times(times, ta) #7

    if config.forest_type == 'xgboost' and not skip_feature_selection:
        #path_to_model = f'tmp_model_{cv_counter}.dump'
        #util.storeModel(forest, None, config, path_to_model, None)

        ta = add_to_times(times, ta) #8

        test_feature_matrix = slice_slice.get_test_feature_matrix(samples)
        y_pred = forest.predict(test_feature_matrix)

        ta = add_to_times(times, ta) #9

        #_, acc_feat_impacts = perturb_xgb(config, path_to_model, test_feature_matrix, cv_slice.feature_names, y_pred, slice_slice.test_targets)
        acc_feat_impacts, shap_times = shap_analysis(config, forest, test_feature_matrix, slice_slice.feature_names, y_pred, slice_slice.test_targets)
        times.append(shap_times)

        ta = add_to_times(times, ta) #10
        if acc_feat_impacts is None:
            return return_zero(zero_return, remote, cv_slice)

        config.logger.info(f'{acc_feat_impacts[:5]=}\n{acc_feat_impacts[-5:]=}')
        feats_to_remove = filtered_features[:]
        for feat_name, feat_impact in acc_feat_impacts:
            if feat_impact >= config.feat_impact_thresh:
                feats_to_remove.append(feat_name)

        config.logger.info(f'{len(feats_to_remove)=} {len(filtered_features)=} {len(cv_slice.feature_names) + len(filtered_features)=}')

        if len(feats_to_remove) >= (len(cv_slice.feature_names)+ len(filtered_features)):
            return return_zero(zero_return, remote, cv_slice)
        
        if len(feats_to_remove) > len(filtered_features):
            cv_slice.filterFeatures(feats_to_remove)
            slice_slice.filterFeatures(feats_to_remove)
            
            ta = add_to_times(times, ta) #11

            if config.weighting == "geometric":
                slice_slice.calcSampleWeights(config, distance_map)
            elif config.weighting == "subsample_distance":
                weights_updated = slice_slice.calcSubsampleDistanceWeights(config, para_number=proc)
            train_feature_matrix: list[list[int | float | None]] = slice_slice.get_train_feature_matrix(samples, sub_sampling=config.sub_sample_factor)
            
            #with FileLock("rf_regressor.lock"):
            
            protwise_test_data_tuples = slice_slice.get_prot_wise_test_data_tuples(samples)
            
            if config.multi_gpu is None or config.multi_gpu < 2:
                es_list = []
                data_tuple_list: list[tuple[list[list[int | float | None]], list[float]]] = []
                for n, prot_id in enumerate(protwise_test_data_tuples):
                    es = xgb.callback.EarlyStopping(
                        rounds=config.early_stopping,
                        min_delta=1e-3,
                        save_best=True,
                        maximize=False,
                        data_name=f"validation_{n}",
                        metric_name="irho"
                    )
                    es_list.append(es)
                    data_tuple_list.append(protwise_test_data_tuples[prot_id])

                forest: xgb.XGBRegressor = xgb.XGBRegressor(
                    n_jobs=n_jobs,
                    device=device,
                    n_estimators=n_of_trees,
                    max_depth=depth,
                    reg_alpha = config.xgb_alpha,
                    reg_lambda = config.xgb_lambda,
                    colsample_bytree = config.colsample_bytree,
                    max_delta_step = config.max_delta_step,
                    gamma=config.xgb_gamma,
                    learning_rate=config.learning_rate,
                    min_child_weight=config.min_child_weight,
                    early_stopping_rounds=config.early_stopping,
                    subsample=max_sample_parameter,
                    verbosity=0,
                    callbacks=es_list,
                    eval_metric=util.rho_eval_for_xgboost,
                    tree_method=tree_method,
                )
            else:
                forest = None
                data_tuple_list = None
            forest = xgb_train_wrapper(config, forest, train_feature_matrix, train_targets, data_tuple_list, protwise_test_data_tuples, train_weights)
            if forest is None:
                return return_zero(zero_return, remote, cv_slice)
            """
            with contextlib.redirect_stdout(None):
                forest.fit(
                    train_feature_matrix,
                    train_targets,
                    eval_set=data_tuple_list,
                    sample_weight=train_weights,
                )
            """

            slice_slice.filterFeatures([])

        ta = add_to_times(times, ta) #12

    if skip_scoring:
        return forest, None, cv_counter, cv_slice, times, slice_slices

    if debug:
        cv_slice.printBalance(config)

    test_feature_matrix = cv_slice.get_test_feature_matrix(samples)
    try:
        y_pred = forest.predict(test_feature_matrix)
    except ValueError:
        [e, f, g] = sys.exc_info()
        g = traceback.format_exc()
        config.logger.info(f'Catched error {config.forest_type=} {skip_feature_selection=}: {e}\n{f}\n{g}\n')
        return return_zero(zero_return, remote, cv_slice)
    if score_train:
        if config.forest_type == "xgboost" and not skip_feature_selection:
            train_feature_matrix = cv_slice.get_train_feature_matrix(samples, sub_sampling=config.sub_sample_factor)
        if config.sub_sample_factor == 1.0:
            train_sample_ids = cv_slice.train_sample_ids
            train_targets = cv_slice.train_targets
            train_weights = cv_slice.train_class_weight_vector
        else:
            train_sample_ids = cv_slice.sub_sampled_train_ids
            train_targets = cv_slice.sub_sampled_train_targets
            train_weights = cv_slice.sub_sampled_train_class_weight_vector
        done = False
        n = 1
        while not done:
            try:
                if n > 1:
                    x_pred = cut_and_predict(train_feature_matrix, forest)
                else:
                    x_pred = forest.predict(train_feature_matrix)
                done = True
            except ValueError:
                if n < 4:
                    time.sleep(n**2)
                    done = False
                    n += 1
                else:
                    [e, f, g] = sys.exc_info()
                    g = traceback.format_exc()
                    config.logger.info(f'Catched error {config.forest_type=} {skip_feature_selection=}: {e}\n{f}\n{g}\n')
                    return return_zero(zero_return, remote, cv_slice)

    ta = add_to_times(times, ta) #8/13

    if test_for_constant_array(y_pred):
        if print_out:
            zero_scores.printOut()
        if config.verbosity >= 3:
            config.logger.info("Return Zero: test_for_constant_array was True")
        return return_zero(zero_return, remote, cv_slice)

    if config.verbosity >= 5:
        util.sanity_check_value_list(
            cv_slice.test_class_weight_vector,
            label_vector=cv_slice.test_sample_ids,
            datastructure_name=f"Test weight vector of {cv_slice.name}",
        )
        util.sanity_check_value_list(
            cv_slice.test_targets,
            label_vector=cv_slice.test_sample_ids,
            datastructure_name=f"Test target of {cv_slice.name}",
        )
        util.sanity_check_value_list(
            y_pred,
            label_vector=cv_slice.test_sample_ids,
            datastructure_name=f"Predictions of {cv_slice.name}",
        )

    scores_obj = calc_scores_obj(cv_slice.test_targets, y_pred, cv_slice.test_sample_ids, cv_slice.test_class_weight_vector, cv_slice.feature_names)
    if score_train:
        train_scores_obj = calc_scores_obj(train_targets, x_pred, train_sample_ids, train_weights, cv_slice.feature_names)
        scores_obj.train_scores = train_scores_obj
        if print_out:
            config.logger.info('Train scores:')
            train_scores_obj.printOut()
    
    if print_out or debug:
        scores_obj.printOut()

    ta = add_to_times(times, ta) #9/14

    if remote:
        return scores_obj, pack(cv_slice), cv_counter, times, slice_slices
    return forest, scores_obj, cv_counter, cv_slice, times, slice_slices


def calc_scores_obj(target_vector, prediction_vector, sample_id_vector, weight_vector, feature_names):
    weighted_r2 = r2_score(target_vector, prediction_vector, sample_weight=weight_vector)
    weighted_mse = mean_squared_error(target_vector, prediction_vector, sample_weight=weight_vector)

    r2 = r2_score(target_vector, prediction_vector)
    mse = mean_squared_error(target_vector, prediction_vector)

    # observed_value_threshold = config.binary_thresh

    tv_median = util.median(target_vector)

    test_targets_b = makeBinaryClassifier(target_vector, tv_median)

    # middle_value = (max(prediction_vector) + min(prediction_vector))/2.
    prediction_vector_median = util.median(prediction_vector)

    prediction_vector_b = makeBinaryClassifier(prediction_vector, prediction_vector_median)
    mcc = matthews_corrcoef(test_targets_b, prediction_vector_b)

    #exp_score = mcc / (1.0 + mse)
    # mcc = exp_score #JUST FOR TESTING

    corr, p_value = stats.spearmanr(target_vector, prediction_vector)
    pearson, pearson_p = stats.pearsonr(target_vector, prediction_vector)

    prot_wise_spearmans, mean_spearman, _ = util.calc_protein_wise_corr(target_vector, prediction_vector, sample_id_vector, stats.spearmanr)
    prot_wise_pearsons, mean_pearson, _ = util.calc_protein_wise_corr(target_vector, prediction_vector, sample_id_vector, stats.pearsonr)

    scores_obj = util.Scores(
        mse=mse,
        r2=r2,
        corr=corr,
        mcc=mcc,
        pearson_r=pearson,
        wmse=weighted_mse,
        wr2=weighted_r2,
        n_of_features=len(feature_names),
        mean_spearman=mean_spearman,
        mean_pearson=mean_pearson,
    )
    return scores_obj

def trainForest(
    config: util.Config,
    cross_val_object: CrossValidationSlice | DataSAIL_cv,
    samples_store_id=None,
    samples=None,
    slice_slices: None | list[CrossValidationSlice] | dict[int, list[CrossValidationSlice]] = None,
    distance_map=None,
    repeat=1,
    print_out=False,
    cv_repeat=False,
    skip_scoring=False,
    remote=True,
    debug=False,
    para_number=None,
    skip_feature_selection=False,
    force_confusion=False,
    score_train=True,
    get_first_scores=False,
    cv_interuption=None
) -> tuple[RandomForestRegressor | None, util.Scores, CrossValidationSlice | DataSAIL_cv, None | list[CrossValidationSlice] | dict[int, list[CrossValidationSlice]]]:
    # if cv_repeat is False, the cross_val_object is a cross validation slice object instead
    zero_scores_obj = util.Scores(zero=True)
    # if para_number == 1:

    if config.suppress_remote_forests or debug or config.gpu_mode:
        remote = False
        para_number = None

    if not cv_repeat:
        if len(cross_val_object.feature_names) < 1:
            config.logger.info(f"Call of trainForest without features: {cross_val_object.name}")
            return None, zero_scores_obj, cross_val_object, slice_slices

    if config.verbosity >= 2 or debug:
        config.logger.info(f"Call of trainForest: {repeat=}, {cv_repeat=}, {remote=}, {para_number=}, {skip_feature_selection=}, {debug=}")

    t0 = time.time()

    scores_list = []
    worst_scores = None
    worst_forest = None
    forest = None
    total_times = []
    if not cv_repeat:
        for i in range(0, repeat):  # This can be used to ensure the robustness of the current parameter configuration
            if print_out:
                config.logger.info(f"Training with #of features: {len(cross_val_object.feature_names)} and #of samples: {len(cross_val_object.sub_sampled_train_targets)}")

            if config.regression:
                (
                    forest,
                    scores_obj,
                    cv_counter,
                    cv_slice,
                    reg_forest_times,
                    _slice_slices,
                ) = trainRegressionForest(
                    config,
                    cross_val_object,
                    samples_store_id=samples_store_id,
                    samples=samples,
                    slice_slices=slice_slices,
                    distance_map=distance_map,
                    print_out=print_out,
                    skip_scoring=skip_scoring,
                    debug=debug,
                    skip_feature_selection=skip_feature_selection,
                    overwrite_proc_n=para_number,
                    force_confusion=force_confusion,
                    score_train=score_train,
                )
                total_times = aggregate_times(total_times, reg_forest_times)

            else:
                forest, scores_obj, cv_counter, cv_slice = trainClassificationForest(
                    config,
                    cross_val_object,
                    print_out=print_out,
                    skip_scoring=skip_scoring,
                    skip_feature_selection=skip_feature_selection,
                )
            if repeat > 1:
                if i == 0:
                    worst_scores = scores_obj
                    worst_forest = forest
                elif util.objective_function_criterium(config, worst_scores, scores_obj):  # if worst_scores ar better than scores_obj
                    worst_scores = scores_obj
                    worst_forest = forest
            if config.optimize_mean:
                scores_list.append(scores_obj)
        if repeat > 1:
            scores_obj = worst_scores
            forest = worst_forest
            if config.optimize_mean:
                scores_obj = util.mean_scores(scores_list)

        if slice_slices is None:
            slice_slices: list[CrossValidationSlice] = _slice_slices
        elif _slice_slices is not None:
            for slice_number, slice_slice in enumerate(_slice_slices):
                if slice_slice is not None:
                    slice_slices[slice_number] = slice_slice
        ret_slice_slices = slice_slices

    else:  # This can be used to perform a hyperparameter optimization on the whole dataset
        slice_result_ids = []

        if config.cv_hpo_limiter is not None:
            if config.cv_counters is not None:
                cv_counters = config.cv_counters
            else:
                cv_counters = list(cross_val_object.slices.keys())[0 : config.cv_hpo_limiter]
                config.cv_counters = cv_counters
                if config.verbosity >= 1:
                    config.logger.info(f"\nCross validation in HPO limited to: {cv_counters}\n")
        else:
            cv_counters = cross_val_object.slices.keys()

        if remote:
            if para_number is None:
                para_number = config.proc_n // len(cv_counters)
            else:
                para_number = para_number // len(cv_counters)
            cv_slice_stores = {}
        else:
            cv_slice_stores = None

        for cv_counter in cv_counters:
            cv_slice: CrossValidationSlice = cross_val_object.slices[cv_counter]
            if remote:
                t01 = time.time()
                packed_cv_slice: bytes = pack(cv_slice)
                cv_slice_stores[cv_counter] = packed_cv_slice
                t02 = time.time()
                if config.verbosity >= 3:
                    config.logger.info(f"Time for packing cv_slice {cv_counter=} in trainForest: {t02 - t01} {slice_slices is None=} {para_number=} {(samples_store_id is None)=}")

            if slice_slices is not None and cv_counter in slice_slices:
                s_slice_slices = slice_slices[cv_counter]
            else:
                s_slice_slices = None

            if remote:
                if config.regression:
                    if samples_store_id is None:
                        samples_store_id = ray.put(pack(samples))
                    slice_result_ids.append(
                        trainRegressionForestWrapper.remote(
                            config,
                            packed_cv_slice,
                            samples_store_id=[samples_store_id],
                            slice_slices=s_slice_slices,
                            distance_map=distance_map,
                            print_out=print_out,
                            cv_counter=cv_counter,
                            debug=debug,
                            skip_feature_selection=skip_feature_selection,
                            overwrite_proc_n=para_number,
                            skip_scoring=skip_scoring,
                            force_confusion=force_confusion,
                            score_train=score_train,
                        )
                    )
                else:
                    slice_result_ids.append(
                        trainClassificationForestWrapper.remote(
                            config,
                            packed_cv_slice,
                            print_out=print_out,
                            cv_counter=cv_counter,
                            skip_scoring=skip_scoring,
                            skip_feature_selection=skip_feature_selection,
                        )
                    )
            else:
                if config.regression:
                    slice_result_ids.append(
                        trainRegressionForest(
                            config,
                            cv_slice,
                            samples=samples,
                            samples_store_id=samples_store_id,
                            slice_slices=s_slice_slices,
                            distance_map=distance_map,
                            print_out=print_out,
                            cv_counter=cv_counter,
                            overwrite_proc_n=para_number,
                            debug=debug,
                            skip_feature_selection=skip_feature_selection,
                            skip_scoring=skip_scoring,
                            force_confusion=force_confusion,
                            score_train=score_train,
                        )
                    )
                    if cv_interuption is not None and len(slice_result_ids) == 1:
                        (
                            forest,
                            scores_obj,
                            cv_counter,
                            cv_slice,
                            reg_forest_times,
                            _slice_slices,
                        ) = slice_result_ids[0]
                        margin, best_first_scores = cv_interuption
                        if not util.objective_function_criterium(config, scores_obj, best_first_scores, feature_penalty=config.feature_penalty, margin=margin):
                            return forest, (scores_obj, scores_obj), cross_val_object, slice_slices
                else:
                    slice_result_ids.append(
                        trainClassificationForest(
                            config,
                            cv_slice,
                            print_out=print_out,
                            cv_counter=cv_counter,
                            skip_scoring=skip_scoring,
                            skip_feature_selection=skip_feature_selection,
                        )
                    )

        if remote:
            results = ray.get(slice_result_ids)
        else:
            results = slice_result_ids

        ret_slice_slices: dict[int, list[CrossValidationSlice]] = slice_slices
        if ret_slice_slices is None:
            ret_slice_slices = {}

        for res in results:
            if remote:
                (
                    scores_obj,
                    packed_cv_slice,
                    cv_counter,
                    reg_forest_times,
                    _slice_slices,
                ) = res
                cv_slice = unpack(packed_cv_slice)
            else:
                (
                    forest,
                    scores_obj,
                    cv_counter,
                    cv_slice,
                    reg_forest_times,
                    _slice_slices,
                ) = res

            total_times = aggregate_times(total_times, reg_forest_times)

            if cv_slice is not None:
                del cross_val_object.slices[cv_counter]
                cross_val_object.slices[cv_counter] = cv_slice
                if remote:
                    cv_slice_stores = None

            if cv_counter not in ret_slice_slices:
                ret_slice_slices[cv_counter] = _slice_slices
            elif ret_slice_slices[cv_counter] is None:
                ret_slice_slices[cv_counter] = _slice_slices
            elif _slice_slices is not None:
                ret_slice_slices[cv_counter] = _slice_slices

            if scores_obj is None:
                raise "Scores must not be None here"

            if config.optimize_mean:
                scores_list.append(scores_obj)
            else:
                if worst_scores is None:
                    worst_scores = scores_obj
                elif util.objective_function_criterium(config, worst_scores, scores_obj):  # if worst_scores ar better than scores_obj
                    worst_scores = scores_obj
        del results

        if config.optimize_mean:
            first_scores = scores_list[0]
            scores_obj = util.mean_scores(scores_list)
            if get_first_scores:
                scores_obj = (first_scores, scores_obj)
        else:
            scores_obj = worst_scores
            forest = worst_forest

    if config.verbosity >= 2:
        print_times(total_times, label = 'Train forest')

    t1 = time.time()
    if config.verbosity >= 2:
        config.logger.info(f"Time for trainForest: {t1 - t0}")

    return forest, scores_obj, cross_val_object, ret_slice_slices
