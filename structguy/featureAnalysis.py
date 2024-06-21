import sys
import numpy as np
import ray
import statistics
import time
import random

from structguy import learn
from structman.base_utils.base_utils import calculate_chunksizes, pack, unpack

def findAndAnalyseInterestingSample(forest, cv_slice, config):
    worst_sample, best_effect_sample = findInterestingSamples(forest, cv_slice, config)

    repr_tree = findRepresentativeTree(forest, cv_slice.test_feature_matrix[best_effect_sample])

    tracebackTree(repr_tree, cv_slice.test_feature_matrix[best_effect_sample], cv_slice.feature_names, cv_slice.test_targets[best_effect_sample])

def findInterestingSamples(forest, cv_slice, config):
    y_pred = forest.predict(cv_slice.test_feature_matrix)

    effect_thresh = 0.2
    max_error = 0.
    min_error = 5.0
    for n, pred in enumerate(y_pred):
        err = abs(pred - cv_slice.test_targets[n])
        if err > max_error:
            max_error = err
            worst_sample = n
        if cv_slice.test_targets[n] < effect_thresh:
            if err < min_error:
                min_error = err
                best_effect_sample = n
    print('worst sample:', worst_sample, max_error)
    print('best sample with effect:', best_effect_sample, min_error, cv_slice.train_sample_ids[best_effect_sample])
    return worst_sample, best_effect_sample

def analysisSample(forest, sample_id, cv_slice, config):
    low_tree, high_tree = findMostImportantTrees(forest, cv_slice.test_feature_matrix[sample_id], cv_slice.test_targets[sample_id])

    print('=== Low tree traceback ===')
    tracebackTree(low_tree, cv_slice.test_feature_matrix[sample_id], cv_slice.feature_names, cv_slice.test_targets[sample_id])
    print('==========================\n')

    print('=== High tree traceback ==')
    tracebackTree(high_tree, cv_slice.test_feature_matrix[sample_id], cv_slice.feature_names, cv_slice.test_targets[sample_id])
    print('==========================')

def findMostImportantTrees(forest, feat_vec, true_value):

    print('Looking for the most important trees, true value:',true_value)

    minimum = 5.0
    maximum = -5.0

    for tree_id, tree in enumerate(forest.estimators_):
        pred = tree.predict([feat_vec])
        if pred < minimum:
            minimum = pred
            low_tree = tree_id
        if pred > maximum:
            maximum = pred
            high_tree = tree_id

    return forest.estimators_[low_tree], forest.estimators_[high_tree]

def findRepresentativeTree(forest, feat_vec):

    forest_pred = forest.predict([feat_vec])[0]

    d = 5.

    for tree_id, tree in enumerate(forest.estimators_):
        pred = tree.predict([feat_vec])
        if abs(pred - forest_pred) < d:
            d = abs(pred - forest_pred)
            repr_tree = tree_id

    return forest.estimators_[repr_tree]


def get_base_stats(forest):
    n_of_trees = len(forest.estimators_)
    total_nodes = 0
    for tree in forest.estimators_:
        n_nodes = tree.tree_.node_count
        total_nodes += n_nodes
    return n_of_trees, total_nodes


@ray.remote(max_calls = 1)
def calculate_tree_weights(store, chunk):
    forest = store

    output = []

    for sample_id, pred_x, feat_vec in chunk:
        weight_vector = []
        tree_preds = []
        for tree in forest.estimators_:
            tree_pred = tree.predict([feat_vec])[0]
            tree_preds.append(tree_pred)
            weight = max([0, abs(pred_x - tree_pred)])
            weight_vector.append(weight)

        pred_std = statistics.stdev(tree_preds)
        output.append((sample_id, weight_vector, pred_std))
    return output

def explain_decisions(config, forest, prediction_vector, feat_vecs, feature_names):
    t0 = time.time()

    weight_vectors = [0]*len(feat_vecs)
    weighted_feat_threshs = []
    pred_std_vector = [0]*len(feat_vecs)
    store = ray.put(forest)

    small_chunksize, big_chunksize, n_of_small_chunks, n_of_big_chunks = calculate_chunksizes(config.proc_n, len(feat_vecs))

    chunk_process_ids = []
    chunk = []

    for sample_id, feat_vec in enumerate(feat_vecs):
        weighted_feat_threshs.append({}) #just for initialization
        chunk.append((sample_id, prediction_vector[sample_id], feat_vec))
        if n_of_small_chunks > 0 and len(chunk_process_ids) < n_of_small_chunks:
            if len(chunk) == small_chunksize:
                chunk_process_ids.append(calculate_tree_weights.remote(store, chunk))
                chunk = []
                continue
        else:
            if len(chunk) == big_chunksize:
                chunk_process_ids.append(calculate_tree_weights.remote(store, chunk))
                chunk = []
                continue

    para_results = ray.get(chunk_process_ids)

    for chunk_result in para_results:
        for (sample_id, weight_vector, pred_std) in chunk_result:
            weight_vectors[sample_id] = weight_vector
            pred_std_vector[sample_id] = pred_std


    t1 = time.time()
    print(f'Explain decisions part 1: {t1-t0}')

    small_chunksize, big_chunksize, n_of_small_chunks, n_of_big_chunks = calculate_chunksizes(config.proc_n, len(forest.estimators_))

    chunk_process_ids = []
    chunk = []
    store = ray.put((feat_vecs, feature_names))

    for tree_id, tree in enumerate(forest.estimators_):
        chunk.append((tree_id, tree))
        if n_of_small_chunks > 0 and len(chunk_process_ids) < n_of_small_chunks:
            if len(chunk) == small_chunksize:
                chunk_process_ids.append(calc_thresh_maps.remote(chunk, store))
                chunk = []
                continue
        else:
            if len(chunk) == big_chunksize:
                chunk_process_ids.append(calc_thresh_maps.remote(chunk, store))
                chunk = []
                continue

    t2 = time.time()
    print(f'Explain decisions part 2: {t2-t1}')

    para_results = ray.get(chunk_process_ids)

    t3 = time.time()
    print(f'Explain decisions part 3: {t3-t2}, {len(para_results)}')

    for packed_chunk_result in para_results:
        chunk_result = unpack(packed_chunk_result)
        for tree_id, threshold_maps in chunk_result:
            for sample_id, threshold_map in enumerate(threshold_maps):
                weight = weight_vectors[sample_id][tree_id]
                for feat_name in threshold_map:
                    threshs = threshold_map[feat_name]
                    if feat_name not in weighted_feat_threshs[sample_id]:
                        weighted_feat_threshs[sample_id][feat_name] = [0, []]
                    weighted_feat_threshs[sample_id][feat_name][0] += weight*len(threshs)
                    weighted_feat_threshs[sample_id][feat_name][1] += threshs

    feat_name_backmap = {}
    for feat_number, feat_name in enumerate(feature_names):
        feat_name_backmap[feat_name] = feat_number

    t4 = time.time()
    print(f'Explain decisions part 4: {t4-t3}')

    decisions = []
    for sample_id, weighted_feat_thresh_map in enumerate(weighted_feat_threshs):
        processed_feat_thresh_vector = []
        for feat_name in weighted_feat_thresh_map:
            feat_value = feat_vecs[sample_id][feat_name_backmap[feat_name]]
            total_weight, all_threshs = weighted_feat_thresh_map[feat_name]
            l = None
            r = None
            for thresh in all_threshs:
                if thresh <= feat_value:
                    if l is None:
                        l = thresh
                    elif thresh > l:
                        l = thresh
                else:
                    if r is None:
                        r = thresh
                    elif thresh < r:
                        r = thresh
            processed_feat_thresh_vector.append((feat_name, total_weight, l, r))
        processed_feat_thresh_vector.sort(key=lambda x:x[1],reverse=True)
        decisions.append(processed_feat_thresh_vector)

    t5 = time.time()
    print(f'Explain decisions part 5: {t5-t4}')
    return decisions, pred_std_vector


def tracebackTree(t, feat_vec, feature_names, true_value):
    #indicator, n_nodes_ptr = forest.decision_path([feat_vec])

    #print(feature_names)
    #print(feat_vec)

    X_test = np.array([feat_vec])
    y_test = [true_value]

    n_nodes_ = t.tree_.node_count
    children_left_ = t.tree_.children_left
    children_right_ = t.tree_.children_right
    feature_ = t.tree_.feature
    threshold_ = t.tree_.threshold

    explore_tree(t, n_nodes_, children_left_, children_right_, feature_, threshold_, X_test, y_test,
                print_tree = False, sample_id=0, feature_names = feature_names)

def tracebackDecision(estimator, feat_vec, feature_names, true_value):
    #indicator, n_nodes_ptr = forest.decision_path([feat_vec])

    #print(feature_names)
    #print(feat_vec)

    X_test = np.array([feat_vec])
    y_test = [true_value]

    n_nodes_ = [t.tree_.node_count for t in estimator.estimators_]
    children_left_ = [t.tree_.children_left for t in estimator.estimators_]
    children_right_ = [t.tree_.children_right for t in estimator.estimators_]
    feature_ = [t.tree_.feature for t in estimator.estimators_]
    threshold_ = [t.tree_.threshold for t in estimator.estimators_]

    explore_tree(estimator.estimators_[0], n_nodes_[0], children_left_[0], children_right_[0], feature_[0], threshold_[0], X_test, y_test,
                print_tree = False, sample_id=0, feature_names = feature_names)

def add_tree_CM(tree_confusion_map, pre_confusion_map):
    for feat in tree_confusion_map:
        if not feat in pre_confusion_map:
            pre_confusion_map[feat] = list(tree_confusion_map[feat])
        else:
            pre_confusion_map[feat][0] += tree_confusion_map[feat][0]
            pre_confusion_map[feat][1] += tree_confusion_map[feat][1]
    return pre_confusion_map

def add_tree_RCM(tree_raw_confusion_map, pre_raw_confusion_map):
    for feat in tree_raw_confusion_map:
        if not feat in pre_raw_confusion_map:
            pre_raw_confusion_map[feat] = tree_raw_confusion_map[feat]
        else:
            pre_raw_confusion_map[feat] += tree_raw_confusion_map[feat]
    return pre_raw_confusion_map


def calcSliceConfusion(forest, cv_slice, samples = None, samples_store_id = None, remote = True, para_number = None, err_warping_exp = 1, goodwill_interval = 0.25, norm_exp = 1.2):
    pre_confusion_map = {}
    pre_raw_confusion_map = {}

    if para_number == 1:
        remote = False

    if remote:
        if samples_store_id is None:
            samples_store_id = ray.put(samples)
        store = ray.put((samples_store_id, pack(cv_slice)))
    else:
        if samples is None:
            samples = ray.get(samples_store_id)
        test_feature_matrix = cv_slice.get_test_feature_matrix(samples)

    result_ids = []

    n_trees = len(forest.estimators_)
    if para_number is None:
        para_number = n_trees

    print(f'Call of calcSliceConfusion with number of trees: {n_trees}, para_number: {para_number}')

    for i,tree in enumerate(forest.estimators_):
        if remote:
            if i == para_number:
                break
            result_ids.append(calcConfusionMapWrapper.remote(tree, store, err_warping_exp = err_warping_exp, goodwill_interval = goodwill_interval))
        else:
            result_ids.append(calcConfusionMap(tree, test_feature_matrix, cv_slice.test_targets, cv_slice.feature_names, err_warping_exp = err_warping_exp, goodwill_interval=goodwill_interval))

    if remote:
        while True:
            ready, not_ready = ray.wait(result_ids)
            new_result_ids = []
            if len(ready) > 0:
                for package in ray.get(ready):
                    tree_confusion_map, raw_tree_confusion_map = unpack(package)
                    pre_confusion_map = add_tree_CM(tree_confusion_map, pre_confusion_map)
                    pre_raw_confusion_map = add_tree_RCM(raw_tree_confusion_map, pre_raw_confusion_map)
                    if i < n_trees:
                        tree = forest.estimators_[i]
                        new_result_ids.append(calcConfusionMapWrapper.remote(tree, store, goodwill_interval = goodwill_interval, err_warping_exp = err_warping_exp))
                        i += 1
            result_ids = new_result_ids + not_ready
            if len(result_ids) == 0:
                break
        del result_ids
        del store
    else:
        for tree_confusion_map, raw_tree_confusion_map in result_ids:
            pre_confusion_map = add_tree_CM(tree_confusion_map, pre_confusion_map)
            pre_raw_confusion_map = add_tree_RCM(raw_tree_confusion_map, pre_raw_confusion_map)


    confusion_map = []
    for feat in cv_slice.feature_names:
        if feat in pre_confusion_map:
            confusion_map.append([feat, round(pre_confusion_map[feat][0]/(pre_confusion_map[feat][1]**norm_exp), 5)])
            #confusion_map.append([feat,pre_confusion_map[feat][0]])
        else:
            confusion_map.append([feat, 1.-goodwill_interval])

    confusion_map = sorted(confusion_map, key=lambda x: x[1], reverse=True)

    return confusion_map, pre_raw_confusion_map

def confusion_map_from_raw_confusion_map(cv_slice, raw_confusion_map, goodwill_interval, err_warping_exp, norm_exp):
    confusion_map = []
    for feat in raw_confusion_map:
        warped_err_sum = 0.
        for raw_err in raw_confusion_map[feat]:
            warped_err = calc_confusion(raw_err, goodwill_interval, err_warping_exp)
            warped_err_sum += warped_err
        confusion_map.append([feat, round(warped_err_sum/(len(raw_confusion_map[feat])**norm_exp), 5)])
    for feat in cv_slice.feature_names:
        if feat not in raw_confusion_map:
            confusion_map.append([feat, 1.-goodwill_interval])
    confusion_map = sorted(confusion_map, key=lambda x: x[1], reverse=True)
    return confusion_map

@ray.remote(max_calls = 1)
def calcConfusionMapWrapper(estimator, store, err_warping_exp = 1, goodwill_interval = 0.25):
    samples_store_id, packed_cv_slice = store
    cv_slice = unpack(packed_cv_slice)
    samples = ray.get(samples_store_id)

    X_test = cv_slice.get_test_feature_matrix(samples)
    y_test = cv_slice.test_targets
    feature_names = cv_slice.feature_names

    return pack(calcConfusionMap(estimator, X_test, y_test, feature_names, err_warping_exp = err_warping_exp, goodwill_interval=goodwill_interval))

def calcConfusionMap(estimator, X_test, y_test, feature_names, err_warping_exp = 1, goodwill_interval = 0.25, max_samples = 10000):
    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    if len(X_test) > max_samples:
        subselection = random.sample(range(len(X_test)), max_samples)
        X_test = [X_test[n] for n in subselection]
        y_test = [y_test[n] for n in subselection]

    y_pred = estimator.predict(X_test)

    feature = estimator.tree_.feature

    node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(X_test)

    #print(f'LLI: {len(leave_id)}')

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    confusion_map = {}
    raw_confusion_map = {}

    for sample_id, true_value in enumerate(y_test):

        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]

        raw_err = abs(true_value - y_pred[sample_id])
        warped_err = calc_confusion(raw_err, goodwill_interval, err_warping_exp)

        #print(f'LNI: {len(node_index)}')

        for node_id in node_index:

            if leave_id[sample_id] == node_id:
                continue

            feat = feature_names[feature[node_id]]

            #print(feat)

            if not feat in confusion_map:
                confusion_map[feat] = [warped_err,1]
                raw_confusion_map[feat] = [raw_err]
            else:
                confusion_map[feat][0] += warped_err
                #confusion_map[feat][0] += err
                confusion_map[feat][1] += 1
                raw_confusion_map[feat].append(raw_err)

    #print(f'Add the end of calcConfusionMap: {len(confusion_map)} {len(raw_confusion_map)}')
    return confusion_map, raw_confusion_map

def calc_confusion(raw_err, goodwill_interval, err_warping_exp):
    err = raw_err - goodwill_interval
    if err >= 0:
        sign = 1
    else:
        sign = -1
    warped_err = sign * (abs(err) ** err_warping_exp)
    return warped_err

def calc_tree_threshold_maps(tree, feat_vecs, feature_names):
    node_indicator = tree.decision_path(feat_vecs)
    leave_id_vector = tree.apply(feat_vecs)

    threshold_maps = []
    for sample_id, feat_vec in enumerate(feat_vecs):
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]
        threshold_map = {}
        for node_id in node_index:
            if leave_id_vector[sample_id] == node_id:
                continue
            feat_name = feature_names[tree.tree_.feature[node_id]]
            thresh = tree.tree_.threshold[node_id]
            if feat_name not in threshold_map:
                threshold_map[feat_name] = []
            threshold_map[feat_name].append(thresh)
        threshold_maps.append(threshold_map)
    return threshold_maps

@ray.remote(max_calls = 1)
def calc_thresh_maps(trees, store):
    feat_vecs, feature_names = store
    output = []
    for tree_id, tree in trees:
        threshold_maps = calc_tree_threshold_maps(tree, feat_vecs, feature_names)
        output.append((tree_id, threshold_maps))
    return pack(output)

def explore_tree(estimator, n_nodes, children_left,children_right, feature, threshold, X_test, y_test,
                print_tree = False, sample_id=0, feature_names=None):

    if not feature_names:
        feature_names = feature


    assert len(feature_names) == X_test.shape[1], "The feature names do not match the number of features."
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes"
          % n_nodes)
    if print_tree:
        print("Tree structure: \n")
        for i in range(n_nodes):
            if is_leaves[i]:
                print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            else:
                print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                      "node %s."
                      % (node_depth[i] * "\t",
                         i,
                         children_left[i],
                         feature[i],
                         threshold[i],
                         children_right[i],
                         ))
            print("\n")
        print()

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    #sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    #print(X_test[sample_id,:])

    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        # tabulation = " "*node_depth[node_id] #-> makes tabulation of each level of the tree
        tabulation = ""
        if leave_id[sample_id] == node_id:
            print("%s==> Predicted leaf index \n"%(tabulation))
            #continue

        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("%sdecision id node %s : (X_test[%s, '%s'] (= %s) %s %s)"
              % (tabulation,
                 node_id,
                 sample_id,
                 feature_names[feature[node_id]],
                 X_test[sample_id, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))
    print("%sPrediction for sample %d: %s (true value: %s)"%(tabulation,
                                          sample_id,
                                          estimator.predict(X_test)[sample_id],
                                          y_test[sample_id]))

    if sample_id > 0:
        # For a group of samples, we have the following common node.
        sample_ids = [sample_id, 1]
        common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                        len(sample_ids))

        common_node_id = np.arange(n_nodes)[common_nodes]

        print("\nThe following samples %s share the node %s in the tree"
              % (sample_ids, common_node_id))
        print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))

        for sample_id_ in sample_ids:
            print("Prediction for sample %d: %s"%(sample_id_,
                                              estimator.predict(X_test)[sample_id_]))

if __name__ == "__main__":
    fn = sys.argv[1]
    #forest, feature_names, config = learn.loadModel(fn)

    #samples = learn.createTrainingSet(config,config.session,infile=sys.argv[2])

    #samples.oneHotifyAll()

    #full_slice = sampleSpace.FullSlice(samples, config)


    #feat_vec = full_slice.slices[0].train_feature_matrix[14]
    #true_value = full_slice.slices[0].train_targets[14]

    #findMostImportantTrees(forest, feat_vec, true_value)

    #tracebackDecision(forest, feat_vec, feature_names, true_value)


    forests, cross_val_object, config = learn.loadCV(fn)

    for cv_counter in cross_val_object.slices:
        print('Feature Analysis for slice:', cv_counter)
        cv_slice = cross_val_object.slices[cv_counter]

        forest = forests[cv_counter]

        conf_map = calcSliceConfusion(forest, cv_slice)

        print(conf_map)
        """
        worst_sample, best_effect_sample = findInterestingSamples(forest, cv_slice, config)

        print('================= Analyse worst sample ==============================')
        analysisSample(forest, worst_sample, cv_slice, config)
        print('=====================================================================\n\n')

        print('================= Analyse best sample with effect ===================')
        analysisSample(forest, best_effect_sample, cv_slice, config)
        print('=====================================================================\n\n\n')
        """


