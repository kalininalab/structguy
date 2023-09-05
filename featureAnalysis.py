import learn
import sys
import sampleSpace
import numpy as np
import ray
import trainForest

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
            pre_confusion_map[feat] = tree_confusion_map[feat]
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

def calcSliceConfusion(forest, cv_slice, remote = True, para_number = None, err_warping_exp = 1, goodwill_interval = 0.25, norm_exp = 1.2):
    pre_confusion_map = {}
    pre_raw_confusion_map = {}

    if para_number == 1:
        remote = False

    if remote:
        store = ray.put((cv_slice.test_feature_matrix, cv_slice.test_targets, cv_slice.feature_names))

    result_ids = []

    n_trees = len(forest.estimators_)
    if para_number is None:
        para_number = n_trees

    for i,tree in enumerate(forest.estimators_):
        if remote:
            if i == para_number:
                break
            result_ids.append(trainForest.nested_ray_wrapper_for_conf_map_calculation.remote(tree, store, err_warping_exp = err_warping_exp, goodwill_interval = goodwill_interval))
        else:
            result_ids.append(calcConfusionMap(tree, (cv_slice.test_feature_matrix, cv_slice.test_targets, cv_slice.feature_names), err_warping_exp = err_warping_exp, goodwill_interval=goodwill_interval))

    if remote:
        while True:
            ready, not_ready = ray.wait(result_ids)
            new_result_ids = []
            if len(ready) > 0:
                for tree_confusion_map, raw_tree_confusion_map in ray.get(ready):
                    pre_confusion_map = add_tree_CM(tree_confusion_map, pre_confusion_map)
                    pre_raw_confusion_map = add_tree_RCM(raw_tree_confusion_map, pre_raw_confusion_map)
                    if i < n_trees:
                        tree = forest.estimators_[i]
                        new_result_ids.append(trainForest.nested_ray_wrapper_for_conf_map_calculation.remote(tree, store, goodwill_interval = goodwill_interval, err_warping_exp = err_warping_exp))
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
    return calcConfusionMap(estimator, store, err_warping_exp = err_warping_exp, goodwill_interval=goodwill_interval)

def calcConfusionMap(estimator, store, err_warping_exp = 1, goodwill_interval = 0.25):
    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    X_test, y_test, feature_names = store

    y_pred = estimator.predict(X_test)

    feature = estimator.tree_.feature

    node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    confusion_map = {}
    raw_confusion_map = {}

    for sample_id, true_value in enumerate(y_test):
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]

        raw_err = abs(true_value - y_pred[sample_id])
        warped_err = calc_confusion(raw_err, goodwill_interval, err_warping_exp)

        for node_id in node_index:

            if leave_id[sample_id] == node_id:
                continue

            feat = feature_names[feature[node_id]]

            if not feat in confusion_map:
                confusion_map[feat] = [warped_err,1]
                raw_confusion_map[feat] = [raw_err]
            else:
                confusion_map[feat][0] += warped_err
                #confusion_map[feat][0] += err
                confusion_map[feat][1] += 1
                raw_confusion_map[feat].append(raw_err)
    return confusion_map, raw_confusion_map

def calc_confusion(raw_err, goodwill_interval, err_warping_exp):
    err = raw_err - goodwill_interval
    if err >= 0:
        sign = 1
    else:
        sign = -1
    warped_err = sign * (abs(err) ** err_warping_exp)
    return warped_err

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

    #full_slice.slices[0].reorderByNames(feature_names)

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


