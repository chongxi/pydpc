# first line: 1
def _hdbscan_boruvka_kdtree(X, min_samples=5, alpha=1.0,
                            metric='minkowski', p=2, leaf_size=40,
                            approx_min_span_tree=True,
                            gen_min_span_tree=False,
                            core_dist_n_jobs=4):
    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')

    if leaf_size < 3:
        leaf_size = 3

    if core_dist_n_jobs < 1:
        raise ValueError('Parallel core distance computation requires 1 or more jobs!')

    size = X.shape[0]
    min_samples = min(size - 1, min_samples)

    tree = KDTree(X, metric=metric, leaf_size=leaf_size)
    alg = KDTreeBoruvkaAlgorithm(tree, min_samples, metric=metric, leaf_size=leaf_size // 3,
                                 approx_min_span_tree=approx_min_span_tree, n_jobs=core_dist_n_jobs)
    min_spanning_tree = alg.spanning_tree()
    #Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]
    #Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)

    if gen_min_span_tree:
        return single_linkage_tree, min_spanning_tree
    else:
        return single_linkage_tree, None
