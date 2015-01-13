__author__ = 'ardila'


import numpy as np

def feature_split(n_features, n_samples, n_bootstrap):
    """
    Select indexes up to n_features without replacement or overlap
    :param n_features: number of features to select from
    :param n_samples: number of distinctly sized samples (2 at a time, 3 at a time... etc)
    :param n_bootstrap: number of times to shuffle and reselect per size
    """
    feature_splits = []
    # The least you can take is 2
    min_features = 2
    # The most you can take without replacement or overlap is n_features/2
    max_features = n_features/2

    # Therefore this is the max amount of distinct sizes of samples to take
    max_n_samples =  max_features-min_features+1

    assert max_n_samples >= n_samples, 'Max value of n_samples is %s' %max_n_samples

    chunk_sizes = [int(size) for size in np.round(np.linspace(min_features, max_features, n_samples))]

    all_inds = range(n_features)

    rng = np.random.RandomState(0)

    for chunk_size in chunk_sizes:
        n_chunks  = int(n_features)/int(chunk_size)
        for _ in range(n_bootstrap):
            rng.shuffle(all_inds)
            for chunk_ind in range(0, n_chunks):
                feature_splits.append(all_inds[chunk_ind*chunk_size:(chunk_ind+1)*chunk_size])

    return feature_splits
