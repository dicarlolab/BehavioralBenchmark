__author__ = 'ardila'



def get_features_by_name(feature_name):

# TODO: Add support for NYU and other features

    if feature_name == 'IT':
        import dldata.stimulus_sets.hvm as hvm
        dataset = hvm.HvMWithDiscfade()
        features = dataset.neuronal_features[:, dataset.IT_NEURONS]
        meta = dataset.meta

    elif feature_name == 'V4':
        import dldata.stimulus_sets.hvm as hvm
        dataset = hvm.HvMWithDiscfade()
        features = dataset.neuronal_features[:, dataset.V4_NEURONS]
        meta = dataset.meta

    return features, meta


def get_size_by_name(feature_name):
    # TODO this could be more efficient...
    return get_features_by_name(feature_name).shape[1]
