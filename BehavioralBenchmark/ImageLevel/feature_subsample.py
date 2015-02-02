from BehavioralBenchmark.ImageLevel import store_feature_results, feature_loader, decoder_models

__author__ = 'ardila'
import sys
import gridfs
import pymongo as pm



feature_name = sys.argv[1]

decoder_model_name = 'StandardModel' # Can make this an option later

feature_split = [int(ind) for ind in sys.argv[2].split(',')]

# Load feature from name

features, meta = feature_loader.get_features_by_name(feature_name)

# For now, this always uses the standard decoder model, but this can be reconfigured easily

decoder_model = decoder_models.get_decoder_model_by_name(decoder_model_name)


fs = store_feature_results.get_gridfs(decoder_model_name=decoder_model_name, # Decide where to store things
                                      feature_name=feature_name)

additional_info = {'feature_split': feature_split}

store_feature_results.store_subsampled_feature_results(features, meta,
                                                       decoder_model, fs,
                                                       feature_split,
                                                       additional_info)
