from BehavioralBenchmark.ImageLevel import store_feature_results, feature_loader, decoder_models

__author__ = 'ardila'
import sys
import gridfs
import pymongo as pm

DB = pm.MongoClient(port=22334)['ModelBehavior']

feature_name = sys.argv[1]

feature_split = [int(ind) for ind in sys.argv[2].split(',')]

# Load feature from name

features, meta = feature_loader.get_features_by_name(feature_name)

# For now, this always uses the standard decoder model, but this can be reconfigured easily

decoder_model = decoder_models.StandardModel

gridfs_name = '_'.join([decoder_model['name'], feature_name, 'results'])

fs = gridfs.GridFS(DB, gridfs_name) # Decide where to store things

additional_info = {'feature_split': feature_split}

store_feature_results.store_subsampled_feature_results(features, meta,
                                                       decoder_model, fs,
                                                       feature_split,
                                                       additional_info)
