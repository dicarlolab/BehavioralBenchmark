from BehavioralBenchmark.ImageLevel import store_feature_results, feature_loader

__author__ = 'ardila'
import sys


feature_name = sys.argv[1]

feature_split = [int(ind) for ind in sys.argv[2].split(',')]

# Code to load feature from name

features, meta = feature_loader.get_features_by_name(feature_name)

#Todo: model name option, for now fixed


decoder_model = None#Get model_config from ipython notebooks

fs = None#Decide where to store things

additional_info = None#Decide on additional info

store_feature_results.store_subsampled_feature_results(features, meta, eval_config, fs, feature_split, additional_info)
