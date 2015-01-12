__author__ = 'ardila'
import sys
from feature_split import feature_split

feature_name = sys.argv[1]

feature_splits = feature_split(n_features, n_sample, n_bootstrap)


for feature_split in feature_splits:
    #Code to submit a job to slurm that runs store_feature_results on the named features