__author__ = 'ardila'
import sys
import os
from feature_split import feature_split

feature_name = sys.argv[1]

feature_splits = feature_split(n_features, n_sample, n_bootstrap)


for feature_split in feature_splits:
    # Submit a job to slurm that runs store_feature_results on the named features
    feature_split = ','.join(str(ind) for ind in feature_split)
    command = 'sbatch -n 11 --mem=250000 run_feature_subsample.sh %s %s'%(feature_name, feature_split)
    os.system(command)