__author__ = 'ardila'
import sys
import os
from feature_split import feature_split
from feature_loader import get_size_by_name

### Used to distribute jobs on openmind for feature subsampling
#### Usage: python feature_subsample_distribute_jobs.py 'feature_name'
#### Make sure feature_name is recognized by feature_loader.py


feature_name = sys.argv[1]


n_features = get_size_by_name(feature_name)
feature_splits = feature_split(n_features = n_features,
                               n_samples = 40,
                               n_bootstrap = 2,
                               max_samples_per_size = 50)


for feature_split in feature_splits:
    # Submit a job to slurm that runs store_feature_results on the named features
    feature_split = ','.join(str(ind) for ind in feature_split)
    command = 'sbatch -n 11 --mem=250000 run_feature_subsample.sh %s %s'%(feature_name, feature_split)
    os.system(command)