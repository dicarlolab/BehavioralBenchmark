__author__ = 'ardila'
import sys
import os
from feature_split import feature_split_large
from feature_loader import get_size_by_name
import store_feature_results
import pymongo as pm
import time


### Used to distribute jobs on openmind for feature subsampling
#### Usage: python feature_subsample_distribute_jobs.py 'feature_name'
#### Make sure feature_name is recognized by feature_loader.py


feature_name = sys.argv[1]
decoder_model_name = sys.argv[2]

coll = store_feature_results.get_file_collection(decoder_model_name=decoder_model_name,
                                                 feature_name=feature_name)



precalculated = set([str(e['feature_split']) for e in coll.find()])


n_features = get_size_by_name(feature_name)
feature_splits = feature_split_large(n_features = n_features,
                               n_samples = 20,
                               n_bootstrap = 3)
print len(precalculated)/float(len(feature_splits))

for feature_split in feature_splits[-2:]:
    if str(feature_split) not in precalculated:
        # Submit a job to slurm that runs store_feature_results on the named features
        time.sleep(1)
        feature_split = ','.join(str(ind) for ind in feature_split)
        command = 'sbatch -n 1 --mem=2000 run_feature_subsample.sh %s %s %s'%(feature_name,
                                                                              feature_split,
                                                                              decoder_model_name)
        print command
        os.system(command)
