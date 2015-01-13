__author__ = 'ardila'

from BehavioralBenchmark.experiments import hvm_dense_smp_v6_2rpw
import os
import numpy as np

ImageSet1 = np.load(os.path.join(hvm_dense_smp_v6_2rpw.__file__, 'inds.npy'))

# labelfunc = 'category'
# n_train = None
# n_test = len(ImageSetTest)
# n_splits = 1
# Classifier = 'svm.LinearSVC'#'svm.LinearSVC'
# eval_config     = { "train_q": lambda x : ((x['var'] in 'V6' )and (x['_id'] not in ImageSetTest)),
#                     "test_q": lambda x : (x['_id'] in ImageSetTest) ,
#                     "labelfunc": labelfunc,
#                     "split_by": None,
#                     "npc_train": n_train,
#                     "npc_test": n_test,
#                     "npc_validate": 0,
#                     "num_splits": n_splits,
#                     "metric_screen": "classifier",
#                     "metric_kwargs": {'model_type': Classifier ,
#                                       'model_kwargs': {'GridSearchCV_params':{'C':
#                                     [1e-5, 1e-4, 1e-3,.25e-3, .5e-3, .75e-3, 1e-2, .25e-2, .5e-2, .75e-2,  1e-1, 1, 10]}}
#                                                  }
#                                 }

