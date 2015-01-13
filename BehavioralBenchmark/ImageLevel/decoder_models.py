__author__ = 'ardila'

from BehavioralBenchmark.experiments import hvm_dense_smp_v6_2rpw
import os
import numpy as np


# This is the image set used in the first set of image level experiments, 2 from each object
ImageSet1 = np.load(os.path.join(os.path.dirname(hvm_dense_smp_v6_2rpw.__file__), 'inds.npy'))

StandardModel = dict(name='StandardModel',
                     train_q=lambda x: ((x['var'] in 'V6' ) and (x['_id'] not in ImageSet1)),
                     test_q=lambda x: (x['_id'] in ImageSet1), labelfunc='category', split_by=None, npc_train=None,
                     npc_test=len(ImageSet1), npc_validate=0, num_splits=1, metric_screen="classifier",
                     metric_kwargs={'model_type': 'svm.LinearSVC',
                                    'model_kwargs': {'GridSearchCV_params': {'C':
                                                                                 [1e-5, 1e-4, 1e-3, .25e-3, .5e-3,
                                                                                  .75e-3,
                                                                                  1e-2, .25e-2, .5e-2, .75e-2, 1e-1, 1,
                                                                                  10],
                                                                             'n_jobs': 13}}
                     })

