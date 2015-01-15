__author__ = 'ardila'

from BehavioralBenchmark.experiments import hvm_dense_smp_v6_2rpw
import os
import numpy as np
import dldata.stimulus_sets.hvm as hvm

# This is the image set used in the first set of image level experiments, 2 from each object
ImageSet1_inds = np.load(os.path.join(os.path.dirname(hvm_dense_smp_v6_2rpw.__file__), 'inds.npy'))

def get_decoder_model_by_name(decoder_model_name):
    if decoder_model_name == 'StandardModel':
        all_ids = list(hvm.HvMWithDiscfade().meta['_id'])
        ImageSet1 = [all_ids[i] for i in ImageSet1_inds]
        not_ImageSet1 = list(set(all_ids) - set(ImageSet1))
        model = {'name': 'StandardModel', 'train_q': {'var': ['V6'], '_id': not_ImageSet1}, 'test_q': {'_id': ImageSet1},
                         'labelfunc': 'category', 'split_by': None, 'npc_train': None, 'npc_test': len(ImageSet1),
                         'npc_validate': 0, 'num_splits': 1, 'metric_screen': "classifier",
                         'metric_kwargs': {'model_type': 'svm.LinearSVC',
                                           'model_kwargs': {'GridSearchCV_params': {'C':
                                                                                        [1e-5, 1e-4, 1e-3, .25e-3, .5e-3,
                                                                                         .75e-3,
                                                                                         1e-2, .25e-2, .5e-2, .75e-2, 1e-1, 1,
                                                                                         10]},
                                                            'GridSearchCV_kwargs': {'n_jobs': 13}}
                         }}
        return model

    else:
        raise ValueError, 'Model not recognized'
