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
        imageset1 = [all_ids[i] for i in ImageSet1_inds]
        not_imageset1 = list(set(all_ids) - set(imageset1))
        model = dict(name='StandardModel', train_q={'var': ['V6'], '_id': not_imageset1}, test_q=dict(_id=imageset1),
                     labelfunc='category', split_by=None, npc_train=None, npc_test=len(imageset1), npc_validate=0,
                     num_splits=1, metric_screen="classifier", metric_kwargs={'model_type': 'svm.LinearSVC',
                                                                              'model_kwargs': {
                                                                              'GridSearchCV_params': {'C':
                                                                                                          [1e-5, 1e-4,
                                                                                                           1e-3, .25e-3,
                                                                                                           .5e-3,
                                                                                                           .75e-3,
                                                                                                           1e-2, .25e-2,
                                                                                                           .5e-2,
                                                                                                           .75e-2, 1e-1,
                                                                                                           1,
                                                                                                           10]},
                                                                              'GridSearchCV_kwargs': {'n_jobs': 1}}
            })
        return model

    if decoder_model_name == 'StandardModelWithMargins':
        model = get_decoder_model_by_name('StandardModel')
        model['metric_kwargs']['margins'] = True
        return model

    if decoder_model_name == 'LogisticRegressionModel':
        model = get_decoder_model_by_name('StandardModel')
        model['metric_kwargs']['model_type'] = 'linear_model.LogisticRegression'
        model['metric_kwargs']['probabilities'] = True
        return model

    if decoder_model_name == 'SVMModel':
        model = get_decoder_model_by_name('StandardModel')
        model['metric_kwargs']['model_type'] = 'svm.SVC'
        model['metric_kwargs']['model_kwargs']['kernel'] = 'linear'
        model['metric_kwargs']['model_kwargs']['probability'] = True
        model['metric_kwargs']['probabilities'] = True
        return model

    else:
        raise ValueError, 'Model not recognized'
