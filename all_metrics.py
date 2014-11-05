import numpy as np
import copy
import dldata.human_data.confusion_matrices as CM
from dldata.metrics.utils import metrics_from_confusion_mat, dprime_bangmetric, symmetrize_confusion_matrix,performances_bangmetric, row_normalized_first_column
import os
from joblib import Parallel, delayed
import itertools
import cPickle
import scipy.stats
import pymongo as pm
import hashlib


small_cm_metrics = ['dp_standard',
                    'dp',
                    'acc',
                    'symmetrized_raw',
                    'diagonal'] # Requirements: square
large_cm_metrics = ['off_diagonal'] # Requirements: square, larger than 3*3
rm_metrics = ['row_normalized_first_column', 'raw'] # Requirements, none


#dataset
#task
#response_grain


IC_collection = pm.MongoClient(port=22334)['BehavioralBenchmarkResults']['InternalConsistency']



def cache_composite_individual_self_consistency_all_metrics(trials, image_property, response_property):

    """

    :param trials: List of trial tab arrays, one per task.
            Each entry has information about a bunch of properties per trial
    :param image_property: What property of the image presented in trials to build response matrices out of
    :param response_property: What property of the response chosen to build response matrices out of
    """
    #Check if results already exist in the database
    trials_hash = hashlib.sha1(trials).hexdigest
    query = {'trials_hash': trials_hash, 'image_property': image_property, 'response_property': response_property,
             'consistency_type': 'composite_individual'}
    num_entries = IC_collection.find(query).count()
    if num_entries == 0:
        results = composite_individual_self_consistency_all_metrics(trials, image_property, response_property)
        IC_collection.insert(results)
    if num_entries == 1:
        return IC_collection.find_one(query)
    else:
        raise ValueError, 'Cache error! check entries matching query %s' % query



    # Build response matrices

def composite_individual_self_consistency_all_metrics(trials, image_property, response_property):
    # Get response matrices
    RMs = []
    trials_hash = hashlib.sha1(str(trials)).hexdigest()
    if (image_property in trials.dtype.names) and (response_property in trials.dtype.names)
    for trial_array in trials:
        response_matrix, _, _ = CM.get_response_matrix(trial_array, image_property,
                                                 response_property, condition=None,
                                                 group_by_worker=True)
        RMs.append(response_matrix)

    # What metrics can we apply to these response matrices?

    if image_property == 'task_category' and  response_property == 'Response':
        metrics = small_cm_metrics+rm_metrics
        if min([response_matrix.shape[0] for response_matrix in RMs]) > 2:
            metrics += large_cm_metrics
    else:
        metrics = rm_metrics

    metrics_results = {}

    for metric in metrics:
        #filename = image_property+response_property+metric
        #if os.path.exists(filename):
        #    return cPickle.load(open(filename, 'rb'))
        #else:
        m, sc, me, sce = metrics_from_confusion_mat(RMs, metric)
        metrics_results[metric] = {'metric_values': m, 'self_consistency': sc,
                                   'metric_error': me, 'self_consistency_error': sce}

    return {'trials_hash': trials_hash, 'response_property': response_property, 'image_property': image_property,
            'metrics_results': metrics_results, 'consistency_type': 'composite_individual'}



def cache_pool_self_consistency(trials, image_property, response_property):
    # TODO use trial split bootstraps to get internal consistency
    pass






