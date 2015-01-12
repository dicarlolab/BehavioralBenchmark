__author__ = 'ardila'
import scipy

import numpy as np
import dldata.human_data.confusion_matrices as CM
from dldata.metrics.utils import get_rm_metric
import scipy.stats
from bson import ObjectId
import datetime
import collections



def rms_spearman_consistency(RMs1, RMs2, metric, metrickwargs):
    """
    Calculates the spearman consistency between two sets of metrics

    :param RMs: List of matrices, each of which is [n_image_properties, n_response_properties, n_subjects]
    :param metric: A response matrix metric which is registered in the method RM_metrics
    :param metrickwargs: keyword arguments for the metric
    """
    metric_func, kwargs = get_rm_metric(metric, metrickwargs)
    m1 = []
    m2 = []
    for M1, M2 in zip(RMs1, RMs2):
        m1.extend(metric_func(M1, **kwargs))
        m2.extend(metric_func(M2, **kwargs))
    return scipy.stats.spearmanr(np.ravel(m1), np.ravel(m2))[0]


def trial_split_half_RMs(trials, image_property, response_property, rng):
    if rng is None:
        rng = np.random.RandomState(0)
    inds = rng.permutation(trials.shape[0])
    half = inds.shape[0]/2
    inds1, inds2 = inds[:half], inds[half:]
    trials1 = trials[inds1]
    trials2 = trials[inds2]
    RM1 = CM.get_response_matrix(trials1, image_property, response_property, group_by_worker=True)
    RM2 = CM.get_response_matrix(trials2, image_property, response_property, group_by_worker=True)
    return RM1, RM2




# def evaluate_task():
#
#     ## run a cache aware compute_metric_base
#     pass


# Imagenet: 40 8-way tasks. Get a dprime for each one, or just mean dprime per 8 way.
#HvM Basic level: 1 8-way task. Dprime for each one.

