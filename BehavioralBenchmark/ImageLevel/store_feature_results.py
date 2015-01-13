__author__ = 'ardila'

from dldata.metrics.utils import compute_metric_base
import cPickle
import datetime
import numpy as np
import collections
from bson import ObjectId

def SONify(arg, memo=None):
    """
    A utility to convert python objects into a format ok for storing in the database
    :param arg: Document being stored
    :param memo: Additional info to store
    :return: :raise TypeError: SONified document
    """
    if memo is None:
        memo = {}
    if id(arg) in memo:
        rval = memo[id(arg)]
    if isinstance(arg, ObjectId):
        rval = arg
    elif isinstance(arg, datetime.datetime):
        rval = arg
    elif isinstance(arg, np.float):
        rval = float(arg)
    elif isinstance(arg, np.int):
        rval = int(arg)
    elif isinstance(arg, (list, tuple)):
        rval = type(arg)([SONify(ai, memo) for ai in arg])
    elif isinstance(arg, collections.OrderedDict):
        rval = collections.OrderedDict([(SONify(k, memo), SONify(v, memo))
            for k, v in arg.items()])
    elif isinstance(arg, dict):
        rval = dict([(SONify(k, memo), SONify(v, memo))
            for k, v in arg.items()])
    elif isinstance(arg, (basestring, float, int, type(None))):
        rval = arg
    elif isinstance(arg, np.ndarray):
        if arg.ndim == 0:
            rval = SONify(arg.sum())
        else:
            rval = map(SONify, arg) # N.B. memo None
    # -- put this after ndarray because ndarray not hashable
    elif arg in (True, False):
        rval = int(arg)
    else:
        raise TypeError('SONify', arg)
    memo[id(rval)] = rval
    return rval

def store_compute_metric_results(F, meta, eval_config, fs, additional_info):
    """
    Used to store results of a decoder model (as specified by eval_config) on top of features F
    :param F: Features to compute a classifier on
    :param meta: meta information about the images
    :param eval_config: eval config (see compute metric base)
    :param fs: grid fs filesystem for storage
    :param additional_info: Additional info about this classifier experiment
    :return: tuple of results from compute_metric_base and the id of the record stored
    """
    results = compute_metric_base(F, meta, eval_config)
    additional_info['eval_config'] = eval_config
    blob = cPickle.dumps(results, protocol=cPickle.HIGHEST_PROTOCOL)
    idval = fs.put(blob, **SONify(additional_info))
    return results, idval


def store_subsampled_feature_results(F, meta, eval_config, fs, feature_inds, additional_info):
    """
    Used to store results of a decoder model (as specified by eval_config) on top of features F,
    subsampled by feature_inds
    :param F: Features to compute a classifier on
    :param meta: meta information about the images
    :param eval_config: eval config (see compute metric base)
    :param fs: grid fs filesystem for storage
    :param feature_inds: indices of the features to use
    :param additional_info: Additional info about this classifier experiment
    :return: tuple of results from compute_metric_base and the id of the record stored
    """
    F = F[:, feature_inds]
    store_compute_metric_results(F, meta, eval_config, fs, additional_info)





