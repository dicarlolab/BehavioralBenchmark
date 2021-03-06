__author__ = 'ardila'

from dldata.metrics.utils import compute_metric_base
import cPickle
import datetime
import numpy as np
import collections
import gridfs
import pymongo as pm
from bson import ObjectId
import copy
import decoder_models

DB = pm.MongoClient(port=22334)['ModelBehavior']

def SONify(arg, memo=None):
    """
    A utility to convert python objects into a format ok for storing in the database
    :param arg: Document being stored
    :param memo: Additional info to store
    :return: SONified document
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

def perminverse(s):
    '''
    Fast inverse of a (numpy) permutation.
    **Paramters**
            **s** :  sequence
                    Sequence of indices giving a permutation.
    **Returns**
            **inv** :  numpy array
                    Sequence of indices giving the inverse of permutation `s`.
    '''
    X = np.array(range(len(s)))
    X[s] = range(len(s))
    return X


def reorder_to(A, B):
    C = A.copy()
    C.sort()
    s = np.searchsorted(C, B)
    t = np.searchsorted(C, A)
    return perminverse(t)[s]


def get_name(decoder_model_name, feature_name):
    return '_'.join([decoder_model_name, feature_name, 'results'])


def get_file_collection(decoder_model_name, feature_name):
    collname = get_name(decoder_model_name, feature_name)+'.files'
    return DB[collname]

def get_gridfs(decoder_model_name, feature_name):
    name = get_name(decoder_model_name, feature_name)
    return gridfs.GridFS(DB, name)

def get_metric_ready_result(results, meta, desired_order=decoder_models.ImageSet1_inds):

    test_split = np.array(results['splits'][0][0]['test'])
    new_order = reorder_to(test_split, desired_order)
    probs = results['split_results'][0].get('test_probabilities')
    if probs is None:
        correct = np.squeeze(np.array(results['split_results'][0]['test_errors']))
        correct = correct[new_order]
        return correct
    else:
        labels = meta['category'][results['splits'][0][0]['test']]
        label_to_Y = {cat: i for i, cat in enumerate(results['split_results'][0]['labelset'])}
        Y = np.array([label_to_Y[label] for label in labels])
        Y_canonical = Y[new_order]
        probs_canonical = probs[new_order]
        correct_class_prob = np.array([probs[y] for y, probs in zip(Y_canonical, probs_canonical)])
        return correct_class_prob


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
    results = compute_metric_base(F, meta, eval_config, return_splits=True, attach_models=True)
    additional_info['eval_config'] = SONify(copy.deepcopy(eval_config))
    blob = cPickle.dumps(results, protocol=cPickle.HIGHEST_PROTOCOL)
    M = get_metric_ready_result(results, meta)
    additional_info['metric_ready_result'] = M
    additional_info_SON = SONify(additional_info)


    idval = fs.put(blob, **additional_info_SON)
    print 'Stored results in %s'%idval
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
    F = np.copy(F[:, feature_inds])
    return store_compute_metric_results(F, meta, eval_config, fs, additional_info)





