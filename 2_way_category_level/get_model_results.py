__author__ = 'ardila'
import dldata.stimulus_sets.hvm as hvm
from bson import ObjectId
import itertools
import numpy as np
import dldata.metrics.utils as u
import pymongo as pm
import tabular as tb
import gridfs
import cPickle
import copy

import utils as utils

dataset = hvm.HvMWithDiscfade()

DB = pm.MongoClient(port=22334)['ModelBehavior']


FEATURES = ['NYU_MODEL',
 'v1like_features',
 'v2like_features',
 'pht2_features_0',
 'pht2_features_2',
 'pht2_features_3',
 'PLOS09_L3_3281',
 'sift_features',
 'slf_features']


classifier_config = {
            'metric_screen': 'classifier',
            'metric_kwargs':
                {'model_type': 'svm.LinearSVC',
                 'model_kwargs': {
                    'GridSearchCV_params': {'C': np.logspace(-5, 4, 22)},
                    'GridSearchCV_kwargs': {'n_jobs': 22},
                    'class_weight': 'auto'
                    }
                },
            'num_splits': 50,
            'npc_validate': 0}


def store_two_way(F, two_way_type, train_config, collname, type_tag):
    """
    Store one two way result in a database
    :param F: Features to run classifier on
    :param two_way_type: Name of task
    :param train_config: N_test and train and how to select data to train and test on
    :param collname: Name to get fs where results are stored via get_results fs
    :param type_tag: Name of the set of tasks this belongs to ('subordinate', 'basic')
    """
    results_fs = get_results_fs(collname)
    query = {'two_way_type': two_way_type}
    count = results_fs._GridFS__files.find(query).count()
    if count >= 1:
        idval = results_fs._GridFS__files.find_one(query)['_id']
        print 'Found %s as %s in %s'%(two_way_type, idval, collname)
    else:
        eval_config = copy.deepcopy(classifier_config)
        eval_config.update(train_config)
        results = u.compute_metric_base(F, dataset.meta, eval_config,
                                        return_splits=True)
        info = utils.SONify(dict(two_way_type=two_way_type, type_tag=type_tag, eval_config=eval_config))
        blob = cPickle.dumps(results, protocol=cPickle.HIGHEST_PROTOCOL)
        count = results_fs._GridFS__files.find(query).count()
        if count <1:
            idval = results_fs.put(blob, **info)
        print 'Stored %s as %s in %s'%(two_way_type, idval, collname)


def store_subordinate_results(F, obj1, obj2, collname):
    """
    Store one subordinate two way from HvM
    :param F: Features used
    :param obj1: Object 1
    :param obj2: Object 2
    :param collname: Collection to store in
    """
    two_way_type = '_'.join(sorted([obj1, obj2]))
    train_config = {
        'labelfunc': 'obj',
        'npc_test': 5,
        'npc_train': 85,
        'split_by': 'obj',
        'test_q': {'obj': [obj1, obj2], 'var': 'V6'},
        'train_q': {'obj': [obj1, obj2]}}
    store_two_way(F, two_way_type, train_config, collname, 'subordinate')


def store_basic_results(F, cat1, cat2, collname):
    """
    Store one basic two way from HvM
    :param F: Features used
    :param obj1: Object 1
    :param obj2: Object 2
    :param collname: Collection to store in
    """
    two_way_type = '_'.join(sorted([cat1, cat2]))
    train_config = {
            'labelfunc': 'category',
            'npc_test': 19,
            'npc_train': 611,
            'split_by': 'category',
            'test_q': {'category': [cat1, cat2], 'var': 'V6'},
            'train_q': {'category': [cat1, cat2]}}
    store_two_way(F, two_way_type, train_config, collname, 'basic')



def get_hvm_attached_feature_results_subordinate(feature_name):
    """
    Given a set of features, store all 2-way subordinate results in the database
    :param feature_name: Name of features to calculate for
    """
    F = get_features(feature_name)
    collname = feature_name
    m = dataset.meta
    for cat in np.unique(m['category']):
        for obj1, obj2 in itertools.combinations(np.unique(m[m['category']==cat]['obj']), 2):
            store_subordinate_results(F, obj1, obj2, collname)


def get_hvm_attached_feature_results_basic(feature_name):
    """
    Given a set of features, store all 2-way basic results in the database
    :param feature_name: Name of features to calculate for
    """
    F = get_features(feature_name)
    collname = feature_name
    for cat1, cat2 in itertools.combinations(np.unique(dataset.meta['category']), 2):
        store_basic_results(F, cat1, cat2, collname)



def get_features(feature_name):
    """
    Interface to get features. Will eventually be unified on the dataset itself
    :param feature_name:
    :return:
    """
    if feature_name == 'NYU_MODEL':
        F = dataset.get_features(
            dict(crop=None, dtype=u'float32', mask=None, mode=u'RGB', normalize=False, resize_to=[256, 256]),
                               ObjectId('542927872c39ac23120db840'),
                               u'fc6')[:]
    else:
        F = dataset.machine_features(feature_name)
    return F


def deduplicate(coll):
    #   WILL NOT WORK FOR NEW FS FORMAT HAS TO BE REWRITTEN
    for t_type in coll.distinct('two_way_type'):
        if coll.find({'two_way_type': t_type}).count() > 1:
            _id = coll.find_one({'two_way_type': t_type})['_id']
            coll.remove({'_id': _id})


def get_model_behavior(feature_name, type_tag):
    fs = get_results_fs(feature_name)
    return get_trials(fs, type_tag)


def get_trials(fs, type_tag):
    recs = [rec for rec in fs._GridFS__files.find({'type_tag': type_tag})]
    trials = []
    for rec in recs:
        results_dic = cPickle.loads(fs.get_last_version(_id=rec['_id']).read())
        trials.append(trials_from_results_dic(results_dic, rec['two_way_type']))
    return tb.tab_rowstack(trials)


def trials_from_results_dic(results_dic, two_way_type, label_field):
    trials = []
    for i, split in enumerate(results_dic['splits'][0]):
        split_results = results_dic['split_results'][i]
        correct = np.array(split_results['test_errors'][0])==0
        Response = np.array(split_results['test_prediction'])
        meta = dataset.meta[split['test']]
        t_type = np.array([two_way_type]*meta.shape[0])
        worker_ids = np.array([i]*meta.shape[0])  # Modeling subjects as splits
        meta = meta.addcols([correct, Response, t_type, worker_ids],
                      names=['correct', 'Response', 'two_way_type', 'WorkerId'])
        trials.append(meta)
    return tb.tab_rowstack(trials)


#NYU_COLL = pm.MongoClient(port=22334)['BehavioralBenchmark']['NYU_Model_Results']


def get_results_fs(feature_name):
    return gridfs.GridFS(DB, feature_name+'_results')




