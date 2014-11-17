__author__ = 'ardila'
import dldata.stimulus_sets.hvm as hvm
from bson import ObjectId
import itertools
import numpy as np
import dldata.metrics.utils as u
import pymongo as pm
import utils as utils
import hvm_2way_consistency as h
import tabular as tb
dataset = hvm.HvMWithDiscfade()

DB = pm.MongoClient(port=22334)['BehavioralBenchmark']


def store_subordinate_results(F, obj1, obj2, collname):
    coll = DB[collname]
    two_way_type = '_'.join(sorted([obj1, obj2]))
    print two_way_type
    query = {'two_way_type': two_way_type}
    if coll.find(query).count() == 1:
        pass
    elif coll.find(query).count() == 0:
        eval_config = {
            'labelfunc': 'obj',
            'metric_screen': 'classifier',
            'metric_kwargs':
                {'model_type': 'svm.LinearSVC', 'model_kwargs': {
                    # 'C': 5e4, 'penalty': 'l2','loss': 'l2',
                    'GridSearchCV_params': {'C': np.logspace(-5, 4, 22)},
                    'GridSearchCV_kwargs': {'n_jobs': 22},
                    'class_weight': 'auto'
                }},
            'npc_test': 5,
            'npc_train': 85,
            'npc_validate': 0,
            'num_splits': 50,
            'split_by': 'obj',
            'test_q': {'obj': [obj1, obj2], 'var': 'V6'},
            'train_q': {'obj': [obj1, obj2]}}
        results = u.compute_metric_base(F, dataset.meta, eval_config,
                                        return_splits=True)
        doc = utils.SONify({'two_way_type': two_way_type,
                            'results': results,
                            'type_tag': 'subordinate'})
        coll.insert(doc)
    else:
        print 'More than one precomputed result found for %s' % two_way_type
        raise ValueError


def store_basic_results(F, cat1, cat2, collname):
    coll = DB[collname]
    two_way_type = '_'.join(sorted([cat1, cat2]))
    query = {'two_way_type': two_way_type}
    print two_way_type
    if coll.find(query).count() == 1:
        return coll.find_one(query)['results']
    elif coll.find(query).count() == 0:
        eval_config = {
            'labelfunc': 'category',
            'metric_screen': 'classifier',
            'metric_kwargs':
                {'model_type': 'svm.LinearSVC', 'model_kwargs': {
                    # 'C': 5e4, 'penalty': 'l2','loss': 'l2',
                    'GridSearchCV_params': {'C': np.logspace(-5, 4, 22)},
                    'GridSearchCV_kwargs': {'n_jobs': 22},
                    'class_weight': 'auto'
                }},
            'npc_test': 19,
            'npc_train': 611,
            'npc_validate': 0,
            'num_splits': 50,
            'split_by': 'category',
            'test_q': {'category': [cat1, cat2], 'var': 'V6'},
            'train_q': {'category': [cat1, cat2]}}
        results = u.compute_metric_base(F, dataset.meta, eval_config,
                                        return_splits=True)
        doc = utils.SONify({'two_way_type': two_way_type,
                            'results': results,
                            'type_tag': 'basic'})
        coll.insert(doc)
    else:
        print 'More than one precomputed result found for %s' % two_way_type
        raise ValueError


def get_nyu_basic_results():
    F = dataset.get_features({u'crop': None,
                              u'dtype': u'float32',
                              u'mask': None,
                              u'mode': u'RGB',
                              u'normalize': False,
                              u'resize_to': [256, 256]},
                             ObjectId('542927872c39ac23120db840'),
                             u'fc6')[:]
    collname = 'NYU_Model_Results'
    for cat1, cat2 in itertools.combinations(np.unique(dataset.meta['category']), 2):
        store_basic_results(F, cat1, cat2, collname)


def get_nyu_subordinate_results():
    F = dataset.get_features({u'crop': None,
                              u'dtype': u'float32',
                              u'mask': None,
                              u'mode': u'RGB',
                              u'normalize': False,
                              u'resize_to': [256, 256]},
                             ObjectId('542927872c39ac23120db840'),
                             u'fc6')[:]
    collname = 'NYU_Model_Results'
    m = dataset.meta
    for cat in np.unique(m['category']):
        for obj1, obj2 in itertools.combinations(np.unique(m[m['category']==cat]['obj']), 2):
            store_subordinate_results(F, obj1, obj2, collname)


def deduplicate(coll):
    for t_type in coll.distinct('two_way_type'):
        if coll.find({'two_way_type': t_type}).count() > 1:
            _id = coll.find_one({'two_way_type': t_type})['_id']
            coll.remove({'_id': _id})


def add_type_tag(coll):
    basic_two_way_types = np.unique(h.get_basic_human_data()['two_way_type'])
    sub_two_way_types = np.unique(h.get_subordinate_human_data()['two_way_type'])
    for entry in coll.find():
        type_tag = 'other'
        if entry['two_way_type'] in basic_two_way_types:
            type_tag = 'basic'
        elif entry['two_way_type'] in sub_two_way_types:
            type_tag = 'subordinate'
        coll.update({'_id':entry['_id']}, {'$set': {'type_tag': type_tag}})


def subordinate_trials(coll):
    data = coll.find({'type_tag': 'subordinate'})
    return get_model_trials(data)


NYU_COLL = pm.MongoClient(port=22334)['BehavioralBenchmark']['NYU_Model_Results']

def basic_trials(coll):
    data = coll.find({'type_tag': 'basic'})
    return get_model_trials(data)

def get_model_trials(data):
    trials = []
    for entry in data:
        for i, split in enumerate(entry['results']['splits'][0]):
            split_results = entry['results']['split_results'][i]
            correct = np.array(split_results['test_errors'][0])==0
            Response = split_results['test_prediction']
            meta = dataset.meta[split['test']]
            two_way_type = [entry['two_way_type']]*meta.shape[0]
            worker_ids = [i]*meta.shape[0]  # Modeling subjects as splits
            trials.append(meta.addcols([correct, Response, two_way_type, worker_ids],
                                names=['correct', 'Response', 'two_way_type', 'WorkerId']))
    return tb.tab_rowstack(trials)

# for two_way_type in
# results = coll.find_one({'two_way_type': two_way_type})
#for r in results['split_results']:

