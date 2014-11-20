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
import gridfs
import cPickle
dataset = hvm.HvMWithDiscfade()

DB = pm.MongoClient(port=22334)['BehavioralBenchmark']

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
    results_fs = get_results_fs(collname)
    query = {'two_way_type': two_way_type}
    count = results_fs._GridFS__files.find(query).count()
    if count >= 1:
        pass
    eval_config = classifier_config.update(train_config)
    results = u.compute_metric_base(F, dataset.meta, eval_config,
                                        return_splits=True)
    info = {'two_way_type': two_way_type,
            'results': results,
            'type_tag': type_tag,
            'eval_config': eval_config}

    blob = cPickle.dumps(results, protocol=cPickle.HIGHEST_PROTOCOL)
    idval = results_fs.put(blob, **info)
    print 'Stored %s as %s in %s'%(two_way_type, idval, collname)

def store_subordinate_results(F, obj1, obj2, collname):
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
    two_way_type = '_'.join(sorted([cat1, cat2]))
    train_config =  {
            'labelfunc': 'category',
            'npc_test': 19,
            'npc_train': 611,
            'split_by': 'category',
            'test_q': {'category': [cat1, cat2], 'var': 'V6'},
            'train_q': {'category': [cat1, cat2]}}
    store_two_way(F, two_way_type, train_config, collname, 'basic')


def all_hvm_basic_2ways(F, collname):
    for cat1, cat2 in itertools.combinations(np.unique(dataset.meta['category']), 2):
        store_basic_results(F, cat1, cat2, collname)


def all_hvm_subordinate_2ways(F, collname):
    m = dataset.meta
    for cat in np.unique(m['category']):
        for obj1, obj2 in itertools.combinations(np.unique(m[m['category']==cat]['obj']), 2):
            store_subordinate_results(F, obj1, obj2, collname)


def get_hvm_attached_feature_results_basic(feature_name):
    F = dataset.machine_features(feature_name)
    collname = feature_name
    all_hvm_basic_2ways(F, collname)

def get_hvm_attached_feature_results_subordinate(feature_name):
    F = dataset.machine_features(feature_name)
    collname = feature_name
    all_hvm_subordinate_2ways(F, collname)


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


def get_trials(fsname, type_tag):
    fs = gridfs.GridFS(DB)
    recs = [rec for rec in fs._GridFS__files.find({'type_tag': type_tag})]
    trials = []
    for rec in recs:
        entry = cPickle.loads(fs.get_last_version(_id=rec['_id']).read())
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


NYU_COLL = pm.MongoClient(port=22334)['BehavioralBenchmark']['NYU_Model_Results']


def get_results_fs(feature_name):
    return gridfs.GridFS(DB, feature_name+'_results')




def get_model_trials(data):
    trials = []



