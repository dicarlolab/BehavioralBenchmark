__author__ = 'ardila'
import pymongo as pm
db = pm.MongoClient(port=22334)['BehavioralBenchmarkResults']

task_sets = {'9c1362cca709800db514e1334e7e6f4ce2e04057': 'HvM Basic Categorization',
             '6bf1f29c098c98147466076bb6977357250aec8c': 'HvM Subordinate Tasks',
             '680368705caeb5758cb961b1c09845db8feca427': 'HvM All Tasks',
             'ed4c76f2ede3d96fc7dbb4e0e6b8ab06cf0f908e': 'HvM Figure Ground'}
def self_consistencies():
    coll = db['InternalConsistency']
    for task_set_hash, task_set_name in task_sets.items():
        print task_set_name
        print '__________________'
        print 'Consistency Type:' + 'composite_individual'
        print '__________________'
        for result in coll.find({'trials_hash': task_set_hash, 'consistency_type': 'composite_individual'}):
            print 'Image property: ' + result['image_property'], ', Response property: '+ result['response_property']
            for metric in result['metrics_results']:
                print metric+' ' + str(result['metrics_results'][metric]['self_consistency']) + ' error:'+ str(result['metrics_results'][metric]['self_consistency_error'])
            print '--------------------'

