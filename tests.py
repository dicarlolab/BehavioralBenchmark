__author__ = 'ardila'
import dldata.human_data.confusion_matrices as CM
import all_metrics
import dldata.metrics.utils as u
import os
from benchmark import benchmark
import hvm_2way_consistency as h
import get_model_results as g
def test_all_metrics():
    trials = CM.get_data('hvm_basic_categorization_new', 'category')

    results = all_metrics.composite_individual_self_consistency_all_metrics([trials], image_property = 'task_category',     response_property='Response')
    mr = results['metrics_results']
    print [(metric, mr[metric]['self_consistency'], mr[metric]['self_consistency_error']) for metric in mr.keys()]

def test_off_diagonal():
    RM, _, _ = CM.get_response_matrix(CM.get_data('hvm_basic_categorization_new', 'category'),
                                     'task_category', 'Response')
    print u.symmetrize_confusion_matrix(RM, take='off_diagonal')
    #print u.metrics_from_confusion_mat([RM], metric='off_diagonal')

def test_benchmark():
    benchmark('hvm_basic_categorization', parallel=True)
    benchmark('hvm_subordinate_tasks', parallel=True)
    benchmark('hvm_all_categorization_tasks', parallel=True)
    benchmark('hvm_figure_ground', parallel=True)

def test_nyu_model_results():
    trials = g.basic_trials(g.NYU_COLL)
    names = ('obj', 'rxz', 'rxy', 'ryz', 'ty', 'tz', 's', 'bg_id', 'size', 'var', '_id', 'filename', 'id', 'category',
           'rxz_semantic', 'rxy_semantic', 'ryz_semantic', 'correct', 'Response', 'two_way_type')
    assert trials.dtypes.names == names

def test_nyu_basic_consistency():
    human_data = h.get_basic_human_data()
    model_data = g.basic_trials(g.NYU_COLL)
    print h.trial_split_consistency(human_data, model_data,
                                    'dp_standard', 'two_way_type', 'category')

def test_nyu_subordinate_consistency():
    human_data = h.get_subordinate_human_data()
    model_data = g.subordinate_trials(g.NYU_COLL)
    print h.trial_split_consistency(human_data, model_data, 'dp_standard', 'two_way_type', 'obj', bstrapiter=3)

test_nyu_subordinate_consistency()
