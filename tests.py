__author__ = 'ardila'
import dldata.human_data.confusion_matrices as CM
import all_metrics
import dldata.metrics.utils as u
import os
from benchmark import benchmark
from hvm_2way_consistency import standard_subordinate_dprime_IC
from get_model_results import get_nyu_basic_results, get_nyu_subordinate_results

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

def test_hvm_basic_2way_consistency():
    standard_subordinate_dprime_IC()

def test_nyu_model_results():
    get_nyu_basic_results()
    get_nyu_subordinate_results()

test_hvm_basic_2way_consistency()
