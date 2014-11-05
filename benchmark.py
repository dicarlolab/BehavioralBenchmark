__author__ = 'ardila'
from all_metrics import cache_composite_individual_self_consistency_all_metrics
import dldata.human_data.confusion_matrices as CM
import itertools
from joblib import Parallel, delayed
import sys
image_properties = ['task_category', 'obj', 'filename']
response_properties = ['Response', 'correct']

tasks = [(u'hvm_subordinate_identification_Chairs', 'obj', None),
         (u'hvm_subordinate_identification_Animals', 'obj', None),
         (u'hvm_subordinate_identification_Boats', 'obj', None),
         (u'hvm_subordinate_identification_Cars', 'obj', None),
         (u'hvm_subordinate_identification_Faces', 'obj', None),
         (u'hvm_subordinate_identification_Fruits', 'obj', None),
         (u'hvm_subordinate_identification_Planes', 'obj', None),
         (u'hvm_subordinate_identification_Tables', 'obj', None),
         ('hvm_basic_categorization_new', 'category', None),
         ('hvm_figure_ground_2', 'dot_on', None)].extend(
        [('stratified_8_ways', 'synset', {'eight_way_ind': i}) for i in range(39)])


# Evaluate composite self consistency for HvM basic categorization, all properties and metrics
task_sets= {'hvm_basic_categorization': [tasks[9]]}

task_set = task_sets[sys.argv[1]]

trials = [CM.get_data(collection, task_category, condition) for collection, task_category, condition in task_set]

def get_valid_properties(trials):
    ip  = set(trials.dtype.names) and set(image_properties)
    rp = set(trials.dtype.names) and set(response_properties)
    return ip, rp

ips, rps = get_valid_properties(trials)
Parallel(verbose=700, n_jobs=10)\
    (delayed(cache_composite_individual_self_consistency_all_metrics)\
         (trials, image_property, response_property)\
     for image_property, response_property in itertools.product(image_properties, response_properties))

