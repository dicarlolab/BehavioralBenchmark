__author__ = 'ardila'
import dldata.human_data.confusion_matrices as CM
import itertools
from joblib import Parallel, delayed

from all_metrics import cache_composite_individual_self_consistency_all_metrics

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
         ('hvm_figure_ground_2', 'dot_on', None)]
tasks.extend(
        [('stratified_8_ways', 'synset', {'eight_way_ind': i}) for i in range(39)])

# Evaluate composite self consistency for HvM basic categorization, all properties and metrics
task_sets = {'hvm_basic_categorization': [tasks[8]],
             'hvm_subordinate_tasks': tasks[:8],
             'hvm_all_categorization_tasks': tasks[:9],
             'hvm_figure_ground': [tasks[9]],
             'imagenet_8ways': tasks[10:]}

#task_set = task_sets[sys.argv[1]]

def benchmark(task_set, parallel=True):
    task_set = task_sets[task_set]
    trials = [CM.get_data(collection, task_category, condition) for collection, task_category, condition in task_set]

    def get_valid_properties(trials):
        ips = []
        rps = []
        for trial_array in trials:
            ips.append(set(trial_array.dtype.names) and set(image_properties))
            rps.append(set(trial_array.dtype.names) and set(response_properties))
        ip = set.intersection(*ips)
        rp = set.intersection(*rps)
        return ip, rp

    ips, rps = get_valid_properties(trials)
    if parallel:
        Parallel(verbose=700, n_jobs=10)\
            (delayed(cache_composite_individual_self_consistency_all_metrics)\
                 (trials, image_property, response_property)\
             for image_property, response_property in itertools.product(ips, rps))
    else:
        [cache_composite_individual_self_consistency_all_metrics(trials, image_property, response_property)
             for image_property, response_property in itertools.product(ips, rps)]

