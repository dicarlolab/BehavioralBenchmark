__author__ = 'ardila'
import dldata.human_data.confusion_matrices as cm
import numpy as np
import dldata.metrics.utils as u
import scipy.stats
import pymongo as pm
import get_model_results as g
import utils
import tabular as tb

benchmark_db = pm.MongoClient(port=22334)['BehavioralBenchmark']


def trial_split_half_consistency(trials, metric, kwargs, split_field,
                                 image_property, response_property, bstrapiter=900, rng=None,
                                 spearman_brown_correction=True):
    """

    :param trials: Data in trial tabular array format as returned by dldata.confusion_matrices.get_data
    :param metric: Oneof the metrics registered in dldata.metrics.utils get_rm_metrics()
    :param kwargs: Kwargs to pass that metric
    :param split_field: Generate one response matrix per unique value of this field in trials
    :param image_property: What property of the image to use for response matrix
    :param response_property: What property of the response to use for the response matrix
    :param bstrapiter: NUmber of iterations to repeat bootstrap
    :param rng: random number generator, as in np.random.RandomState
    :param spearman_brown_correction: Whether to correct IC using spearman-brown prediction formula
    :return: mean, standard deviation over bootstrap
    """
    metric_func, kwargs = u.get_rm_metric(metric, kwargs)
    if rng is None:
        rng = np.random.RandomState(0)
    ICs = []
    for rep in range(bstrapiter):
        if rep % 100 == 0:
            print rep / float(bstrapiter)
        inds = rng.permutation(range(trials.shape[0]))
        inds1, inds2 = inds[:inds.shape[0] / 2], inds[inds.shape[0] / 2:]
        CMS1 = get_rms(trials[inds1], split_field, image_property, response_property)
        CMS2 = get_rms(trials[inds2], split_field, image_property, response_property)
        m1 = []
        m2 = []
        for CM1, CM2 in zip(CMS1, CMS2):
            m1.extend(metric_func(CM1, **kwargs))
            m2.extend(metric_func(CM2, **kwargs))
        IC, _ = scipy.stats.spearmanr(m1, m2)
        if spearman_brown_correction:
            IC = spearman_brown_correct(IC)
        ICs.append(IC)
    return np.mean(ICs), np.std(ICs)


def spearman_brown_correct(IC):
    return 2 * IC / (1 + IC)


def get_rms(data, split_field, image_property, response_property, split_field_vals=None):
    RMs = []
    if split_field is None:
        RMs.append(cm.get_response_matrix(data, image_property, response_property, group_by_worker=True))
    else:
        if split_field_vals is None:
            split_field_vals = np.unique(data[split_field])
        for fval in split_field_vals:
            rel_data = data[data[split_field] == fval]
            RMs.append(cm.get_response_matrix(rel_data, image_property=image_property,
                                              response_property=response_property, group_by_worker=True)[0])
    return RMs


def get_basic_human_data():
    meta_field = 'category'
    data = cm.get_data('hvm_basic_2ways',
                       meta_field, trial_data=['ImgData'])
    data = clean_and_two_way_type(data, meta_field)
    return clean_and_two_way_type(data, meta_field)


def get_subordinate_human_data():
    meta_field = 'obj'
    data = cm.get_data('hvm_subordinate_2ways',
                       meta_field, trial_data=['ImgData'])
    return clean_and_two_way_type(data, meta_field)


def clean_and_two_way_type(data, meta_field):
    data = data[data['trialNum'] > 10]
    two_way_types = []
    for d in data:
        choice1 = d['ImgData']['Test'][0][meta_field]
        choice2 = d['ImgData']['Test'][1][meta_field]
        two_way_types.append('_'.join(sorted([choice1, choice2])))
    data = data.addcols(two_way_types, 'two_way_type')
    return data


# #
# # def standard_subordinate_dprime_IC():
# human_data = get_subordinate_data()
#     model_data = get_suborddinate_data()
#     print trial_split_half_consistency(data, 'dp_standard', kwargs={},
#                                        split_field='two_way_type',
#                                        meta_field='obj')
#
#
#
# def dprime_consistency():
#     model_data = get_trials()
#     trial_split_half_consistency(data, 'dp_standard', kwargs={},
#                                        split_field='two_way_type',
#                                        meta_field='category')
#

def apply_metric(RMs, metric_func, kwargs):
    m = []
    for RM in RMs:
        m.extend(metric_func(RM, **kwargs))
    return m


def trial_split_consistency(data1, data2, metric, split_field,
                            image_property, response_property, kwargs=None, bstrapiter=900):
    metric_func, kwargs = u.get_rm_metric(metric, kwargs)
    split_field_vals = None
    if split_field is not None:
        split_field_vals = np.unique(data1[split_field])
    CMS1 = get_rms(data1, split_field, image_property, response_property, split_field_vals)
    CMS2 = get_rms(data2, split_field, image_property, response_property, split_field_vals)
    m1 = []
    m2 = []
    for CM1, CM2 in zip(CMS1, CMS2):
        m1.extend(metric_func(CM1, **kwargs))
        m2.extend(metric_func(CM2, **kwargs))
    R, _ = scipy.stats.spearmanr(m1, m2)
    consistencies = []
    rng = np.random.RandomState(0)
    ICs1 = []
    ICs2 = []
    for rep in range(bstrapiter):
        if rep % 100 == 0:
            print rep / float(bstrapiter)

        # Split first data
        inds1 = rng.permutation(range(data1.shape[0]))
        inds11, inds12 = inds1[:inds1.shape[0] / 2], inds1[inds1.shape[0] / 2:]
        RMs11 = get_rms(data1[inds11], split_field, image_property, response_property)
        RMs12 = get_rms(data1[inds12], split_field, image_property, response_property)

        #Split second data
        inds2 = rng.permutation(range(data1.shape[0]))
        inds21, inds22 = inds2[:inds2.shape[0] / 2], inds2[inds2.shape[0] / 2:]
        RMs21 = get_rms(data1[inds21], split_field, image_property, response_property)
        RMs22 = get_rms(data1[inds22], split_field, image_property, response_property)

        #Calculate metrics
        m11, m12, m21, m22 = tuple([apply_metric(RMs, metric_func, kwargs) for RMs in [RMs11, RMs12, RMs21, RMs22]])

        #noise level
        IC1, _ = scipy.stats.spearmanr(m11, m12)
        IC2, _ = scipy.stats.spearmanr(m21, m22)
        noise = np.sqrt(IC1 * IC2)

        #Consistencies
        R1, _ = scipy.stats.spearmanr(m11, m21)
        R2, _ = scipy.stats.spearmanr(m11, m22)
        R3, _ = scipy.stats.spearmanr(m12, m21)
        R4, _ = scipy.stats.spearmanr(m11, m22)
        ICs1.append(IC1)
        ICs2.append(IC2)
        consistencies.extend([R1 / noise, R2 / noise, R3 / noise, R4 / noise])
    return np.mean(consistencies), np.std(consistencies), np.mean(ICs1), np.std(ICs2), np.mean(ICs1), np.mean(ICs2)


def store_consistency(behavior_name, consistency_type):
    results_coll = benchmark_db[behavior_name]
    results_key_name = '_'.join(['hvm', consistency_type, 'two_way'])
    if consistency_type == 'subordinate':
        human_data = get_subordinate_human_data()
        model_data = g.get_model_behavior(behavior_name, consistency_type)
    elif consistency_type == 'basic':
        human_data = get_basic_human_data()
        model_data = g.get_model_behavior(behavior_name, consistency_type)
    elif consistency_type == 'all':
        human_data = tb.tab_rowstack([get_basic_human_data(),
                                      get_subordinate_human_data()])
        model_data = tb.tab_rowstack([g.get_trials(behavior_name, 'basic'),
                                      g.get_trials(behavior_name, 'subordinate')])
    else:
        print "%s Not recognized as a consistency type" % consistency_type
        raise ValueError

    consistency_kwargs = {'metric': 'dp_standard', 'kwargs': None, 'split_field': 'two_way_type',
                          'image_property': 'obj', 'response_property': 'Response', 'bstrapiter': 3}
    results = {'consistency_kwargs': consistency_kwargs}
    results[results_key_name] = trial_split_consistency(human_data, model_data, **consistency_kwargs)
    results_coll.insert(utils.SONify(results))





