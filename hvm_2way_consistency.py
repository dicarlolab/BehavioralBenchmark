__author__ = 'ardila'
import dldata.human_data.confusion_matrices as cm
import numpy as np
import dldata.metrics.utils as u
import scipy.stats


def trial_split_half_consistency(trials, metric, kwargs, split_field,
                                 image_property, response_property, bstrapiter = 900, rng = None,
                                 spearman_brown_correction=True):
    metric_func, kwargs = u.get_rm_metric(metric, kwargs)
    if rng is None:
        rng = np.random.RandomState(0)
    ICs = []
    for rep in range(bstrapiter):
        inds = rng.permutation(range(trials.shape[0]))
        inds1, inds2 = inds[:inds.shape[0]/2], inds[inds.shape[0]/2:]
        CMS1 = get_rms(trials[inds1], split_field, image_property, response_property)
        CMS2 = get_rms(trials[inds2], split_field, image_property, response_property)
        m1 = []
        m2 = []
        for CM1, CM2 in zip(CMS1, CMS2):
            m1.extend( metric_func(CM1, **kwargs))
            m2.extend( metric_func(CM2, **kwargs))
        IC, _ = scipy.stats.spearmanr(m1,m2)
        if spearman_brown_correction:
            IC = spearman_brown_correct(IC)
        ICs.append(IC)
    return np.mean(ICs), np.std(ICs)


def spearman_brown_correct(IC):
    return 2*IC/(1+IC)


def get_rms(data, split_field, image_property, response_property, split_field_vals=None):
    RMs = []
    if split_field is None:
        RMs.append(cm.get_response_matrix(data, image_property, response_property, group_by_worker=True))
    else:
        if split_field_vals is None:
            split_field_vals = np.unique(data[split_field])
        for fval in split_field_vals:
            rel_data = data[data[split_field] == fval]
            RMs.append(cm.get_confusion_matrix(rel_data, image_property, response_property, group_by_worker=True)[0])
    return RMs


def get_basic_human_data():
    meta_field = 'category'
    data = cm.get_data('hvm_basic_2ways',
                       meta_field, trial_data=['ImgData'])
    data = data[data['trialNum']>10]
    two_way_types = []
    for d in data:
        choice1 = d['ImgData']['Test'][0][meta_field]
        choice2 = d['ImgData']['Test'][1][meta_field]
        two_way_types.append('_'.join(sorted([choice1, choice2])))
    data = data.addcols(two_way_types, 'two_way_type')
    return data


def get_subordinate_human_data():
    meta_field = 'obj'
    data = cm.get_data('hvm_subordinate_2ways',
                       meta_field, trial_data=['ImgData'])
    data = data[data['trialNum']>10]
    two_way_types = []
    for d in data:
        choice1 = d['ImgData']['Test'][0][meta_field]
        choice2 = d['ImgData']['Test'][1][meta_field]
        two_way_types.append('_'.join(sorted([choice1, choice2])))
    data = data.addcols(two_way_types, 'two_way_type')
    return data
# #
# # def standard_subordinate_dprime_IC():
#     human_data = get_subordinate_data()
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
        m1.extend( metric_func(CM1, **kwargs))
        m2.extend( metric_func(CM2, **kwargs))
    R, _ = scipy.stats.spearmanr(m1,m2)
    denominators =[]
    rng = np.random.RandomState(0)
    for rep in range(bstrapiter):
        IC1, _ = trial_split_half_consistency(data1, metric, kwargs, split_field,
                                              image_property, response_property, bstrapiter=1, rng=rng)
        IC2, _ = trial_split_half_consistency(data2, metric, kwargs, split_field,
                                              image_property, response_property, bstrapiter=1, rng=rng)
        denominators.append(np.sqrt(IC1*IC2))
    consistencies = R/np.array(denominators)
    return np.mean(consistencies), np.std(consistencies)




