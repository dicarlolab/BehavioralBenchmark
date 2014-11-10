__author__ = 'ardila'
import dldata.human_data.confusion_matrices as cm
import numpy as np
import dldata.metrics.utils as u
import scipy.stats

meta_field = 'category'
data = cm.get_data('hvm_basic_2ways', meta_field, trial_data=['ImgData'])

data = data[data['trialNum']>10]

two_way_types = []
for d in data:
    choice1 = d['ImgData']['Test'][0][meta_field]
    choice2 = d['ImgData']['Test'][1][meta_field]
    two_way_types.append('_'.join(sorted([choice1, choice2])))

data = data.addcols(two_way_types, 'two_way_type')


def trial_split_half_consistency(trials, metric, kwargs, split_field, meta_field, bstrapiter = 900, rng = None, spearman_brown_correction=True):
    metric_func, kwargs = u.get_rm_metric(metric, kwargs)
    if rng is None:
        rng = np.random.RandomState(0)
    ICs = []
    for rep in range(bstrapiter):
        if rep % 100 == 0:
            print rep
            print np.mean(ICs)
        inds = rng.permutation(range(trials.shape[0]))
        inds1, inds2 = inds[:inds.shape[0]/2], inds[inds.shape[0]/2:]
        CMS1 = get_cms(trials[inds1], split_field, meta_field)
        CMS2 = get_cms(trials[inds2], split_field, meta_field)
        m1 = []
        m2 = []
        for CM1, CM2 in zip(CMS1, CMS2):
            m1.extend( metric_func(CM1, **kwargs))
            m2.extend( metric_func(CM2, **kwargs))
        IC, _ = scipy.stats.spearmanr(m1,m2)
        ICs.append(spearman_brown_correction(IC))
    return np.mean(ICs), np.std(ICs)

def spearman_brown_correction(IC):
    return 2*IC/(1+IC)

def get_cms(data, split_field, meta_field):
    CMs = []
    for fval in np.unique(data[split_field]):
        rel_data = data[data[split_field] == fval]
        CMs.append(cm.get_confusion_matrix(rel_data, meta_field)[0])
    return CMs

def standard_dprime_IC():
    print trial_split_half_consistency(data, 'dp_standard', kwargs={},
                                   split_field='two_way_type',
                                   meta_field='category')
