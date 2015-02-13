# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import scipy.stats as ss
import dldata.stimulus_sets.hvm as hvm
import pymongo as pm
import dldata.human_data.confusion_matrices as confusion_matrices
import numpy as np
from decoder_models import ImageSet1_inds
from copy import deepcopy



def get_human_data_densely_sampled():
    dataset = hvm.HvMWithDiscfade()
    raw_data = confusion_matrices.get_data('hvm_dense_smp_v6_2rpw', field='category')
    # Add rep number to raw data, then clean
    which_rep = {}
    for worker in np.unique(raw_data['WorkerId']):
        which_rep[worker] = {}
        for filename in np.unique(raw_data['filename']):
            which_rep[worker][filename] = 0
    rep = np.zeros(raw_data['filename'].shape[0])
    for i, trial in enumerate(raw_data):
        filename = trial['filename']
        worker = trial['WorkerId']
        rep[i] = which_rep[worker][filename]
        which_rep[worker][filename] += 1
    raw_data_with_rep = raw_data.addcols([rep], names=['rep'])

    # Get rid of everything but first two reps, get rid of learning reps (Images of V3 and V0)
    data = raw_data_with_rep[raw_data_with_rep['rep'] < 2]
    data = data[data['var'] == 'V6']

    # Reformat to matrix
    human_matrix = []  # images, reps, worker
    canonical_order = dataset.meta['_id'][ImageSet1_inds]
    workers = np.unique(data['WorkerId'])
    n_workers = len(workers)
    for worker in workers:
        worker_data = data[data['WorkerId'] == worker]
        rep0 = worker_data[worker_data['rep'] == 0]
        rep1 = worker_data[worker_data['rep'] == 1]
        c0 = []
        c1 = []
        for Imid in canonical_order:
            c0.append(rep0[rep0['_id'] == Imid]['correct'])
            c1.append(rep1[rep1['_id'] == Imid]['correct'])
        X = np.column_stack([np.array(c0), np.array(c1)])
        X = np.expand_dims(X, 2)
        human_matrix.append(X)
        assert set(np.unique(worker_data['_id'])) == set(canonical_order)
    human_matrix = np.concatenate(human_matrix, 2)
    assert human_matrix.shape == (128, 2, n_workers)
    human_individuals = deepcopy(human_matrix)
    human_reps = np.concatenate((human_matrix[:, 0, :], human_matrix[:, 1, :]), 1)
    return human_reps, human_individuals, raw_data_with_rep

#if len(M.shape) == 1:
#    assert len(M) == get_human_data_densely_sampled()[0].shape[0] == get_human_data_densely_sampled()[1].shape[0]

def empirical_consistency_to_humans(human_reps, M, number_of_iterations):

    # It computes the individual human to pool empirical consistency
    rng = np.random.RandomState(0)
    if len(M.shape) == 1:
        consistency_uncorrected = metric(M, human_reps.mean(1))
        consistencies = np.zeros(number_of_iterations)
        for nb in range(number_of_iterations):
            spH0, spH1 = split_half_reps(human_reps, rng)
            consistencies[nb] = consistency_uncorrected/metric(spH0, spH1)
        return consistencies
    else:
        consistencies = np.zeros(number_of_iterations)
        for nb in range(number_of_iterations):
            spH0, spH1 = split_half_reps(human_reps, rng)
            spM0, spM1 = split_half_reps(M, rng)

            HH = metric(spH0, spH1)
            MM = metric(spM0, spM1)

            H0M1 = metric(spH0, spM1)
            M0H0 = metric(spM0, spH0)
            H1M0 = metric(spH1, spM0)
            M1H1 = metric(spM1, spH1)
            consistencies[nb] = np.mean([H0M1, M0H0, H1M0, M1H1]) / np.sqrt(HH * MM)
    return consistencies

def normalized_human_to_pool_consistency(human_individuals):
    if len(human_individuals.shape) != 3:
        print "To compute Normalized_Human_to_Pool_consistency the data should be in format images*reps*workers"
        raise ValueError

    individual_to_pool_empirical = []
    noise_level = []
    mean_individual_to_meanpool = []
    for worker in range(human_individuals.shape[2]):
        # Get first rep non worker pool
        rep0_pool = np.delete(np.squeeze(np.copy(human_individuals[:, 0, :])), worker, 1).mean(1)
        # Get second rep non worker pool
        rep1_pool = np.delete(np.squeeze(np.copy(human_individuals[:, 1, :])), worker, 1).mean(1)

        non_worker_pool = (rep0_pool + rep1_pool) / 2
        rep0_trials = human_individuals[:, 0, worker]
        rep1_trials = human_individuals[:, 1, worker]

        consistencies_empirical = [metric(rep0_trials, rep0_pool), metric(rep1_trials, rep1_pool)]
        noise_level.append(np.sqrt(metric(rep0_pool, rep1_pool) * metric(rep0_trials, rep1_trials)))

        assert rep0_pool.shape == (128,)
        individual_to_pool_empirical.extend(consistencies_empirical)

    normalized_humans = []
    for split1, split2, noise in zip(individual_to_pool_empirical[::2], individual_to_pool_empirical[1::2],
                                     noise_level):
        normalized_humans.append(np.mean([split1, split2]) / noise)
    return normalized_humans


def metric(x, y):
    # returns the pearson correlation of inputs
    return ss.spearmanr(x, y)[0]

def split_half_reps(x, rng):

    # Splits the X into two columns and return the mean of each split
    n_reps = x.shape[1]
    shuffrange = range(n_reps)
    rng.shuffle(shuffrange)
    split0 = x[:, shuffrange[0:n_reps / 2]].mean(1)
    split1 = x[:, shuffrange[n_reps / 2:]].mean(1)
    return split0, split1

def empirical_consistency(M, n_boots, get_normalized_humans=True, human_reps=None, human_individuals=None):
    """
    Calculate the empirical consistency between a matrix and human data (grabbed by get_human_data), as well as the
    normalized empirical consistency for every human individual in the population that the data is from (to be used
    as a benchmark)
    """

    if human_reps is None:
        human_reps, human_individiduals, raw = get_human_data_densely_sampled()
    empirical_consistencies = empirical_consistency_to_humans(human_reps, M, n_boots)
    if get_normalized_humans:
        normalized_humans_to_pool_consistencies = normalized_human_to_pool_consistency(human_individiduals)

        return empirical_consistencies, normalized_humans_to_pool_consistencies
    else:
        return empirical_consistencies


# <codecell>


