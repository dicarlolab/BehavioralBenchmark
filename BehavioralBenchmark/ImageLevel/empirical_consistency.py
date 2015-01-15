# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import scipy.stats as ss
import dldata.stimulus_sets.hvm as hvm
import pymongo as pm
import dldata.human_data.confusion_matrices as CM
import numpy as np
from copy import deepcopy
rng = np.random.RandomState(0)

def empirical_consistency(M,n_boots):
    
    def get_human_data_densely_sampled():
        raw_data = CM.get_data('hvm_dense_smp_v6_2rpw', field='category')
        #Add rep number to raw data, then clean
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
            which_rep[worker][filename] = which_rep[worker][filename]+1
        raw_data_with_rep = raw_data.addcols([rep], names=['rep'])

        #Get rid of everything but first two reps, get rid of learning reps (Images of V3 and V0)
        data = raw_data_with_rep[raw_data_with_rep['rep']<2]
        data = data[data['var'] == 'V6']
        #Reformat to matrix
        H = [] #images, reps, worker
        canonical_order = np.unique(data['filename'])
        workers = np.unique(data['WorkerId'])
        n_workers = len(workers)
        for worker in workers:
            worker_data = data[data['WorkerId']==worker]
            rep0 = worker_data[worker_data['rep']==0]
            rep1 = worker_data[worker_data['rep']==1]
            c0 = []
            c1 = []
            for Imid in canonical_order:
                c0.append(rep0[rep0['filename'] == Imid]['correct'])
                c1.append(rep1[rep1['filename'] == Imid]['correct'])
            X = np.column_stack([np.array(c0), np.array(c1)])
            X = np.expand_dims(X,2)
            H.append(X)
            assert set(np.unique(worker_data['filename'])) == set(canonical_order)
        H = np.concatenate(H,2)
        assert H.shape == (128, 2, n_workers)
        Human_individiduals = deepcopy(H)
        Human_reps = np.concatenate((H[:,0,:],H[:,1,:]),1)
        return Human_reps, Human_individiduals

    if len(M.shape) == 1:
        assert len(M) == get_human_data_densely_sampled()[0].shape[0] == get_human_data_densely_sampled()[1].shape[0]


    def empirical_consistency_to_Humans(Human_reps,M,n_boots):
        
        # It computes the individual human to pool empirical consistency  

        
        if len(M.shape) == 1:
            Consistencies = [metric(M, Human_reps.mean(1))]
        else:
            Consistencies = np.zeros((n_boots))
            for nb in range(n_boots):
                spH0, spH1 = split_half_reps(Human_reps)
                spM0, spM1 = split_half_reps(M)

                HH = metric(spH0, spH1)
                MM = metric(spM0, spM1)

                H0M1 = metric(spH0, spM1)
                M0H0 = metric(spM0, spH0)
                H1M0 = metric(spH1, spM0)
                M1H1 = metric(spM1, spH1)
                Consistencies[nb] = mean([H0M1, M0H0 , H1M0 ,M1H1])/np.sqrt(HH*MM)
        return Consistencies

    def Normalized_Human_to_Pool_consistency(Human_individiduals):
        if len(Human_individiduals.shape) != 3: 
            raise AssertionError, 'To compute Normalized_Human_to_Pool_consistency the data should be in format images*reps*workers'
        individual_to_pool_empirical = []
        noise_level = []
        meanindidual_to_meanpool = []
        for worker in range(Human_individiduals .shape[2]):
            #Get first rep non worker pool
            rep0_pool = np.delete(np.squeeze(np.copy(Human_individiduals [:,0,:])), worker, 1).mean(1)
            #Get second rep non worker pool
            rep1_pool = np.delete(np.squeeze(np.copy(Human_individiduals [:,1,:])), worker, 1).mean(1)

            non_worker_pool = (rep0_pool+rep1_pool)/2
            rep0_trials = Human_individiduals [:, 0, worker]
            rep1_trials = Human_individiduals [:, 1, worker]

            consistencies_empirical = [metric(rep0_trials, rep0_pool), metric(rep1_trials, rep1_pool)]
            noise_level.append(sqrt(metric(rep0_pool, rep1_pool)*metric(rep0_trials, rep1_trials)))

            assert rep0_pool.shape == (128,)
            individual_to_pool_empirical.extend(consistencies_empirical)





        normalized_humans = []
        for split1, split2, noise in zip(individual_to_pool_empirical[::2], individual_to_pool_empirical[1::2], noise_level):

            normalized_humans.append(mean([split1,split2])/noise)
        return normalized_humans
    def split_half_reps(X):
            # Splits the X into two columns and retrun the mean of each split
            n_reps = X.shape[1]
            shuffrange = range(n_reps)
            rng.shuffle(shuffrange)
            split0 = X[:,shuffrange[0:n_reps/2]].mean(1)
            split1 = X[:,shuffrange[n_reps/2:]].mean(1)
            return split0,split1
    def metric(x,y):
        # returns the pearson correlation of inputs
        return ss.pearsonr(x,y)[0]
    
    
    
    Human_reps, Human_individiduals = get_human_data_densely_sampled()
    Consistencies = empirical_consistency_to_Humans(Human_reps,M,n_boots)
    normalized_humans_to_pool_consistencies = Normalized_Human_to_Pool_consistency(Human_individiduals)
    
    
    return Consistencies,normalized_humans_to_pool_consistencies



# <codecell>


