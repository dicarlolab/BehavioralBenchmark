__author__ = 'ardila'
#!/usr/bin/env python
import numpy as np
import sys
import copy
import dldata.stimulus_sets.hvm as hvm
from mturkutils.base import Experiment
import random as rand
from copy import deepcopy

#REPEATS_PER_QE_IMG = 4
ACTUAL_TRIALS_PER_HIT = 64*2*2
LEARNING_PERIOD = 16

#repeat_inds = [3440, 3282, 3321, 3802, 5000, 3202, 4041, 4200]
#practice_inds = [880, 720, 760, 1240, 2440, 640, 1480, 1640, 3480, 3360, 4560, 3840, 5040, 3240, 4160, 4240]

EXPERIMENT_1_INDS = {3210, 3215, 3253, 3278, 3293, 3295, 3334, 3345, 3373, 3375, 3410, 3429, 3446, 3454, 3481, 3484,
                     3535, 3559, 3576, 3598, 3604, 3619, 3649, 3670, 3682, 3718, 3743, 3747, 3775, 3784, 3802, 3839,
                     3869, 3878, 3880, 3918, 3947, 3949, 3982, 3999, 4011, 4028, 4048, 4058, 4103, 4106, 4146, 4150,
                     4161, 4171, 4203, 4228, 4257, 4258, 4284, 4292, 4334, 4355, 4365, 4390, 4409, 4414, 4451, 4466,
                     4501, 4507, 4539, 4556, 4575, 4580, 4603, 4606, 4644, 4658, 4680, 4694, 4735, 4756, 4780, 4792,
                     4810, 4811, 4872, 4878, 4898, 4906, 4932, 4933, 4971, 4976, 5004, 5032, 5044, 5050, 5085, 5105,
                     5144, 5156, 5180, 5199, 5203, 5212, 5240, 5243, 5288, 5298, 5329, 5353, 5394, 5396, 5417, 5424,
                     5453, 5472, 5481, 5487, 5525, 5559, 5577, 5592, 5633, 5635, 5641, 5642, 5687, 5696, 5725, 5756}

PRACTICE_INDS = [20, 0, 50, 60, 2650, 70, 30, 40, 10, 2740, 3100,
                 3010, 2560, 2830, 2920, 3190]

class RepeatedExperiment(Experiment):
    """
    Plan: 1 hit with 512 images, each repeated once. Make sure to
    """
    def createTrials
    # Calculate indices to test, making sure to include the old ones, for a total of 8 per object
    dataset = hvm.HvMWithDiscfade()
    obj_counts = {}
    inds = []
    for i, meta_entry in enumerate(dataset.meta):
        if i in EXPERIMENT_1_INDS:
            inds.append(i)
        if meta_entry['var'] == 'V6':
            obj = meta_entry['obj']
            # There are 2 per object guaranteed, so we need 6 more
            if obj_counts[obj] < 6:
                obj_counts[obj] = obj_counts.get(obj, 0) + 1
                inds.append(i)
########## IN PROGRESS

    self._trials = {'imgFiles': imgs, 'imgData': imgData, 'labels': labels,
                    'meta_field': [meta_field] * len(labels)}
