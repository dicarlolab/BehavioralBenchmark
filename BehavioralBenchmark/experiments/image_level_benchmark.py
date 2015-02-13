__author__ = 'ardila'
import numpy as np

import dldata.stimulus_sets.hvm as hvm
dataset = hvm.HvMWithDiscfade()
meta = dataset.meta

inds = []

rng = np.random.RandomState(0)
for obj in np.unique(dataset.meta['obj']):
    var6_obj = lambda x: x['obj'] == obj and x['var'] == 'V6'
    var6_obj_inds = np.ravel(np.argwhere(map(var6_obj, meta)))
    eight_obj_inds = rng.choice(var6_obj_inds, 8, replace=False)
    inds.extend(eight_obj_inds)

def test_inds(inds):
    assert len(inds) == 512
    #Test that there are 8 per object, and 64 objects
    object_count = {}
    for i in inds:
        object_count[meta['obj'][i]] = object_count.get(meta['obj'][i], 0) + 1

    assert len(object_count.keys()) == 64
    for obj in object_count.keys():
        assert object_count[obj] == 8
    print np.unique(inds).shape[0]
    print sorted(list(inds))
    assert np.unique(inds).shape[0] == 512

test_inds(inds)
INDS = inds


np.save('512_inds.npy', np.array(inds))
