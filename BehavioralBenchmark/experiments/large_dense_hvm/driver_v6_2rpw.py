#!/usr/bin/env python
import numpy as np
import sys
import copy
import dldata.stimulus_sets.hvm as hvm
from mturkutils.base import Experiment
import random as rand
from copy import deepcopy

#REPEATS_PER_QE_IMG = 4
ACTUAL_TRIALS_PER_HIT = 64*8*2 # Objects times images per object times rep per image
LEARNING_PERIOD = 16

#repeat_inds = [3440, 3282, 3321, 3802, 5000, 3202, 4041, 4200]
#practice_inds = [880, 720, 760, 1240, 2440, 640, 1480, 1640, 3480, 3360, 4560, 3840, 5040, 3240, 4160, 4240]



class SimpleMatchToSampleExperiment(Experiment):
    def createTrials(self, sampling='without-replacement', verbose=0):
        """
        - Create trials with the given ``html_data``.
        Html data is a spec that can have the following parameters:
        :param dummy_upload: If true, image files are assumed to have been uploaded previouls
        :param preproc: what preproc to use on images
            (see dldata.stimulus_sets.dataset_templates.ImageLoaderPreprocesser)
        :param image_bucket_name: what bucket to upload files to
        :param seed: random seed to use for shuffling
        :param dataset: which dataset to get images from
        :param combs: List of tuples of synsets to measure confusions for
        :param k: Number of times to measure each confusion.
        :param meta_query: subset the dataset according to this query, evaluated once at every meta entry
        sampled equally
        :param: labelfunc: callable that takes a dictionary meta entry and the dataset, and returns the label to be
            printed
        :param: response_images: list of
            tuple of image_urls, imgData, and labels to use for response images. There must be one set
             of responses per confusion to be measured. If this is not
            set, random images from the same category are used by default.
        :param shuffle_test: whether to shuffle order of presentatino of test images
        - Parameter ``sampling`` determines the behavior of image sampling:
          * "without-replacement" (default): no same images will be presented
            across the entire population of subjects.
          * "with-replacement": allows recycling of images.

        """
        assert sampling in ['without-replacement', 'with-replacement']
        html_data = self.html_data

        dataset = html_data.get('dataset')
        preproc = html_data.get('preproc')
        meta_query = html_data.get('meta_query')
        meta_field = html_data.get('meta_field', 'category')
        response_images = html_data.get('response_images')
        dummy_upload = html_data.get('dummy_upload', True)
        image_bucket_name = html_data.get('image_bucket_name')
        combs = html_data['combs']
        labelfunc = html_data.get('labelfunc')
        seed = html_data.get('seed', 0)  # no need to change most cases
        urls = html_data.get('urls')
        meta = html_data.get('meta')
        n_hits = html_data.get('n_hits')
        idx_lrn = list(html_data.get('idx_lrn'))
        #idx_rep = list(html_data.get('idx_rep')) ###
        idx_smp = list(html_data.get('idx_smp'))
        shuffle_test = html_data.get('shuffle_test', False)

        rng = np.random.RandomState(seed=seed)
        actual_trials_per_hit = len(idx_smp)
        #offsets = np.arange(actual_trials_per_hit - 3, -1, -actual_trials_per_hit / float(len(idx_rep))
        #        ).round().astype('int') ###
        
        imgs = []
        labels = []
        imgData = []

        for _n in xrange(n_hits):
            ii = copy.deepcopy(idx_smp)
            rng.shuffle(ii)

            assert len(combs) == len(response_images) == 1
            c, ri = combs[0], response_images[0]

            #for i, offset in enumerate(offsets): ###
            #    ii.insert(offset, idx_rep[i]) ###
            #    ii_new = idx_lrn + ii  ###
            ii_new = idx_lrn + ii ###
            for i in ii_new:
                sample = urls[i]
                # sample_meta = meta[i]
                meta0 = meta[i]
                sample_meta = {name: value for name, value in
                             zip(meta0.dtype.names, meta0.tolist())}
                
                test = ri['urls']
                test_meta = ri['meta']
                lbls = ri['labels']

                si = range(len(lbls))
                assert len(si) == len(test) == len(test_meta)
                if shuffle_test:
                    rng.shuffle(si)

                # write down one prepared trial
                imgs.append([sample, [test[e] for e in si]])
                imgData.append({
                    "Sample": sample_meta,
                    "Test": [test_meta[e] for e in si]})
                labels.append([lbls[e] for e in si])

        # v DONT DO THIS BECAUSE IT BREAKS THE HIT BOUNDARY!!!!!
        # for list_data in [imgs, imgData, labels]:
        #     rng = np.random.RandomState(seed=seed)
        #     rng.shuffle(list_data)

        """
        if verbose > 0:
            print '** max len in left synset_urls =', \
                sorted([(len(synset_urls[e]), e)
                for e in synset_urls])[-1]
            print '** max len in left category_meta_dicts =', \
                sorted([(len(category_meta_dicts[e]), e)
                    for e in category_meta_dicts])[-1]
        if verbose > 1:
            print '** len for each in left synset_urls =', \
                {e: len(synset_urls[e]) for e in synset_urls}
            print '** len for each in left category_meta_dicts =', \
                {e: len(category_meta_dicts[e])
                    for e in category_meta_dicts}
        if verbose > 2:
            print '** synset_urls =', synset_urls
            print '** category_meta_dicts =', category_meta_dicts
        """

        self._trials = {'imgFiles': imgs, 'imgData': imgData, 'labels': labels,
                       'meta_field': [meta_field] * len(labels)}




def get_exp(sandbox=True, dummy_upload=True, short_hit=False):

    dataset = hvm.HvMWithDiscfade()
    # meta = dataset.meta ###
    meta_H = dataset.meta ###
    #inds = np.arange(len(meta))


    n_repeats = 2
    #get inds and practice_inds from file
    inds = list(np.load('512_inds.npy'))
    practice_inds = list(np.load('practice_inds.npy'))
    assert len(inds) == 512
    inds = inds*n_repeats

    def test_inds(inds,n_repeats, practice_inds):
        assert len(inds) == 512*n_repeats
        #Test that there are 4 per object, and 64 objects
        object_count = {}
        for i in inds:
            print 'Counting object %s'%(meta_H['obj'][i])
            object_count[meta_H['obj'][i]] = object_count.get(meta_H['obj'][i], 0) + 1
        print 'Number of unique objects'
        print object_count.keys()
        print len(object_count.keys())
        assert len(object_count.keys()) == 64
        for obj in object_count.keys():
            assert object_count[obj] == 8*n_repeats
        print '__________'
        print len(np.unique(inds))
        assert len(np.unique(inds))*n_repeats == len(inds)
        assert len(set(inds)&set(practice_inds)) == 0
    test_inds(inds,n_repeats, practice_inds)


    meta = meta_H[inds] ###
    #n_hits_from_data = len(meta) / ACTUAL_TRIALS_PER_HIT
    n_hits_from_data = len(meta) ###
    categories =  np.unique(meta['category'])  # dataset.categories ###
    combs = [categories]


    preproc = None
    image_bucket_name = 'hvm_timing'
    urls = dataset.publish_images(range(len(meta_H)), preproc,
                                  image_bucket_name,
                                  dummy_upload=dummy_upload)

    base_url = 'https://canonical_images.s3.amazonaws.com/'
    response_images = [{
        'urls': [base_url + cat + '.png' for cat in categories],
        'meta': [{'category': 'Animals'},
                 {'category': 'Boats'},
                 {'category': 'Cars'},
                 {'category': 'Chairs'},
                 {'category': 'Faces'},
                 {'category': 'Fruits'},
                 {'category': 'Planes'},
                 {'category': 'Tables'}],
        'labels': categories}]

    mult =  100 ### 2
    #ind_repeats = repeat_inds * REPEATS_PER_QE_IMG ###
    #rng = np.random.RandomState(0) ###
    #rng.shuffle(ind_repeats) ###
    ind_learn = practice_inds
    if short_hit:
        inds = inds[0:10]
    html_data = {
            'response_images': response_images,
            'combs': combs,
            # 'num_trials': 90 * 64 * mult,
            'meta_field': 'category',
            'meta': meta_H,
            'idx_smp': inds,
            #'idx_rep': ind_repeats, ###
            'idx_lrn': ind_learn,
            'urls': urls,
            'n_hits': mult,
            'shuffle_test': False,
    }

    additionalrules = [{'old': 'LEARNINGPERIODNUMBER',
                        'new':  str(LEARNING_PERIOD)}]

    trials_per_hit = ACTUAL_TRIALS_PER_HIT +16
    exp = SimpleMatchToSampleExperiment(
            htmlsrc='large_dense_one_worker_per_hit.html',
            htmldst='large_dense_one_worker_per_hit_n%05d.html',
            tmpdir='tmp_dense_smp_v6_2rpw',
            sandbox=sandbox,
            title='Object recognition --- report what you see',
            reward=0.6,
            duration=1500,
            keywords=['neuroscience', 'psychology', 'experiment', 'object recognition'],  # noqa
            description="Complete a visual object recognition task where you report the identity of objects you see. We expect this HIT to take about 10 minutes or less, though you must finish in under 25 minutes.  By completing this HIT, you understand that you are participating in an experiment for the Massachusetts Institute of Technology (MIT) Department of Brain and Cognitive Sciences. You may quit at any time, and you will remain anonymous. Contact the requester with questions or concerns about this experiment.",  # noqa
            comment="hvm dense sampling of 512 V6 images, 2reps per worker",  # noqa
            collection_name = 'large_dense_hvm',
            max_assignments=1,
            bucket_name='large_dense_hvm',
            trials_per_hit=trials_per_hit,  # 144 + 8x4 repeats + 16 training
            html_data=html_data,
            frame_height_pix=1200,
            othersrc = ['dltk.js', 'dltkexpr.js', 'dltkrsvp.js'],
            additionalrules=additionalrules,
            log_prefix='large_dense_hvm__'
            )
    # -- create trials
    exp.createTrials(verbose=1)
    all_ids = [m['Sample']['_id'] for m in exp._trials['imgData']]
    ids = set([str(_) for _ in np.unique(all_ids)])
    #exp.createTrials(sampling='with-replacement', verbose=1) ###
    n_total_trials = len(exp._trials['imgFiles'])
    #assert n_total_trials == mult * (len(meta) + 32 + 16) ###
    if not short_hit:
        assert n_total_trials == mult * (len(meta) +  16) ###

    return exp, html_data


if __name__ == '__main__':
    sandbox = bool(int(sys.argv[1]))
    dummy_upload = bool(int(sys.argv[2]))
    exp, _ = get_exp(sandbox=sandbox, dummy_upload=dummy_upload)
    exp.prepHTMLs()
    exp.testHTMLs()
    exp.uploadHTMLs()
    exp.createHIT(secure=True)

    #hitids = cPickle.load(open('3ARIN4O78FSZNXPJJAE45TI21DLIF1_2014-06-13_16:25:48.143902.pkl'))
    #exp.disableHIT(hitids=hitids)
