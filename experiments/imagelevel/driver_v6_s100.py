#!/usr/bin/env python
import numpy as np
import sys
import copy
import dldata.stimulus_sets.hvm as hvm
from mturkutils.base import Experiment
import experiments.image_level_benchmark




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

        # Get parameters from html data

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
        idx_smp = list(html_data.get('idx_smp'))


        rng = np.random.RandomState(seed=seed)
        actual_trials_per_hit = len(idx_smp)

        imgs = []
        labels = []
        imgData = []

        for _n in xrange(n_hits):
            ii = copy.deepcopy(idx_smp)
            rng.shuffle(ii)
            response_image_for_this_combination = response_images[0]
            assert len(combs) == len(response_images) == 1

            for i in idx_lrn+ii:
                sample = urls[i]
                # sample_meta = meta[i]
                meta0 = meta[i]
                sample_meta = {name: value for name, value in
                             zip(meta0.dtype.names, meta0.tolist())}
                
                test = response_image_for_this_combination['urls']
                test_meta = response_image_for_this_combination['meta']
                lbls = response_image_for_this_combination['labels']

                si = range(len(lbls))
                assert len(si) == len(test) == len(test_meta)

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




def get_exp(sandbox=True, dummy_upload=True):
    LEARNING_PERIOD = 10


    practice_inds_low_var = range(0, 80, 10)
    practice_inds_high_var = range(3200, 3200+90*8,90)
    practice_inds = practice_inds_low_var+practice_inds_high_var

    repeats_per_image =  2 ### 2
    workers_per_image = 60

    dataset = hvm.HvMWithDiscfade()
    # meta = dataset.meta ###
    meta_H = dataset.meta ###
    #inds = np.arange(len(meta))
    inds = experiments.image_level_benchmark.INDS
    assert(len(set(practice_inds).intersection(set(inds))) == 0)
    meta = meta_H[inds] ###
    #n_hits_from_data = len(meta) / ACTUAL_TRIALS_PER_HIT
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


    inds_shown_per_hit = inds*repeats_per_image
    rng = np.random.RandomState(0)
    ind_learn = practice_inds
###############################################################
    html_data = {
            'response_images': response_images,
            'combs': combs,
            # 'num_trials': 90 * 64 * mult,
            'meta_field': 'category',
            'meta': meta_H,
            'idx_smp': inds,
            'idx_lrn': ind_learn,
            'urls': urls,
            'n_hits': workers_per_image,
            'shuffle_test': False,
    }

    additionalrules = [{'old': 'LEARNINGPERIODNUMBER',
                        'new':  str(LEARNING_PERIOD)}]

    trials_per_hit = len(inds_shown_per_hit)+len(practice_inds)
    exp = SimpleMatchToSampleExperiment(
            htmlsrc='hvm_dense_smp_v6_s100.html',
            htmldst='hvm_dense_smp_v6_s100_n%05d.html',
            tmpdir='tmp_dense_smp_v6_s100',
            sandbox=sandbox,
            title='Object recognition --- report what you see',
            reward=0.10,
            duration=1500,
            keywords=['neuroscience', 'psychology', 'experiment', 'object recognition'],  # noqa
            description="***You may complete as many HITs in this group as you want*** Complete a visual object recognition task where you report the identity of objects you see. We expect this HIT to take about 10 minutes or less, though you must finish in under 25 minutes.  By completing this HIT, you understand that you are participating in an experiment for the Massachusetts Institute of Technology (MIT) Department of Brain and Cognitive Sciences. You may quit at any time, and you will remain anonymous. Contact the requester with questions or concerns about this experiment.",  # noqa
            comment="hvm dense sampling of 100 V6 images",  # noqa
            collection_name = 'hvm_dense_smp_v6_s100',
            max_assignments=1,
            bucket_name='hvm_dense_smp_v6_s100',
            trials_per_hit=trials_per_hit,  # 144 + 8x4 repeats + 16 training
            html_data=html_data,
            frame_height_pix=1200,
            othersrc = ['../../lib/dltk.js', '../../lib/dltkexpr.js', '../../lib/dltkrsvp.js'],
            additionalrules=additionalrules,
            log_prefix='hvm_dense_smp_v6_s100__'
            )

    # -- create trials
    exp.createTrials(verbose=1)
    #exp.createTrials(sampling='with-replacement', verbose=1) ###
    n_total_trials = len(exp._trials['imgFiles'])
    assert n_total_trials == mult * (len(meta) + 32 + 16)

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
