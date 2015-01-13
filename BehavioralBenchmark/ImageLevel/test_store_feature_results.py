__author__ = 'ardila'
import unittest
import pymongo as pm
from BehavioralBenchmark.ImageLevel import feature_loader, decoder_models, feature_split, store_feature_results
import gridfs


class TestFeatureStorage(unittest.TestCase):
    def setUp(self):
        DB = pm.MongoClient(port=22334)['ModelBehavior']
        feature_name = 'IT'
        feature_split = [1, 2]
        features, meta = feature_loader.get_features_by_name(feature_name)
        decoder_model = decoder_models.StandardModel
        gridfs_name = '_'.join([decoder_model['name'], feature_name, 'results'])
        fs = gridfs.GridFS(DB, gridfs_name) # Decide where to store things
        additional_info = {'feature_split': feature_split, 'test': True}

        self.F = features
        self.decoder_model = decoder_model
        self.meta = meta
        self.fs = fs
        self.feature_inds = [1,2,3,4]
        self.additional_info = additional_info
        self.ids = []

    def test_store_subsampled_features(self):
        idval, results = store_feature_results.store_subsampled_feature_results(self.F, self.meta,
                                                               self.decoder_model,
                                                               self.fs, self.feature_inds, self.additional_info)
        self.ids.append(idval)

    def tearDown(self):
        for idval in self.ids:
            self.fs.delete(idval)

if __name__ == '__main__':
    unittest.main()