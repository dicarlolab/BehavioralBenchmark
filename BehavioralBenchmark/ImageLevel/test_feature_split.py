__author__ = 'ardila'
from collections import defaultdict
import unittest

from BehavioralBenchmark.ImageLevel.feature_split import feature_split

class TestFeatureSplit(unittest.TestCase):
    """
    Test a few cases of feature splitting
    """

    def test_feature_split(self):
        answer = [[5, 2], [1, 3], # 2 non overlapping of size 2
                    [2, 3, 0], [5, 1, 4]] # 2 non overlapping of size 3
        answer_with_bootstrap = [[5, 2], [1, 3], # 2 non overlapping of size 2
                                 [2, 3], [0, 5], # 2 non overlapping of size 2
                                    [5, 4, 3], [0, 1, 2], # 2 non overlapping of size 3
                                    [2, 3, 0], [1, 4, 5]] # 2 non overlapping of size 3


        test_value = feature_split(6, 2, 1, 2)
        self.assertItemsEqual(answer, test_value)
        test_value = feature_split(6, 2, 2, 2)
        print test_value
        self.assertItemsEqual(answer_with_bootstrap, test_value)

    def test_large_feature_split(self):
        feature_splits = feature_split(128, 30, 1, 2)
        features = defaultdict(set)
        for fs in feature_splits:
            overlapping = False
            for f in fs:
                if f not in features[len(fs)]:
                    features[len(fs)].add(f)
                else:
                    overlapping = True
                assert not overlapping, "%s \n %s"%(features[len(fs)], fs)

if __name__ == '__main__':
    unittest.main()
