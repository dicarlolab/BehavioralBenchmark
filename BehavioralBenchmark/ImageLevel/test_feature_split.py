__author__ = 'ardila'
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

if __name__ == '__main__':
    unittest.main()