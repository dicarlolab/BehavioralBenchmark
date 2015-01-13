__author__ = 'ardila'
import unittest

from BehavioralBenchmark.ImageLevel.feature_split import feature_split

class TestFeatureSplit(unittest.TestCase):
    """
    Test a few cases of feature splitting
    """

    def test_feature_split(self):
        answer = [[5, 2], [1, 3], [0, 4], [2, 3, 0], [5, 1, 4]]
        test_value = feature_split(6, 2, 1)
        print test_value
        self.assertItemsEqual(answer, test_value)

if __name__ == '__main__':
    unittest.main()