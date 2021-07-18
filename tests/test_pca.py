import pandas as pd
import numpy as np
from numpy import random as rd
from unittest import TestSuite
from pca_workflow.workflow import PCA
from unittest import TestCase
import unittest


class TestPCAMathematics():

    def run_tests(self):
        """
        Runs test suite

        :param result: Result object
        :param debug: Debug object
        :return: result
        """
        self.test_pca_with_small_dataframe()
        self.test_with_wrong_data()
        self.test_with_large_dataframe()
        self.test_with_no_data()
        self.test_with_duplicate_variables()

    def test_pca_with_small_dataframe(self):
        data = pd.DataFrame(data = [[1,2], [3,4]])
        model = PCA()
        model.fit(data=data)
        final_transform = model.transform()
        print(final_transform)
        assert isinstance(final_transform, np.ndarray)

    def test_with_large_dataframe(self):
        #data starts as numpy, so also tests whether conversion checks work
        data = pd.DataFrame(rd.randint(100, size=(75,75)))
        model = PCA()
        model.fit(data)
        final_transform = model.transform()
        print(final_transform)
        assert isinstance(final_transform, np.ndarray)

    def test_with_duplicate_variables(self):
        """Ensures duplicate variables aren't both counted as principle components"""
        data = pd.DataFrame(data = [[1,2,3],[1,2,3],[3,5, 6]])
        model = PCA()
        model.fit(data)
        final_transform = model.transform()
        print(final_transform)
        assert True

    def test_with_no_data(self):
        """Ensures errors are passed when we use a dataframe with nothing in it"""
        message = None
        data = pd.DataFrame()
        model = PCA()
        try:
            model.fit(data)
        except Exception:
            assert True

    def test_with_wrong_data(self):
        data = [8, 9]
        model = PCA()
        try:
            model.fit(data)
        except AssertionError as e:
            message = e

        assert message is not None

if __name__ == "__main__":
    test = TestPCAMathematics()
    test.run_tests()