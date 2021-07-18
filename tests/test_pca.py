import pandas as pd
import numpy as np
from numpy import random as rd
from unittest import TestSuite
from pca_workflow.workflow import PCA  


class TestPCAMathematics(TestSuite):

    def test_pca_with_small_dataframe(self):
        """

        :return:
        """
        data = pd.DataFrame(data = [[1,2], [3,4]])
        model = PCA()
        PCA.fit(data)
        final_transform = PCA.transform()


    def test_with_large_dataframe(self):
        #data starts as numpy, so also tests whether conversion checks work
        data = rd.randint(100, size=(50, 75))
        model = PCA()
        PCA.fit(data)
        final_transform = PCA.transform()

    def test_with_duplicate_variables(self):
        """Ensures duplicate variables aren't both counted as principle components"""
        data = pd.DataFrame(data = [[1,2],[1,2],[3,5]])
        model = PCA()
        PCA.fit(data)
        final_transform = PCA.transform()

    def test_with_no_data(self):
        """Ensures errors are passed when we use a dataframe with nothing in it"""
        data = pd.DataFrame()
        model = PCA()
        PCA.fit(data)
        final_transform = PCA.transform()

    def test_with_wrong_data(self):
        data = [8, 9]
        model = PCA()
        PCA.fit(data)
        final_transform = PCA.transform()