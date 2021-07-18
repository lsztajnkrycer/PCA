import pandas as pd
import numpy as np
from numpy import random as rd
from unittest import TestSuite


class TestPCAMathematics(TestSuite):

    def test_pca_with_small_dataframe(self):
        """

        :return:
        """
        data = pd.DataFrame(data = [[1,2], [3,4]])


    def test_with_large_dataframe(self):
        #data starts as numpy, so also tests whether conversion checks work
        data = rd.randint(100, size=(50, 75))

    def test_with_duplicate_variables(self):
        """Ensures duplicate variables aren't both counted as principle components"""
        pass

    def test_with_no_data(self):
        """Ensures errors are passed when we use a dataframe with nothing in it"""
        data = pd.DataFrame()