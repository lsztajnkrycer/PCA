import typing
import numpy as np
from numpy.linalg import eig
import pandas as pd


class Utils:

    def standardize_data(self, data: pd.DataFrame):
        """
        Standardizes Pandas DataFrame

        :param data: a Pandas DataFrame to standardize
        """
        for col in data.columns:
            data[col] = (data[col] - np.mean(data[col]))/ np.std(data[col])

    def vectorize_dataframe(self, data: pd.DataFrame):
        """
        vectorizes a pandas dataframe

        :param data: a pandas dataframe input
        :return: a numpy vectorization
        """
        return data.to_numpy()

    def singular_value_decomposition(self):
        pass

    def compute_eigenvector(self):

        return eig()

    def compute_covariance_matrix(self):
        pass


if __name__ == "__main__":
    util = Utils()
    df = pd.DataFrame({"a": [1,2,3], "b":[1,2,3]})
    print(util.vectorize_dataframe(df))


