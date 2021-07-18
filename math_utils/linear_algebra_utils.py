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

    def compute_covariance_matrix(self, X: np.ndarray):
        """
        Computes the covariance matrix for a given matrix

        :return: computes the covariance of a given numpy array
        """
        return np.cov(X)

    def singular_value_decomposition(self, data: np.ndarray):
        """
        Performs Singular Value Decomposition on a numpy array

        :param data: an mxn numpy array.
        :return: an mxn numpy array eigenvectors, which consists of the eigenvectors of X_transpose * X.
        """
        try:
            U,S,V_trans = np.linalg.svd(data)
            eigenvectors_xtx = np.transpose(V_trans)
            return eigenvectors_xtx

        except np.linalg.LinAlgError:
            print('SVD computation did not converge.')
            print('Please ensure you are using a full m x n matrix, and please try again.')
            raise AttributeError

    def compute_eigenvector(self, data: np.ndarray):
        """
        computes the eigenvectors of a single square array

        :return: the eigenvectors of a single square array
        """
        try:
            values, vectors = eig(data)

        # Elaboration upon error message
        except np.linalg.LinAlgError:
            print("computation failed, please ensure"
                  "you've passed in a proper n-dimensional array")
            print("Please recall the function with proper input")
            raise AttributeError

        # Returning eigenvectors
        return vectors

    def compute_principal_components_with_svd(self):
        pass

    def compute_principal_components_with_diagonalization(self):
        pass