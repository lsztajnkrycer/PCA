import typing
import numpy as np
from numpy.linalg import eig
import pandas as pd


class Utils:

    @staticmethod
    def standardize_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes Pandas DataFrame

        :param data: a Pandas DataFrame to standardize
        """
        for col in data.columns:
            data[col] = (data[col] - np.mean(data[col]))/ np.std(data[col])

        return data

    @staticmethod
    def vectorize_dataframe(data: pd.DataFrame) -> np.ndarray:
        """
        vectorizes a pandas DataFrame. If given a numpy array to start with, returns the data as is.

        :param data: a pandas dataframe input
        :return: a numpy vectorization
        """
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise AttributeError("Please input a pandas DataFrame or numpy array.")

    @staticmethod
    def singular_value_decomposition(data: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Performs Singular Value Decomposition on a numpy array

        :param data: an mxn numpy array.
        :return: an mxn numpy array eigenvectors, which consists of the eigenvectors of X_transpose * X.
        """
        try:
            U,singular_vals,V_trans = np.linalg.svd(data)
            eigenvectors_xtx = np.transpose(V_trans)
            return singular_vals, eigenvectors_xtx

        except np.linalg.LinAlgError:
            print('SVD computation did not converge.')
            print('Please ensure you are using a full m x n matrix, and please try again.')
            raise AttributeError

    @staticmethod
    def compute_eigenvector(data: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def compute_principal_components_with_svd(data: np.ndarray = None) -> np.ndarray:
        """
        Computes principal components using the svd function defined earlier.

        :param data: an m x n input array for which to compute principle component values
        :return: the principle components for the data array using SVD
        (effectively the eigenvectors of data_transpose * data)
        """
        if data is None:
            raise AttributeError("no data was given")
        else:
            singular_vals, principal_components = Utils.singular_value_decomposition(data)
            return principal_components

    @staticmethod
    def compute_principal_components_with_diagonalization(data: np.ndarray = None) -> np.ndarray:
        """
        :param data: an input for which to compute principle component values
        :return: the principal components using the diagonalization method
        """
        if data is None:
            raise AttributeError("No data was given")
        else:
            # Our principle components are the eigenvalue diagonalization of the variance covariance matrix
            final_result = eig(data * data.transpose())

        return final_result


