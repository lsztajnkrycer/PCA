import typing
import numpy as np
import pandas as pd
from math_utils.linear_algebra_utils import Utils
import copy

class PCA:
    """PCA model class"""

    def __init__(self, n_components: int = 2, workflow: str = "SVD"):
        """
        Initializes data object, number of components, and dimensionality reduction workflow

        :param data:
        :param n_components:
        :param workflow:
        """
        self.n_components = n_components
        self.workflow = workflow
        self._principal_components = None
        self.data = None

    def __deepcopy__(self, memodict={}):
        """
        Defines prototype pattern for PCA class

        :return: a copy of the PCA class
        """
        return copy.deepcopy(self)

    def __str__(self,):
        return "PCA model with attributes of: n_components - " + str(self.n_components) + ", implementation: " + str(self.workflow)

    def fit(self, data: pd.DataFrame):
        """
        Computes principle components for a given model, based on user selection of workflow
        """
        # Attributing
        self.data = data
        try:
            assert isinstance(data, pd.DataFrame) and data is not None and len(data.columns) > 0, "Must provide data as viable DataFrame"
        # Raising so as to be caught by tests
        except AssertionError as error:
            raise AssertionError(error.__str__())

        # Standardizing and vectorizing
        self.data = Utils.standardize_data(self.data)
        self.data = Utils.vectorize_dataframe(self.data)

        # Deciding on methodology
        if self.workflow == "SVD":
            self._principal_components = Utils.compute_principal_components_with_svd(self.data)
        else:
            self._principal_components = Utils.compute_principal_components_with_diagonalization(self.data)

        # Removing all unwanted principle components
        self._principal_components = self.principal_components[0:self.n_components]

    def data(self):
        """
        A getter for data instance (not set as property for fit modification purposes)
        :return: the working data
        """
        return self.data

    @property
    def principal_components(self) -> np.ndarray:
        """
        Accessor for principal component matrix

        :return: principle component matrix
        """
        return self._principal_components

    def transform(self, data: np.ndarray= None) -> np.ndarray:
        """
        Places the data along the dimensions of principal components

        :return modified data
        """
        if data is None:
            data = self.data

        transformed = np.matmul(self.data.transpose(), self._principal_components.transpose())
        return transformed
