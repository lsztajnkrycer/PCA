import typing
import pandas as pd
from math_utils.linear_algebra_utils import Utils
import copy

class PCA:
    """PCA model class"""

    def __init__(self, data: pd.DataFrame, n_components: int = 2, workflow: str = "SVD"):
        """
        Initializes data object, number of components, and dimensionality reduction workflow

        :param data:
        :param n_components:
        :param workflow:
        """
        self.n_components = n_components
        self.data = data
        self.workflow = workflow
        self._principal_components = None

        # Standardizing and vectorizing
        self.data = Utils.standardize_data(data)
        self.data = Utils.vectorize_dataframe(data)

    def __deepcopy__(self, memodict={}):
        """
        Defines prototype pattern for PCA class

        :return: a copy of the PCA class
        """
        return copy.deepcopy(self)

    def fit(self):
        """
        Computes principle components for a given model, based on user selection of workflow
        """
        if self.workflow == "SVD":
            self._principal_components = Utils.compute_principal_components_with_svd(self.data)
        else:
            self._principal_components = Utils.compute_principal_components_with_diagonalization(self.data)

        # Removing all unwanted principle components
        self._principal_components = self.principal_components[,0:self.n_components]

    @property
    def principal_components(self):
        """
        Accessor for principal component matrix

        :return: principle component matrix
        """
        return self._principal_components

    def transform(self):
        """
        Places the data along the dimensions of principal components

        :return modified data
        """
        transformed = (self._principal_components.transpose() * self.data.transpose())
        return transformed