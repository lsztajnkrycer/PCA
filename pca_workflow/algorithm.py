import typing
import pandas as pd
from math_utils.linear_algebra_utils import Utils

class PCA:

    def __init__(self, data: pd.DataFrame, n_components: int = 2, workflow: str = "SVD"):
        """

        :param data:
        :param n_components:
        :param workflow:
        """
        self.n_components = n_components
        self.data = data
        self.workflow = workflow
        self.principal_components = None

        # Standardizing and vectorizing
        self.data = Utils.standardize_data(data)
        self.data = Utils.vectorize_dataframe(data)

    def fit(self):
        """
        Computes principle components for a given model, based on user selection of workflow
        """
        if self.workflow == "SVD":
            self.principal_components = Utils.compute_principal_components_with_svd(self.data)
        else:
            self.principal_components = Utils.compute_principal_components_with_diagonalization(self.data)

    def principal_components(self):
        """
        Accessor for principal component matrix

        :return: principle component matrix
        """
        return self.principal_components

    def transform(self):
        """
        Places the data along the dimensions of principal components

        @return modified data
        """
        pass