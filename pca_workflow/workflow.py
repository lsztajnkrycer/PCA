import pandas as pd
from typing import Union
from pca_workflow.algorithm import PCA

class Workflow:
    """User facing Workflow representing dimensionality reduction experience"""
    
    def workflow(self, data: Union[str, pd.DataFrame]):
        """
        A Workflow example that runs PCA beginning to end
        
        :param data: input data to run the workflow on
        :return: the result, post transformation along the dimensions of principle components
        """
        
        # Reading in data
        input = pd.read_csv(data) if isinstance(data, str) else data
        
        # Instantiating PCA Model
        model = PCA(n_components = 3, workflow = "SVD")
        
        model.fit(data)
        

        