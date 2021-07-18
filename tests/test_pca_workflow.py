from unittest import TestCase
import pandas as pd
from pca_workflow.workflow import Workflow
import numpy as np

class TestWorkflow(TestCase):
    def test_workflow(self):
        data = pd.DataFrame({"a": [1,2,3,4], "b": [5,5, 7, 1], "c": [9,2,3,2], "d": [1,2,5,6]})
        results = Workflow.workflow(data)
        print(results)
        assert isinstance(results, np.ndarray)


