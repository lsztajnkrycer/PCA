import pandas as pd

class Workflow:
    """User facing Workflow representing dimensionality reduction experience"""
    def workflow(self):
        data = pd.read_csv(input("Please attach a csv link to a (Pandas DataFrame): "))