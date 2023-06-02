import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

import os

class results_analytics:
    def __init__(self):
        results_path = './results'
        results_list = []

        for root, _, files in os.walk(results_path):
            for file in files:
                file_path = os.path.join(root, file)
                result = pd.read_pickle(file_path)
                results_list.append(result.attrs)

        self.results = pd.DataFrame.from_records(results_list)


# if __name__ == '__main__':
