import os
import numpy as np
import pandas as pd
from utils import results_dir
import matplotlib.pyplot as plt
import seaborn as sns
import ast

precision_by_class = {str(i): [] for i in range(10)}  # Assuming you have 10 classes

# Initialize a list to store model names
model_names = []
data = []

if __name__ == "__main__":
    for filename in os.listdir(results_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(results_dir, filename)

            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)

            # Extract the model name and test/val identifier from the filename
            model_name, test_or_seed = filename.split('_')[0], filename.split('_')[-1].split('.')[0]

            if "val" in filename:
                test_val = 'val'
            elif "test" in filename:
                test_val = 'test'
            else:
                test_val = None
            # Check if it's a test file
            if test_val == "test":
                model_names.append(model_name)

                # Extract the precision values from the classification report
                for report_str in df['reports']:
                    report_lines = report_str.strip().split('\n')
                    class_precision = {}
                    for line in report_lines[2:12]:  # Assuming there are 10 classes
                        parts = line.split()
                        class_name = parts[0]
                        precision = float(parts[1])
                        recall = float(parts[2])
                        f1 = float(parts[3])

                        data.append([model_name, class_name, precision, f1, recall])

            columns = ['Model', 'class_name', 'Precision', 'F1', 'Recall']
            result_df = pd.DataFrame(data, columns=columns)

    pivot_table = result_df.pivot_table(
        values=['Precision', 'F1', 'Recall'],
        index=['Model', 'class_name'],
        aggfunc={'Precision': ['mean', 'std'], 'F1': ['mean', 'std'], 'Recall': ['mean', 'std']}
    )

    precision_table = pivot_table['Precision'].unstack(level=1)
    f1_table = pivot_table['F1'].unstack(level=1)
    recall_table = pivot_table['Recall'].unstack(level=1)

    # for model in set(model_names):
    #     model_df = result_df[result_df['Model'] == model_name]
    #     for metric in ['Precision', 'F1', 'Recall']:
    #         plt.figure(figsize=(10, 6))
    #         sns.boxplot(data=model_df, x='class_name', y=metric, showfliers=True)
    #         plt.title(f'{metric} Box Plot by Class, {model}')
    #         plt.xticks(rotation=45)
    #         plt.tight_layout()
    #         plt.grid()
    #         plt.show()

    pass