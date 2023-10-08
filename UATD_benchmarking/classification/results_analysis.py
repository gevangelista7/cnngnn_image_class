import os
import pandas as pd
from utils import results_dir
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Initialize an empty list to store the data
data = []

if __name__ == "__main__":
    # Iterate over the files in the directory
    for filename in os.listdir(results_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(results_dir, filename)

            # Extract model name and seed from the filename
            model_name, test_or_seed = filename.split('_')[0], filename.split('_')[-1].split('.')[0]

            if "val" in filename:
                test_val = 'val'
            elif "test" in filename:
                test_val = 'test'
            else:
                test_val = None

            # Read the CSV file into a DataFrame
            df = pd.read_csv(filepath)

            # Iterate over rows in the DataFrame
            for index, row in df.iterrows():
                precision = row['precision']
                f1 = row['f1']
                recall = row['recall']

                # Append the data to the list
                data.append([model_name, test_val, precision, f1, recall])

    # Create a DataFrame from the collected data
    columns = ['Model', 'Partition', 'Precision', 'F1', 'Recall']
    result_df = pd.DataFrame(data, columns=columns)

    #
    test_df = result_df[result_df['Partition']=='test']
    #
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=test_df, x='Model', y='Precision', showfliers=True)
    plt.title(f'Precision Box Plot by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=test_df, x='Model', y='F1', showfliers=True)
    plt.title(f'F1 Box Plot')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=test_df, x='Model', y='Recall', showfliers=True)
    plt.title(f'Recall Box Plot by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.show()

    # data = []
    #
    # # Iterate over the files in the directory
    # for filename in os.listdir(results_dir):
    #     if filename.endswith(".csv"):
    #         filepath = os.path.join(results_dir, filename)
    #
    #         # Extract model name and seed from the filename
    #         filename_parts = filename.split('_')
    #         model_name = filename_parts[0]
    #
    #         # The seed is the character between the underscores
    #         seed = filename_parts[2]
    #
    #         # Read the CSV file into a DataFrame
    #         df = pd.read_csv(filepath)
    #
    #         # Extract classification report from CSV (assuming it's stored as a string)
    #         class_report_str = df.iloc[0]['reports']
    #         class_report_dict = ast.literal_eval(class_report_str)
    #
    #         # Iterate over classes in the classification report
    #         for class_name, class_metrics in class_report_dict.items():
    #             precision = class_metrics['precision']
    #             recall = class_metrics['recall']
    #             f1_score = class_metrics['f1-score']
    #
    #             # Append the data to the list
    #             data.append([model_name, seed, class_name, precision, recall, f1_score])
    #
    #     # Create a DataFrame from the collected data
    #     columns = ['Model', 'Seed', 'Class', 'Precision', 'Recall', 'F1']
    #     result_df = pd.DataFrame(data, columns=columns)
    #
    #     # Select a specific model and seed
    #     selected_model = 'vgg16.tv'
    #     partition = 'test'
    #
    #     # Filter the DataFrame for the selected model and seed
    #     model_seed_df = result_df[(result_df['Model'] == selected_model) & (result_df['Seed'] == partition)]
    #
    #     # Plot box plots using seaborn for each class metric
    #     plt.figure(figsize=(10, 6))
    #     sns.boxplot(data=model_seed_df, x='Class', y='Precision', showfliers=False)
    #     plt.title(f'Precision Box Plot)')
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.show()
    #
    #     plt.figure(figsize=(10, 6))
    #     sns.boxplot(data=model_seed_df, x='Class', y='Recall', showfliers=False)
    #     plt.title(f'Recall Box Plot ')
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.show()
    #
    #     plt.figure(figsize=(10, 6))
    #     sns.boxplot(data=model_seed_df, x='Class', y='F1', showfliers=False)
    #     plt.title(f'F1 Box Plot ')
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.show()