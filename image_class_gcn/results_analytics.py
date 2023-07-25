import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from image_class_gcn.utils import loader_from_pyg_list
from model_evaluation import test
import os


SMALL_SIZE = 12
MEDIUM_SIZE = 16

# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


def treat_preprocessing_filename(path):
    filename = os.path.basename(path)
    filename, _ = os.path.splitext(filename)
    filename = filename.replace("pyg_list_", "")

    preprocessing, graph_forming = filename.split("_", 1)

    if 'cnn' in path:
        feature_origin = 'ResNet18'
        finetune = True if 'ft' in filename else False
        cut = filename.split('cut')[1][0]
        cut = int(cut)
        graph_forming = 'knn_w'
    else:
        feature_origin = 'intensity'
        finetune = False
        cut = 0

    return preprocessing, graph_forming, feature_origin, finetune, cut


def plot_learning_curve(file_path_list, mode, run_name, savefig=False):
    """ assert mode in ['train', 'test']"""
    assert mode in ['train', 'test']

    fig_name = run_name+'_'+mode
    mode = mode+'_acc'

    acc_series_list = []
    config_description = ['model', 'n_layers', 'hidden_channels', 'pooling', 'preprocessing']
    for file_path in file_path_list:
        result = pd.read_pickle(file_path)
        acc_series_list.append({result[config_description]: result[mode]})

    pd.DataFrame.from_records(acc_series_list).plot.line()
    if savefig:
        plt.suptitle('')
        plt.savefig(fig_name)
    else:
        plt.show()

# ['model', 'n_layers', 'hidden_channels', 'pooling', 'datetime',
#        'runtime', 'best_train_acc', 'best_val_acc', 'best_epoch',
#        'final_epoch', 'preprocessing', 'graph_forming']


class ResultsAnalytics:
    def __init__(self):
        results_path = './results'
        results_list = []

        self.parameters = ['model', 'n_layers', 'hidden_channels', 'pooling']
        self.run_parameters = ['preprocessing', 'graph_forming', 'feature_origin', 'finetune', 'cut']
        self.full_parameters = self.parameters+self.run_parameters
        self.metrics = ['best_val_acc', 'best_train_acc', 'best_epoch']

        for root, _, files in os.walk(results_path):
            for file in files:
                file_path = os.path.join(root, file)
                result = pd.read_pickle(file_path)
                result.attrs['best_epoch'] = result.epoch[result.test_acc.argmax()]
                result.attrs['filepath'] = file_path
                results_list.append(result.attrs)

        self.results_df = pd.DataFrame.from_records(results_list)
        self.results_df[['preprocessing', 'graph_forming', 'feature_origin', 'finetune', 'cut']] = \
            self.results_df['preprocess_file'].apply(treat_preprocessing_filename).apply(pd.Series)

        self.results_df = self.results_df.rename(columns={'best_test_acc': 'best_val_acc'})

        self.results_df = self.results_df.drop(columns=['best_model'])
        self.results_df.final_epoch.fillna(300)

    def get_df_filtered(self, filters):
        """filters -> dict {column: value}"""
        df = self.results_df.copy(deep=True)
        for filter_column, filter_value in filters.items():
            df = df[df[filter_column].isin(filter_value)]

        return df

    def get_count_by_configuration(self, filters=None):
        if filters is None:
            filters = {}
        df = self.get_df_filtered(filters)
        df['count'] = df.groupby(self.full_parameters).transform('size')
        return df.pivot_table(index=self.full_parameters,
                              values=['count'])

    def run_analysis(self, study_object, parameter, exp_name, filters, savefig):
        df = self.get_df_filtered(filters)
        df[[parameter, study_object]].boxplot(column=study_object, by=parameter)
        plt.suptitle('')
        plt.title('')
        plt.ylabel('Accuracy')
        plt.tight_layout()

        if savefig:
            filter_name = 'general' if len(filters) == 0 else str(tuple(filters.values()))
            fig_name = 'img_'+exp_name+'_'+parameter+'_analysis_'+filter_name
            plt.savefig('./results_analysis/' + fig_name)
        else:
            plt.show()

    def performance_analysis(self, parameter, exp_name, filters=None, savefig=False):
        """filters -> dict {column: value}"""
        self.run_analysis(study_object='best_val_acc',
                          parameter=parameter,
                          filters=filters,
                          exp_name=exp_name+'_perf',
                          savefig=savefig)

    def performance_train_analysis(self, parameter, exp_name, filters=None):
        """filters -> dict {column: value}"""
        self.run_analysis(study_object='best_train_acc',
                          parameter=parameter,
                          filters=filters,
                          exp_name=exp_name+'_perf',
                          savefig=False)

    def learning_analysis(self, parameter, exp_name, filters=None, savefig=False):
        """filters -> dict {column: value}"""
        self.run_analysis(study_object='best_epoch',
                          parameter=parameter,
                          filters=filters,
                          exp_name=exp_name+'_learn',
                          savefig=savefig)

    def parameter_analysis_table(self, parameter, filters=None):
        """filters -> dict {column: value}"""
        if filters is None:
            filters = {}
        df = self.get_df_filtered(filters)
        return df.pivot_table(index=[parameter], values=self.metrics, aggfunc=np.mean)

    def general_results_table(self, filters):
        df = self.get_df_filtered(filters)

        table = df.pivot_table(index=self.full_parameters,
                               values=self.metrics,
                               aggfunc=np.mean)
        table = table.sort_values(by='best_val_acc', ascending=False)

        return table

    def best_results_table(self, best_result_by='model', n_largest=3, filters=None):
        if filters is None:
            filters = {}
        df = self.get_df_filtered(filters)

        df = df.pivot_table(index=self.full_parameters,
                            values=self.metrics,
                            aggfunc=np.mean)
        best_results_df = df.groupby(best_result_by)['best_val_acc'].nlargest(n_largest)

        return best_results_df

    def get_best_results_files(self, filters=None):
        if filters is None:
            filters = {}
        df = self.get_df_filtered(filters)

        df = df.sort_values(by='best_val_acc', ascending=False)

        return df

    # def get_best_model_file(self, best_result_by='model', n_largest=3, filters=None):
    #     if filters is None:
    #         filters = {}
    #     best_results_idx = self.best_results_table(filters).index
    #     files = self.results_df[best_results_idx]
    #
    #     return best_results_df

    # def results_by_class(self, best_result_by='model', n_largest=3, filters=None):
    #     best_val_results = self.best_results_table(best_result_by=best_result_by, n_largest=n_largest,
    #                                                filters=filters, get_files=True)
    #
    #     for _, row in best_val_results.iterrows():
    #         dataset_file = os.path.join('../datasets/UATD_graphs/', *row.preprocess_file[0].split('/')[-2:])
    #         model_file = row['filepath'].replace('results', 'models')
    #
    #         model = joblib.load(model_file)
    #         test_loader = loader_from_pyg_list(dataset_file, partition='Test_2', shuffle=True)
    #         overall_accuracy, class_accuracies = test(model, test_loader, device='cuda')
    #         pass


if __name__ == '__main__':
    # gather all results
    results = ResultsAnalytics()

    # create filters for results of each experiment     # exp 1 e 2: node features -> mean pixel
    filters = {
        'exp1':    {'graph_forming': ['knn_w'],                 # graph forming only with sp distance, sp 75
                    'feature_origin': ['intensity']},
        # 'exp2':    {'graph_forming': ['knn_w_pix_300']},        # graph forming considering pixel distance, sp 300
        # 'exp3':    {'preprocessing': ['original'],              # get all the runs without preprocessing and intensity
        #             'feature_origin': ['intensity']},

        'exp4':    {'feature_origin': ['ResNet18'],             # ResNet w/o finetune
                    'finetune': [False]},
        'exp5':    {'feature_origin': ['ResNet18']},            # ResNet Compare with and without finetune
        # 'exp6':    {'feature_origin': ['ResNet18'],             # ResNet w/ finetune
        #             'finetune': [True]},
    }
    savefig = True
    exp_name = 'exp5'
    parameters = results.full_parameters

    # print('Top 10 Best General Results:')
    # best_results = results.general_results_table(filters[exp_name])[:5]
    # results.results_by_class(filters=filters[exp_name])
    # print(best_results[['best_val_acc', 'best_train_acc']])
    # print(best_results[['best_val_acc', 'best_epoch']])

    for parameter in parameters:
    # for parameter in ['model']:
        results.performance_analysis(parameter=parameter, exp_name=exp_name, filters=filters[exp_name], savefig=savefig)
    #     results.performance_train_analysis(parameter=parameter, exp_name=exp_name, filters=filters[exp_name])
    #     results.learning_analysis(parameter=parameter, exp_name=exp_name, filters=filters[exp_name], savefig=savefig)
        # print('best results by ' + parameter)
        # print(results.best_results_table(best_result_by=parameter, filters=filters[exp_name], n_largest=2))

        pass

    # print("BEST MODEL")
    # best_model = results.get_best_results_files(filters[exp_name]).iloc[0]
    # print(best_model)
    # print(best_model['preprocess_file'])
    # print(best_model['filepath'])

    # ['model', 'n_layers', 'hidden_channels', 'pooling', 'datetime',
    #        'runtime', 'best_train_acc', 'best_val_acc', 'best_epoch',
    #        'final_epoch', 'preprocessing', 'graph_forming']
