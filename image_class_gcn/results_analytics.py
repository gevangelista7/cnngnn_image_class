import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def treat_preprocessing_filename(path):
    filename = os.path.basename(path)
    filename, _ = os.path.splitext(filename)
    filename = filename.replace("pyg_list_", "")
    preprocessing, graph_forming = filename.split("_", 1)
    return preprocessing, graph_forming


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
#        'runtime', 'best_train_acc', 'best_test_acc', 'best_epoch',
#        'final_epoch', 'preprocessing', 'graph_forming']


class ResultsAnalytics:
    def __init__(self):
        results_path = './results'
        results_list = []

        self.parameters = ['model', 'n_layers', 'hidden_channels', 'pooling', 'preprocessing', 'graph_forming']
        self.metrics = ['best_test_acc', 'best_train_acc', 'best_epoch']

        for root, _, files in os.walk(results_path):
            for file in files:
                file_path = os.path.join(root, file)
                result = pd.read_pickle(file_path)
                result.attrs['best_epoch'] = result.epoch[result.test_acc.argmax()]
                result.attrs['filepath'] = file_path
                results_list.append(result.attrs)

        self.results_df = pd.DataFrame.from_records(results_list)
        self.results_df[['preprocessing', 'graph_forming']] = \
            self.results_df['preprocess_file'].apply(treat_preprocessing_filename).apply(pd.Series)

        self.results_df = self.results_df.drop(columns=['preprocess_file', 'best_model'])
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
        df['count'] = df.groupby(self.parameters).transform('size')
        return df.pivot_table(index=self.parameters,
                              values=['count'])

    def run_analysis(self, study_object, parameter, exp_name, filters, savefig):
        df = self.get_df_filtered(filters)
        df[[parameter, study_object]].boxplot(column=study_object, by=parameter)
        plt.suptitle('')

        if savefig:
            filter_name = 'general' if len(filters) == 0 else str(tuple(filters.values()))
            fig_name = 'img_'+exp_name+'_'+parameter+'_analysis_'+filter_name
            plt.savefig('./results_analysis/' + fig_name)
        else:
            plt.show()

    def performance_analysis(self, parameter, exp_name, filters=None, savefig=False):
        """filters -> dict {column: value}"""
        self.run_analysis(study_object='best_test_acc',
                          parameter=parameter,
                          filters=filters,
                          exp_name=exp_name+'_perf',
                          savefig=savefig)

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

        table = df.pivot_table(index=self.parameters,
                               values=self.metrics,
                               aggfunc=np.mean)
        table = table.sort_values(by='best_test_acc', ascending=False)

        return table

    def best_results_table(self, best_result_by='model', filters=None):
        if filters is None:
            filters = {}
        df = self.get_df_filtered(filters)
        # df = df[['model', 'n_layers', 'hidden_channels', 'pooling', 'preprocessing',
        #         'best_test_acc', 'best_train_acc', 'best_epoch']]

        df = df.pivot_table(index=self.parameters,
                            values=self.metrics,
                            aggfunc=np.mean)

        idx = df.groupby(best_result_by)['best_test_acc'].idxmax()

        return df.loc[idx]


if __name__ == '__main__':
    # gather all results
    results = ResultsAnalytics()

    # create filters for results of each experiment     # exp 1 e 2: node features -> mean pixel
    filters = {
        'general': {'graph_forming': ['knn_w', 'knn_w_pix_300'],
                    'preprocessing': ['original']},       # get all the runs
        'exp1':    {'graph_forming': ['knn_w']},          # graph forming only with sp distance, sp 75
        'exp2':    {'graph_forming': ['knn_w_pix_300']}   # graph forming considering pixel distance, sp 300
    }
    savefig = True
    exp_name = 'general'

    print('Top 10 Best General Results:')
    print(results.general_results_table(filters[exp_name])[:10][['best_test_acc', 'best_train_acc']])
    print(results.general_results_table(filters[exp_name])[:10][['best_test_acc', 'best_epoch']])

    if exp_name == 'general':
        parameters = ['graph_forming']
    else:
        parameters = ['model', 'preprocessing', 'n_layers', 'pooling', 'hidden_channels']

    for parameter in parameters:
        results.performance_analysis(parameter=parameter, exp_name=exp_name, filters=filters[exp_name], savefig=savefig)
        results.learning_analysis(parameter=parameter, exp_name=exp_name, filters=filters[exp_name], savefig=savefig)
        print('best results by ' + parameter)
        print(results.best_results_table(best_result_by=parameter, filters=filters[exp_name]))

        pass

    # ['model', 'n_layers', 'hidden_channels', 'pooling', 'datetime',
    #        'runtime', 'best_train_acc', 'best_test_acc', 'best_epoch',
    #        'final_epoch', 'preprocessing', 'graph_forming']
