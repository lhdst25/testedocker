"""
PATTERN RECOGNITION
"""

# System Libraries
import copy
import math
import numpy as np
import pandas as pd
import time
from scipy.spatial.distance import cityblock
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Third-party Libraries
from DBN.tensorflow import SupervisedDBNRegression as DBNr


def create_dataframe(dict_in):
    """ Create dataframe from jsons. """
    print("Creating dataframe...")
    jsons = copy.deepcopy(dict_in['jsons'])
    features = ['1xX', '1xY', '1xZ', '2xX', '2xY', '2xZ', '3xX', '3xY',
                '3xZ', '4xX', '4xY', '4xZ', 'RMSX', 'RMSY', 'RMSZ', 'speed']
    list_dataset = []
    list_dates = []
    for json in jsons:
        date = json.pop('occurredAt')
        list_dataset.append(json)
        list_dates.append(date)
    dates_raw = np.array(list_dates)
    dataframe_raw = pd.DataFrame(list_dataset, index=dates_raw)
    dataframe_raw = dataframe_raw[features]
    print("dataframe length: {} x {}".format(dataframe_raw.shape[0],
                                             dataframe_raw.shape[1]))
    dict_in['dataframe_raw'] = dataframe_raw
    dict_in['dates_raw'] = dates_raw


def remove_outliers(dict_in):
    """ Detect and remove outliers from the training dataset. """
    print("Removing outliers...")
    x = dict_in['dataframe_raw']
    dates_raw = dict_in['dates_raw']
    x_norm = RobustScaler().fit_transform(x)
    dict_in['min_samples'] = int(round(len(x) * 0.5 * math.log(2, len(x))))
    _get_samples_distance(dict_in, x_norm)
    _get_slope_angle(dict_in)
    _find_outliers(dict_in, x_norm)
    predict_outliers = dict_in['predict_outliers']
    dict_in['dataframe_cleaned'] = x[predict_outliers != -1]
    dict_in['dates_cleaned'] = dates_raw[predict_outliers != -1]


def _get_samples_distance(dict_in, x_norm):
    """ Calculate the distance between every sample. """
    min_samples = dict_in['min_samples']
    samples_distance = np.zeros((len(x_norm), len(x_norm)))
    for i in range(len(x_norm)):
        for j in range(len(x_norm)):
            samples_distance[i, j] = cityblock(x_norm[i, :], x_norm[j, :])
    samples_distance.sort()
    samples_distance = samples_distance[:, range(min_samples + 1)]
    samples_distance = sum(samples_distance.transpose()) / min_samples
    samples_distance.sort()
    dict_in['samples_distance'] = samples_distance


def _get_slope_angle(dict_in):
    """ Calculate the slope of the curve of the distances between
        samples. """
    samples_distance = dict_in['samples_distance']
    min_samples = dict_in['min_samples']
    slope_angle = np.zeros(len(samples_distance)-1)
    eps_arg = []
    j = float('inf')
    for i in range(len(samples_distance)-1):
        j += 1
        slope_angle[i] = samples_distance[i+1] - samples_distance[i]
        if (slope_angle[i] >= .1 and j >= min_samples):
            j = 0
            eps_arg.append(i)
    if eps_arg == []:
        maximum = np.argsort(slope_angle)
        maximum_arg = maximum[-1]
        eps_arg.append(maximum_arg)
    eps_all = []
    for arg in eps_arg:
        eps_all.append(samples_distance[arg])
    dict_in['slope_angle'] = slope_angle
    dict_in['eps_all'] = eps_all


def _find_outliers(dict_in, x_norm):
    """ Find outliers in the dataset. """
    min_samples = dict_in['min_samples']
    eps_all = dict_in['eps_all']
    predict_outliers = np.zeros(len(x_norm))
    predict_outliers[:] = -1
    b = 0
    for eps in eps_all:
        db = DBSCAN(eps=eps, min_samples=min_samples,
                    metric='manhattan')
        a = db.fit_predict(x_norm)
        x_norm = x_norm[a == -1]
        predict_outliers[predict_outliers == -1] =\
            np.where(a == -1, -1, a + b)
        if sum(a >= 0) >= 1:
            b = np.unique(predict_outliers)[-1] + 1
            if b == 0:
                b = 0
        if len(x_norm) == 0:
            break
    dict_in['predict_outliers'] = predict_outliers


def learn_patterns(dict_in):
    """ Fit the training dataset to the Deep Belief Network algorithm to
        learn patterns. """
    print("Learning patterns...")
    dataframe_cleaned = dict_in['dataframe_cleaned']
    scaler_anomaly = MinMaxScaler(feature_range=(0, 1))
    x_norm = scaler_anomaly.fit_transform(dataframe_cleaned)
    np.random.shuffle(x_norm)
    dict_in['scaler_anomaly'] = scaler_anomaly
    _grid_search(dict_in, x_norm)


def _grid_search(dict_in, x_norm):
    """ Grid search to fid the best hyperparameters. """
    time_script = dict_in['time_script']
    time_max = dict_in['time_max']
#    hidden_layers = [[30], [30, 30], [100, 30, 100]]
#    lr_rbm = [.005, .01]
#    lr_dbn = [.5, .2]
#    n_epochs_rbm = [200]
#    n_epochs_dbn = [2000]
#    batch_size = [20]
#    dropout = [0]
    hidden_layers = [[30], [50], [50, 50]]
    lr_rbm = [.005]
    lr_dbn = [.5]
    n_epochs_rbm = [200]
    n_epochs_dbn = [2000]
    batch_size = [30]
    dropout = [0]
    patience_rbm = 10
    patience_dbn = 50
    delta_rbm = 0.01
    delta_dbn = 0.0001
    dict_in['total_models'] = (len(hidden_layers) * len(lr_rbm) * len(lr_dbn) *
                               len(n_epochs_rbm) * len(n_epochs_dbn) *
                               len(batch_size) * len(dropout))
    dict_in['results_patterns'] = []
    i = 0
    for layers in hidden_layers:
        for rbm in lr_rbm:
            for dbn in lr_dbn:
                for epochs_rbm in n_epochs_rbm:
                    for epochs_dbn in n_epochs_dbn:
                        for batch in batch_size:
                            for drop in dropout:
                                i += 1
                                if time.time()-time_script > time_max:
                                    return
                                _fit_to_model(dict_in=dict_in,
                                              iteration=i,
                                              x_norm=x_norm,
                                              patience_rbm=patience_rbm,
                                              patience_dbn=patience_dbn,
                                              delta_rbm=delta_rbm,
                                              delta_dbn=delta_dbn,
                                              hidden_layers_structure=layers,
                                              learning_rate_rbm=rbm,
                                              learning_rate=dbn,
                                              n_epochs_rbm=epochs_rbm,
                                              n_iter_backprop=epochs_dbn,
                                              batch_size=batch,
                                              dropout_p=drop)


def _fit_to_model(dict_in, x_norm, patience_rbm, patience_dbn, delta_rbm,
                  delta_dbn, hidden_layers_structure, learning_rate_rbm,
                  learning_rate, n_epochs_rbm, n_iter_backprop,
                  batch_size, dropout_p, iteration):
    """ Fit the dataset to the Deep Belief Network algorithm. """
    time_script = dict_in['time_script']
    time_max = dict_in['time_max']
    total_models = dict_in['total_models']
    scaler_anomaly = dict_in['scaler_anomaly']
    dataframe_raw = dict_in['dataframe_raw']
    x_norm_all = scaler_anomaly.transform(dataframe_raw)
    time_start = time.time()
    max_accuracy = float('inf')
    mse = 0
    n_cv = 1
    shape = int(len(x_norm)/5) if n_cv == 1 else int(len(x_norm)/n_cv)
    for k in range(n_cv):
        start = k * shape
        end = (k+1) * shape
        delete = np.arange(start, end, 1)
        x_train = np.delete(x_norm, delete, axis=0)
        model = DBNr(activation_function='relu',
                     verbose=False,
                     early_stop=True,
                     patience_rbm=patience_rbm,
                     patience_dbn=patience_dbn,
                     delta_rbm=delta_rbm,
                     delta_dbn=delta_dbn,
                     hidden_layers_structure=hidden_layers_structure,
                     learning_rate_rbm=learning_rate_rbm,
                     learning_rate=learning_rate,
                     n_epochs_rbm=n_epochs_rbm,
                     n_iter_backprop=n_iter_backprop,
                     batch_size=batch_size,
                     dropout_p=dropout_p,
                     time_start=time_script,
                     time_elapsed_max=time_max)
        model.fit(x_train, x_train)
        mse += ((x_norm - model.predict(x_norm)) ** 2).mean(axis=1)
    mse = mse / n_cv
    if mse.max() < max_accuracy:
        max_accuracy = mse.max()
        mse_model = ((x_norm_all -
                      model.predict(x_norm_all)) **
                     2).mean(axis=1)
        dict_in['mse'] = mse_model
        dict_in['model_anomaly'] = model
        dict_in['threshold'] = mse.max() * 2
    time_elapsed = time.time() - time_start
    dict_in['results_patterns'].append({'hidden_layers':
                                        hidden_layers_structure,
                                        'lr_rbm': learning_rate_rbm,
                                        'lr_dbn': learning_rate,
                                        'n_epochs_rbm': n_epochs_rbm,
                                        'n_epochs_dbn': n_iter_backprop,
                                        'batch_size': batch_size,
                                        'dropout': dropout_p,
                                        'accuracy': mse.max(),
                                        'time': time_elapsed})
    print("  - Model {}/{} - {:.2f}s"
          .format(iteration, total_models, time_elapsed))


def optimal_clusters(dict_in):
    """ Find the best number of clusters and its normalizations. """
    print("Obtaining clusters...")
    dataframe_raw = dict_in['dataframe_raw']
    dataframe_cleaned = dict_in['dataframe_cleaned']
    min_samples = dict_in['min_samples']
    eps_all = dict_in['eps_all']
    scaler_anomaly = dict_in['scaler_anomaly']
    rob = RobustScaler()
    x_rob = rob.fit_transform(dataframe_cleaned)
    for eps in eps_all:
        db = DBSCAN(eps=eps, min_samples=min_samples,
                    metric='manhattan')
        predict_dbscan = db.fit_predict(x_rob)
        if np.unique(predict_dbscan)[0] != -1:
            break
    x_norm_cleaned = scaler_anomaly.transform(dataframe_cleaned)
    kmeans = KMeans(n_clusters=len(np.unique(predict_dbscan)),
                    random_state=93).fit(x_norm_cleaned)
    x_norm_raw = scaler_anomaly.transform(dataframe_raw)
    predict_patterns = kmeans.predict(x_norm_raw)
    for pattern in predict_patterns:
        key = 'scalerCluster{}'.format(pattern)
        group = dataframe_raw[predict_patterns == pattern]
        dict_in[key] = group.mean(axis=0)
    dict_in['model_patterns'] = kmeans
    dict_in['predict_patterns'] = predict_patterns
    dict_in['n_patterns'] = len(np.unique(predict_patterns))


def json_out(dict_in):
    """ Return the important parameters of the class as a dictionary. """
    keys = ['minSamples', 'samplesDistance', 'epsAll',
            'slopeAngle', 'predictOutlier', 'threshold', 'mse',
            'predictPattern', 'scalerAnomaly', 'modelAnomaly',
            'modelPattern', 'nClusters', 'resultsPattern']
    params = ['min_samples', 'samples_distance', 'eps_all',
              'slope_angle', 'predict_outliers', 'threshold', 'mse',
              'predict_patterns', 'scaler_anomaly', 'model_anomaly',
              'model_patterns', 'n_patterns', 'results_patterns']
    dict_out = {}
    for key, param in zip(keys, params):
        if param == 'model_anomaly':
            dict_out[key] = copy.deepcopy(dict_in[param].to_dict())
        else:
            dict_out[key] = copy.deepcopy(dict_in[param])
    return dict_out


def main_patterns(dict_in):
    """ Main function for web implementation. """
    try:
        assert 'jsons' in dict_in, "'jsons' key not found"
        dict_in['time_script'] = time.time()
        dict_in['time_max'] = 480
        create_dataframe(dict_in)
        remove_outliers(dict_in)
        learn_patterns(dict_in)
        optimal_clusters(dict_in)
        dict_out = json_out(dict_in)
    except Exception as error:
        print(str(error))
        dict_out = {"error": str(error)}
    return dict_out


if __name__ == "__main__":
    # For local tests only -------------------------------------------------- #
    import pickle
    dict_in = {}
    PATH = ("/Users/luizmanke/Google Drive/WEG/WMO/Machine Learning" +
            "/Projects/Motor Scan/Codes/")
    dict_in = pickle.load(open(PATH + "/dict_in_jsons.pkl", 'rb'))
    # ----------------------------------------------------------------------- #

    dict_out = main_patterns(dict_in)

    # For local tests only -------------------------------------------------- #
    dict_out['jsons'] =\
        pickle.load(open(PATH + "/dict_in_jsons.pkl", 'rb'))['jsons']
    train_samples = []
    for json in dict_out['jsons']:
        train_samples.append(json['occurredAt'])
    dict_out['trainSamples'] = train_samples
    dict_out['mse'] = pd.DataFrame(dict_out['mse'], index=train_samples)
    dict_out['predictPattern'] =\
        pd.DataFrame(dict_out['predictPattern'], index=train_samples)
    dict_out['plot_off'] = True
    import RetrieveResults_v0 as results
    results.plot_distances(dict_out)
    results.plot_outliers_2d(dict_out)
    results.plot_outliers_tl(dict_out)
    results.plot_anomaly_2d(dict_out)
    results.plot_anomaly_tl(dict_out)
    results.plot_mse_tl(dict_out)
    results.plot_pattern(dict_out)
    # ----------------------------------------------------------------------- #
