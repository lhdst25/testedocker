"""
PATTERN PREDICTION
"""

# System Libraries
import copy
import numpy as np
import pandas as pd

# Thid-party libraries
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
    dict_in['dataframe'] = dataframe_raw
    dict_in['dates'] = dates_raw


def predict_anomaly(dict_in):
    """Classify samples into normal and anomaly."""
    print("Predicting anomalies...")
    scaler_anomaly = dict_in['scalerAnomaly']
    threshold = dict_in['threshold']
    x = dict_in['dataframe']
    dict_model = copy.deepcopy(dict_in['modelAnomaly'])
    model = DBNr().from_dict(dict_model)
    x_norm = scaler_anomaly.transform(x)
    mse = ((x_norm - model.predict(x_norm)) ** 2).mean(axis=1)
    anomaly = np.zeros(mse.shape[0])
    anomaly[:] = np.where(mse > threshold, 1, 0)
    dict_in['mse'] = mse
    dict_in['anomaly'] = anomaly


def predict_pattern(dict_in):
    """Classify faulty samples."""
    print("Predicting patterns...")
    x = dict_in['dataframe']
    scaler_anomaly = dict_in['scalerAnomaly']
    model_pattern = dict_in['modelPattern']
    x_norm = scaler_anomaly.transform(x)
    y_pattern = model_pattern.predict(x_norm)
    dict_in['predictPattern'] = y_pattern


def select_keys(dict_in):
    """Return only the selected keys."""
    keys = ['anomaly', 'mse', 'predictPattern']
    dict_out = {}
    for key in keys:
        dict_out[key] = copy.deepcopy(dict_in[key])
    return dict_out


def main_predict_pattern(dict_in):
    """Main function for web implementation."""
    try:
        for key in ['jsons', 'threshold', 'scalerAnomaly', 'modelAnomaly',
                    'modelPattern']:
            assert key in dict_in, '%s not found in dictionary' % key
        create_dataframe(dict_in)
        predict_anomaly(dict_in)
        predict_pattern(dict_in)
        dict_out = select_keys(dict_in)
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
    dict_in = pickle.load(open(PATH + "/dict_out_pattern.pkl", 'rb'))
    dict_in['jsons'] =\
        pickle.load(open(PATH + "/dict_in_jsons.pkl", 'rb'))['jsons']

#    pickle.dump(dict_out, open(PATH + 'dict_out_predict_pattern.pkl', 'wb'))
    # ----------------------------------------------------------------------- #

    dict_out = main_predict_pattern(dict_in)

    # For local tests only -------------------------------------------------- #
    dict_out['jsons'] = copy.deepcopy(dict_in['jsons'])
    dict_out['predictOutlier'] = copy.deepcopy(dict_in['predictOutlier'])
    dict_out['threshold'] = copy.deepcopy(dict_in['threshold'])
    # remover train sample #
    train_samples = []
    for json in dict_out['jsons']:
        train_samples.append(json['occurredAt'])
    dict_out['trainSamples'] = train_samples
    dict_out['mse'] = pd.DataFrame(dict_out['mse'], index=train_samples)
    dict_out['predictPattern'] =\
        pd.DataFrame(dict_out['predictPattern'], index=train_samples)
    import RetrieveResults_v0 as results
    results.plot_anomaly_2d(dict_out)
    results.plot_anomaly_tl(dict_out)
    results.plot_mse_tl(dict_out)
    results.plot_pattern(dict_out)
    # ----------------------------------------------------------------------- #
