# System Libraries
import jsonpickle
import numpy as np
import pandas as pd
import pickle
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow import set_random_seed

np.random.seed(93)
set_random_seed(93)


def main(dict_in):
    """Main function for web implementation."""
    print(dict_in)
    try:
        for key in ["dates", "mse", "threshold", "modelTimeSeriesExists"]:
            assert key in dict_in, "%s not found in dictionary" % key
        if dict_in["modelTimeSeriesExists"]:
            for key in ["modelTimeSeries", "scalerTimeSeries"]:
                assert key in dict_in, "%s not found in dictionary" % key
                dict_in[key] = pickle.loads(jsonpickle.decode(dict_in[key]))

        dict_in["threshold"] = \
            pickle.loads(jsonpickle.decode(dict_in["threshold"]))
        dict_in["lookBack"] = 100
        dict_in["meanSize"] = 10
        dict_in["steps"] = 200

        create_dataframe(dict_in)
        preprocess(dict_in)
        train_series(dict_in)
        forecast(dict_in)
        inverse_preprocess(dict_in)
        select_keys(dict_in)

        return dict_in

    except Exception as err:
        print(str(err))
        return {"error": str(err)}


def create_dataframe(dict_in):
    """Create a dataframe with the samples mse and dates."""
    print("Creating dataframe...")
    df = pd.DataFrame(dict_in["mse"], index=dict_in["dates"], columns=["mse"])
    df.sort_index(inplace=True)
    dict_in["df"] = df


def preprocess(dict_in):
    """Scale, reshape and split the mean squared error."""
    print("Preprocessing data...")
    df = dict_in["df"]
    look_back = dict_in["lookBack"]
    is_trained = dict_in["modelTimeSeriesExists"]
    mean_size = dict_in["meanSize"]
    n_out = 1
    mean = df.rolling(mean_size).mean()
    if not is_trained:
        scaler_time_series = MinMaxScaler(feature_range=(-1, 1))
        mean_norm = scaler_time_series.fit_transform(mean)
        dict_in["scalerTimeSeries"] = scaler_time_series
    else:
        scaler_time_series = dict_in["scalerTimeSeries"]
        mean_norm = scaler_time_series.transform(mean)
    mean_norm = pd.DataFrame(mean_norm, index=mean.index,
                             columns=["mean_norm"])
    cols, names = [], []
    for i in range(look_back, -n_out, -1):
        cols.append(mean_norm.shift(i))
        names.append('t-%d' % i)
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    mean_x = agg.iloc[:, :-n_out]
    mean_y = agg.iloc[:, -n_out:]
    dict_in["mean_x"] = mean_x.values.reshape((len(mean_x), 1, look_back))
    dict_in["mean_y"] = mean_y.values
    dict_in["mean"] = mean.iloc[:, 0].values.tolist()


def train_series(dict_in):
    """Learn or improve time series model."""
    print("Training time series...")
    is_trained = dict_in["modelTimeSeriesExists"]
    n_epochs = 1000
    patience = 100
    if is_trained:
        update_model(dict_in, n_epochs, patience)
    else:
        new_model(dict_in, n_epochs, patience)


def new_model(dict_in, n_epochs, patience):
    """Learn a new time series model."""
    print("Learning new model...")
    mean_x = dict_in["mean_x"]
    mean_y = dict_in["mean_y"]
    look_back = dict_in["lookBack"]
    checkpoint = float("inf")
    batch_size = [100]
    lstm_layers = [[200, 100]]
    results = {}
    i = 0
    for batch in batch_size:
        for layers in lstm_layers:
            model = Sequential()
            for i, unit in enumerate(layers):
                if i == 0 and len(layers) > 1:
                    model.add(LSTM(units=unit, return_sequences=True,
                                   input_shape=(1, look_back)))
                elif i == 0:
                    model.add(LSTM(units=unit, input_shape=(1, look_back)))
                elif i + 1 == len(layers):
                    model.add(LSTM(units=unit))
                else:
                    model.add(LSTM(units=unit, return_sequences=True))
            model.add(Dense(units=1, activation="linear"))
            model.compile(optimizer="adam", loss="mean_squared_error")
            train_size = int(len(mean_x) * 0.8)
            x_train = mean_x[0:train_size]
            y_train = mean_y[0:train_size]
            x_test = mean_x[train_size:]
            y_test = mean_y[train_size:]
            es = EarlyStopping(restore_best_weights=True, min_delta=0,
                               patience=patience, monitor="val_loss")
            history = model.fit(x_train, y_train, epochs=n_epochs,
                                batch_size=batch, verbose=1,
                                validation_data=(x_test, y_test),
                                callbacks=[es])
            if min(history.history["val_loss"]) < checkpoint:
                checkpoint = min(history.history["val_loss"])
                model.fit(mean_x, mean_y, epochs=n_epochs//100,
                          batch_size=batch, verbose=1)
                forecast_model_scaled = model.predict(mean_x, batch_size=1)
                dict_in["forecast_model_scaled"] = forecast_model_scaled
                dict_in["modelTimeSeries"] = model
            results[i] = {"batch_size": batch, "layers": layers,
                          "loss": history.history["loss"],
                          "val_loss": history.history["val_loss"]}
            i += 1
            dict_in["results"] = results


def update_model(dict_in, n_epochs, patience):
    """Update an existing time series model."""
    print("Updating model...")
    mean_x = dict_in["mean_x"]
    mean_y = dict_in["mean_y"]
    model = dict_in["modelTimeSeries"]
    model.compile(optimizer="adam", loss="mean_squared_error")
    batch_size = len(mean_x) // 10
    es = EarlyStopping(restore_best_weights=True, min_delta=0,
                       patience=patience, monitor="loss")
    history = model.fit(mean_x, mean_y, epochs=n_epochs//1000,
                        callbacks=[es], batch_size=batch_size, verbose=1)
    forecast_model_scaled = model.predict(mean_x, batch_size=1)
    results = {"loss": history.history["loss"]}
    dict_in["forecast_model_scaled"] = forecast_model_scaled
    dict_in["modelTimeSeries"] = model
    dict_in["results"] = results


def forecast(dict_in):
    """Make forecasts."""
    print("Forecasting...")
    model = dict_in["modelTimeSeries"]
    mean_x = dict_in["mean_x"]
    mean_y = dict_in["mean_y"]
    steps = dict_in["steps"]
    forecast = []
    x_forecast = mean_x[-1, :]
    x_forecast = np.delete(x_forecast, 0)
    x_forecast = np.append(x_forecast, mean_y[-1])
    x_forecast = x_forecast.reshape(1, 1, x_forecast.shape[0])
    for _ in range(steps):
        prediction = model.predict(x_forecast, batch_size=1)[0][0]
        forecast.append(prediction)
        x_forecast = np.delete(x_forecast, 0)
        x_forecast = np.append(x_forecast, prediction)
        x_forecast = x_forecast.reshape(1, 1, x_forecast.shape[0])
    dict_in["forecast_scaled"] = np.array(forecast).reshape(-1, 1)


def inverse_preprocess(dict_in):
    """Return forcast values to the original scale."""
    scaler = dict_in["scalerTimeSeries"]
    threshold = dict_in["threshold"]
    forecast_scaled = dict_in["forecast_scaled"]
    forecast_model_scaled = dict_in["forecast_model_scaled"]
    forecast = scaler.inverse_transform(forecast_scaled)
    forecast_model = scaler.inverse_transform(forecast_model_scaled)
    dict_in["forecast"] = forecast[:, 0].tolist()
    dict_in["forecastModel"] = forecast_model[:, 0].tolist()
    dict_in["futureAnomaly"] =\
        True if sum(np.array(forecast) > threshold) > 0 else False


def select_keys(dict_in):
    """Return only the selected keys"""
    keys = ["scalerTimeSeries", "results", "modelTimeSeries", "mse",
            "meanSize", "lookBack", "forecast", "forecastModel",
            "futureAnomaly", "dates", "mean", "deviceId"]
    jp_keys = ["scalerTimeSeries", "modelTimeSeries"]
    for key in jp_keys:
        if key in dict_in:
            dict_in[key] = jsonpickle.encode(pickle.dumps(dict_in[key]))
    pop_keys = []
    for key in list(dict_in):
        if key not in keys:
            pop_keys.append(key)
    for key in pop_keys:
        if key not in keys:
            dict_in.pop(key)
