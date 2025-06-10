import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_data(data, for_training=False):
    _input, target, dates = [], [], []
    for i in range(len(data)-360):
        if for_training:
            target.append(data[i+360][1:])
            _input.append(data[i:i + 360][:, 1:])
        else:
            _input.append(data[i+1:i + 361][:, 1:])
            dates.append(data[i+1:i+361][:, :1])

    return np.array(_input, dtype=np.float64), np.array(target, dtype=np.float64), dates


def scale_data(_input, target, i_scaler = None ,t_scaler = None, for_training = False):
    nsamples, ntimesteps, nfeatures = _input.shape
    if i_scaler is None and t_scaler is None:
        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
    else:
        input_scaler = i_scaler
        target_scaler = t_scaler
    _input = input_scaler.fit_transform(_input.reshape(-1, nfeatures))
    _input = _input.reshape(nsamples, ntimesteps, nfeatures)

    if for_training:
        target = target_scaler.fit_transform(target)
        joblib.dump(input_scaler, "../models/weather_model_1_input_scaler.pkl")
        joblib.dump(target_scaler, "../models/weather_model_1_target_scaler.pkl")

    return _input, target

def get_data(path = "data/12375.csv", columns=['date', 'tavg'],time_star = "2020-05-15", time_stop = "2025-05-20", i_scaler = None ,t_scaler = None, for_training = False):

    df = pd.read_csv(path, header=None, names=columns)
    df = df.drop(columns=['tsun', 'coco'])

    df = df[(df['date'] >= time_star) & (df['date'] <= time_stop)]
    df['date'] = pd.to_datetime(df['date'])
    df['snow'] = df['snow'].fillna(0)
    df['prcp'] = df['prcp'].fillna(0)
    df['wpgt'] = df['wpgt'].ffill()
    df['wpgt'] = df['wpgt'].backfill()

    _input, target, dates = prepare_data(df.to_numpy(), for_training)
    _input, target = scale_data(_input, target, i_scaler, t_scaler, for_training)
    return _input, target, dates
