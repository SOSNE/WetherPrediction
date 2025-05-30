import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_data(data):
    _input, target = [], []
    print(len(data))
    for i in range(len(data)-7):
        _input.append(data[i:i+7][:, 1:])
        target.append(data[i+7][1:])
    print(target[0])
    return np.array(_input, dtype=np.float64), np.array(target, dtype=np.float64)


def scale_data(_input, target):
    nsamples, ntimesteps, nfeatures = _input.shape
    scaler = StandardScaler()
    _input = scaler.fit_transform(_input.reshape(-1, nfeatures))
    _input = _input.reshape(nsamples, ntimesteps, nfeatures)
    target = scaler.fit_transform(target)
    joblib.dump(scaler, "models/weather_model_1_scaler.pkl")

    return _input, target

def get_data(path = "data/12375.csv", columns=['date', 'tavg'],time_star = "2020-05-15", time_stop = "2025-05-20"):

    df = pd.read_csv(path, header=None, names=columns)
    df = df.drop(columns=['tsun', 'coco'])

    df = df[(df['date'] >= time_star) & (df['date'] <= time_stop)]
    df['date'] = pd.to_datetime(df['date'])
    df['snow'] = df['snow'].fillna(0)
    df['prcp'] = df['prcp'].fillna(0)
    df['wpgt'] = df['wpgt'].ffill()

    _input, target = prepare_data(df.to_numpy())
    _input, target = scale_data(_input, target)
    return _input, target
