import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


columns = ['date', 'tavg', 'tmin', 'tmax', 'prcp', 'snow',
           'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco']

df = pd.read_csv('data/12375.csv', header=None, names=columns)
df = df.drop(columns=['tsun', 'coco'])

df = df[(df['date'] >= "2020-05-15") & (df['date'] <= "2025-05-20")]
df['date'] = pd.to_datetime(df['date'])
df['snow'] = df['snow'].fillna(0)
df['prcp'] = df['prcp'].fillna(0)
df['wpgt'] = df['wpgt'].ffill()


def prepare_data(data):
    _input, target = [], []
    for i in range(len(data)-7):
        _input.append(data[i:i+7][:, 1:])
        target.append(data[i+7][1:])
    return np.array(_input, dtype=np.float64), np.array(target, dtype=np.float64)

_input, target = prepare_data(df.to_numpy())

nsamples, ntimesteps, nfeatures = _input.shape
scaler = StandardScaler()
_input = scaler.fit_transform(_input.reshape(-1, nfeatures))
_input = _input.reshape(nsamples, ntimesteps, nfeatures)

target = scaler.fit_transform(target)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7, 9)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(9)
])

model.compile(optimizer='adam', loss='mse')

model.fit(_input, target, epochs=10)

model.save("models/weather_model_1.h5")
