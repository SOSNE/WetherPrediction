import pandas as pd
import numpy as np
import tensorflow as tf
from pyasn1_modules.rfc5755 import Target

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


model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(7, 9)),
  tf.keras.layers.LSTM(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(9)
])

model.compile(optimizer='adam', loss='mse')

model.fit(_input, target, epochs=1000)
