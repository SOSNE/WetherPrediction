import tensorflow as tf
from tools.load_data import get_data

columns = ['date', 'tavg', 'tmin', 'tmax', 'prcp', 'snow',
           'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco']

_input, target = get_data(path="../data/12375.csv", columns=columns)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7, 9)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(9)
])

model.compile(optimizer='adam', loss='mse')

model.fit(_input, target, epochs=1000)

model.save("models/weather_model_1.h5")
