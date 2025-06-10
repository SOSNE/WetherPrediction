import tensorflow as tf
from tools.load_data import get_data

columns = ['date', 'tavg', 'tmin', 'tmax', 'prcp', 'snow',
           'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco']

_input, target, dates = get_data(path="../data/12375.csv", columns=columns,
                          time_star = "2018-05-21", time_stop = "2025-05-31", for_training=True)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(360, 9)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.14),
    tf.keras.layers.Dense(9)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(_input, target, epochs=1000, validation_split=0.2)

model.save("../models/weather_model_1.h5")
