from tensorflow.keras.models import load_model
from tools.load_data import get_data
import joblib
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)

columns = ['date', 'tavg', 'tmin', 'tmax', 'prcp', 'snow',
           'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco']

labels = [
    "The average air temperature in °C",
    "The minimum air temperature in °C",
    "The maximum air temperature in °C",
    "The daily precipitation total in mm",
    "The maximum snow depth in mm",
    "The average wind direction in degrees (°)",
    "The average wind speed in km/h",
    "The peak wind gust in km/h",
    "The average sea-level air pressure in hPa"
]


model = load_model("models/weather_model_1.h5")
scaler_input = joblib.load("models/weather_model_1_input_scaler.pkl")
scaler_target = joblib.load("models/weather_model_1_target_scaler.pkl")

_input, target, dates = get_data(path="data/WeatherData.csv", columns=columns,
                          time_star = "2024-06-13", time_stop = "2025-06-10",
                          i_scaler=scaler_input, t_scaler=scaler_target)


_input = _input[-1]
_input = np.expand_dims(_input, axis=0)

prediction = model.predict(_input)


original_prediction_data = scaler_target.inverse_transform(prediction)

for i, label in enumerate(labels):
    print(f"{label}: {original_prediction_data[0][i]:.2f}")
print((dates[-1][-1]+ pd.Timedelta(days=1))[0])

