from tensorflow.keras.models import load_model
from tools.load_data import get_data
import joblib
import numpy as np

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


_input, target = get_data(path="data/Weather Data 12566.csv", columns=columns, time_star = "2025-05-20", time_stop = "2025-05-31")
model = load_model("models/weather_model_1.h5")
scaler = joblib.load("models/weather_model_1_scaler.pkl")
original_target_data = scaler.inverse_transform([target[4]])

_input = _input[2]
_input = np.expand_dims(_input, axis=0)

prediction = model.predict(_input)

original_prediction_data = scaler.inverse_transform(prediction)

for i, label in enumerate(labels):
    print(f"{label}: {original_prediction_data[0][i]:.2f}")

