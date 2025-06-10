# WeatherPrediction

WeatherPrediction is a simple set of tools for predicting weather using machine learning.  
It uses TensorFlow, pandas, joblib, NumPy, and scikit-learn.

By default, the project is configured to work with the [Meteostat](https://meteostat.net/en/) database.

## Installation

Make sure you have the required packages installed:

```bash
pip install tensorflow pandas numpy scikit-learn joblib
```

 ## Setup

1. Open `load_data.py` and set the filenames for the scalers:

 ```python
 joblib.dump(input_scaler, "../models/weather_model_4_input_scaler.pkl")
 joblib.dump(target_scaler, "../models/weather_model_4_target_scaler.pkl")
 ```
 2. Open `model_training.py` and set the training data path="../data/12375.csv"
 ```python
 _input, target, dates = get_data(path="../data/12375.csv", columns=columns,
                          time_star = "2018-05-21", time_stop = "2025-05-31", for_training=True)
 ```  
 
 3. Open `model_training.py` and set the model save path:

 ```python
 model.save("../models/weather_model_4.h5")
 ```

 4. Set the number of training epochs and start training:

 ```python
model.fit(_input, target, epochs=1000, validation_split=0.2)
```

You can also modify the model structure, but remember to update the corresponding parts in `load_data.py`.

 ## Usage

After training the model, you can make predictions by running `main.py`.

 1. Load the model and scalers:

 ```python
 model = load_model("models/weather_model_3.h5")
 scaler_input = joblib.load("models/weather_model_3_input_scaler.pkl")
 scaler_target = joblib.load("models/weather_model_3_target_scaler.pkl")
 ```

 2. Load your input data (make sure the date range and format are correct and match `load_data.py` settings):

 ```python
 _input, target, dates = get_data(
    path="data/Libertow Weather History.csv",
    columns=columns,
    time_star="2024-06-13",
    time_stop="2025-06-10",
    i_scaler=scaler_input,
    t_scaler=scaler_target
)
 ```

Ensure your data has at least **361 days** (default requirement).

 4. Run `main.py` to get your weather predictions.

 ## License

MIT
