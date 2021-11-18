import numpy as np
import pandas as pd
from sktime.forecasting.compose import ReducedRegressionForecaster #Install sktime ver. 0.4.1 to ensure compatibility with the code.
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
#from xgboost import XGBRegressor

def select_regressor(selection):
    regressors = {
        'Linear Regression': LinearRegression(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
       # 'XGBoost': XGBRegressor(verbosity=0),
        'Support Vector Machines': LinearSVR(),
        'Extra Trees': ExtraTreesRegressor(),
    }

    return regressors[selection]


def forecast(data, horizon, model):
    high = data['High'].reset_index(drop=True)
    low = data['Low'].reset_index(drop=True)
    window_length = len(data['High']) - horizon
    fh = np.arange(horizon) + 1
    fl = np.arange(horizon) + 1
    regressor = select_regressor(model)
    index = pd.bdate_range(start=data.index[-1], periods=(horizon + 1))

    forecast_high = ReducedRegressionForecaster(regressor=regressor, window_length=window_length,
                                                strategy='recursive')
    forecast_high.fit(high, fh=fh)
    fore_high = forecast_high.predict(fh).to_numpy()
    # print(type(fore_high))
    fore_high = np.insert(fore_high, 0, data['High'][-1])
    # print(fore_high)
    fore_high = pd.DataFrame(fore_high, index=index)
    fore_high.columns = ['Forecast_High']

    forecast_low = ReducedRegressionForecaster(regressor=regressor, window_length=window_length,
                                               strategy='recursive')
    forecast_low.fit(low, fh=fl)
    fore_low = forecast_low.predict(fl).to_numpy()
    fore_low = np.insert(fore_low, 0, data['Low'][-1])
    fore_low = pd.DataFrame(fore_low, index=index)
    fore_low.columns = ['Forecast_Low']

    data_final = pd.concat([data, fore_high, fore_low], axis=1)
   # return data_final

    y_train, y_test = temporal_train_test_split(high, test_size=horizon)
    window_length = len(data['High'])-horizon-1
    fh = np.arange(y_test.shape[0]) + 1
    forecaster = ReducedRegressionForecaster(regressor=regressor, window_length=window_length,
                                             strategy='recursive')
    forecaster.fit(y_train, fh=fh)
    y_pred = forecaster.predict(fh)
    smape_high =  smape_loss(y_pred, y_test)

    y_train, y_test = temporal_train_test_split(low, test_size=horizon)
    fh = np.arange(y_test.shape[0]) + 1
    forecaster = ReducedRegressionForecaster(regressor=regressor, window_length=window_length,
                                             strategy='recursive')
    forecaster.fit(y_train, fh=fh)
    y_pred = forecaster.predict(fh)
    smape_low =  smape_loss(y_pred, y_test)

    return [data_final,smape_high ,smape_low]



