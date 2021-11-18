import pandas as pd
import numpy as np

def smape(y_true, y_pred):
  numerator = np.abs(y_true - y_pred)
  denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
  ratio = numerator / denominator
  return (ratio.mean())


def SES_model(data, horizon, alpha_high, alpha_low):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    ses_high = SimpleExpSmoothing(data['High'], initialization_method='legacy-heuristic')
    res_high = ses_high.fit(smoothing_level=alpha_high, optimized=False)
    fore_high = res_high.forecast(horizon)
    fore_high = fore_high.to_frame()
    fore_high.columns = ['Forecast_High']
    pred_high = res_high.predict(start=data.index[0], end=data.index[-1])
    smap_high = round(smape(data['High'], pred_high),3)
    #pred_high = pred_high.to_frame()
    #pred_high.columns = ['Pred_High']

    ses_low = SimpleExpSmoothing(data['Low'], initialization_method='legacy-heuristic')
    res_low = ses_low.fit(smoothing_level=alpha_low, optimized=False)
    fore_low = res_low.forecast(horizon)
    fore_low = fore_low.to_frame()
    fore_low.columns = ['Forecast_Low']
    pred_low = res_low.predict(start=data.index[0], end=data.index[-1])
    smap_low = round(smape(data['Low'], pred_low),3)
    #pred_low = pred_low.to_frame()
    #pred_low.columns = ['Pred_Low']

    data_final = pd.concat([data,pred_low,pred_high, fore_high, fore_low], axis=1)
    data_final.loc[data.index[-1], 'Forecast_High'] = data_final.loc[data.index[-1], 'High']
    data_final.loc[data.index[-1], 'Forecast_Low'] = data_final.loc[data.index[-1], 'Low']
    optim_alpha_high = round(ses_high.fit().params['smoothing_level'],2)
    optim_alpha_low = round(ses_low.fit().params['smoothing_level'],2)
    return [data_final,smap_low,smap_high,optim_alpha_high,optim_alpha_low]


def Holt_model(data,horizon, level_high, level_low,trend_high,trend_low):
    from statsmodels.tsa.holtwinters import Holt
    holt_high = Holt(data['High'], initialization_method='legacy-heuristic')
    res_high = holt_high.fit(smoothing_level=level_high,smoothing_trend= trend_high,optimized=False)
    fore_high = res_high.forecast(horizon)
    fore_high = fore_high.to_frame()
    fore_high.columns = ['Forecast_High']
    pred_high = res_high.predict(start=data.index[0], end=data.index[-1])
    smap_high = round(smape(data['High'], pred_high), 3)

    holt_low = Holt(data['Low'], initialization_method='legacy-heuristic')
    res_low = holt_low.fit(smoothing_level= level_low,smoothing_trend=  trend_low, optimized=False)
    fore_low = res_low.forecast(horizon)
    fore_low = fore_low.to_frame()
    fore_low.columns = ['Forecast_Low']
    pred_low = res_low.predict(start=data.index[0], end=data.index[-1])
    smap_low = round(smape(data['Low'], pred_low), 3)

    data_final = pd.concat([data,pred_low,pred_high, fore_high, fore_low], axis=1)
    data_final.loc[data.index[-1], 'Forecast_High'] = data_final.loc[data.index[-1], 'High']
    data_final.loc[data.index[-1], 'Forecast_Low'] = data_final.loc[data.index[-1], 'Low']
    optim_level_high = round(holt_high.fit().params['smoothing_level'],2)
    optim_level_low = round(holt_low.fit().params['smoothing_level'],2)
    optim_trend_high = round(holt_high.fit().params['smoothing_trend'],2)
    optim_trend_low = round(holt_low.fit().params['smoothing_trend'],2)
    return [data_final,smap_low,smap_high,optim_level_high,optim_level_low,optim_trend_high,optim_trend_low]



def Holt_Winter_Model(data,horizon, level_high, level_low,trend_high,trend_low,season_high,season_low):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    hw_high =ExponentialSmoothing(data['High'], initialization_method='legacy-heuristic',trend = 'add',seasonal='add')
    res_high = hw_high.fit(smoothing_level=level_high, smoothing_trend=trend_high, smoothing_seasonal= season_high,optimized=False)
    fore_high = res_high.forecast(horizon)
    fore_high = fore_high.to_frame()
    fore_high.columns = ['Forecast_High']
    pred_high = res_high.predict(start=data.index[0], end=data.index[-1])
    smap_high = round(smape(data['High'], pred_high), 3)

    hw_low = ExponentialSmoothing(data['Low'], initialization_method='legacy-heuristic',trend = 'add',seasonal='add')
    res_low = hw_low.fit(smoothing_level=level_low, smoothing_trend= trend_low, smoothing_seasonal= season_low ,optimized=False)
    fore_low = res_low.forecast(horizon)
    fore_low = fore_low.to_frame()
    fore_low.columns = ['Forecast_Low']
    pred_low = res_low.predict(start=data.index[0], end=data.index[-1])
    smap_low = round(smape(data['Low'], pred_low), 3)

    data_final = pd.concat([data, pred_low, pred_high, fore_high, fore_low], axis=1)
    data_final.loc[data.index[-1], 'Forecast_High'] = data_final.loc[data.index[-1], 'High']
    data_final.loc[data.index[-1], 'Forecast_Low'] = data_final.loc[data.index[-1], 'Low']
    optim_model_high = hw_high.fit()
    optim_model_low = hw_low.fit()
    optim_level_high = round(optim_model_high.params['smoothing_level'], 2)
    optim_level_low = round(optim_model_low.params['smoothing_level'], 2)
    optim_trend_high = round(optim_model_high.params['smoothing_trend'], 2)
    optim_trend_low = round(optim_model_low.params['smoothing_trend'], 2)
    optim_season_high =  round(optim_model_high.params['smoothing_seasonal'],2)
    optim_season_low = round(optim_model_low.params['smoothing_seasonal'], 2)
    return [data_final, smap_low, smap_high, optim_level_high, optim_level_low, optim_trend_high, optim_trend_low,optim_season_high,optim_season_low]

from preprocess import process_high, process_low

def AR_model(data,horizon, p_high,p_low):
    from statsmodels.tsa.arima.model import ARIMA

    ar_high = ARIMA(data['High'],order = (p_high,0,0))
    res_high = ar_high.fit()
    fore_high = res_high.forecast(horizon)
    fore_high = fore_high.to_frame()
    fore_high.columns = ['Forecast_High']
    pred_high = res_high.predict(start=data.index[0], end=data.index[-1])
    smap_high = round(smape(data['High'], pred_high), 3)

    ar_low = ARIMA(data['Low'],order = (p_low,0,0))
    res_low = ar_low.fit()
    fore_low = res_low.forecast(horizon)
    fore_low = fore_low.to_frame()
    fore_low.columns = ['Forecast_Low']
    pred_low = res_low.predict(start=data.index[0], end=data.index[-1])
    smap_low = round(smape(data['Low'], pred_low), 3)

    data_final = pd.concat([data, pred_low, pred_high, fore_high, fore_low], axis=1)
    data_final.loc[data.index[-1], 'Forecast_High'] = data_final.loc[data.index[-1], 'High']
    data_final.loc[data.index[-1], 'Forecast_Low'] = data_final.loc[data.index[-1], 'Low']

    return [data_final,smap_high,smap_low]

def MA_model(data,horizon, q_high,q_low):
    from statsmodels.tsa.arima.model import ARIMA

    ma_high = ARIMA(data['High'],order = (0,0,q_high))
    res_high = ma_high.fit()
    fore_high = res_high.forecast(horizon)
    fore_high = fore_high.to_frame()
    fore_high.columns = ['Forecast_High']
    pred_high = res_high.predict(start=data.index[0], end=data.index[-1])
    smap_high = round(smape(data['High'], pred_high), 3)

    ma_low = ARIMA(data['Low'],order = (0,0,q_low))
    res_low = ma_low.fit()
    fore_low = res_low.forecast(horizon)
    fore_low = fore_low.to_frame()
    fore_low.columns = ['Forecast_Low']
    pred_low = res_low.predict(start=data.index[0], end=data.index[-1])
    smap_low = round(smape(data['Low'], pred_low), 3)

    data_final = pd.concat([data, pred_low, pred_high, fore_high, fore_low], axis=1)
    data_final.loc[data.index[-1], 'Forecast_High'] = data_final.loc[data.index[-1], 'High']
    data_final.loc[data.index[-1], 'Forecast_Low'] = data_final.loc[data.index[-1], 'Low']

    return [data_final,smap_high,smap_low]

def ARMA_model(data,horizon,p_high,p_low, q_high,q_low):
    from statsmodels.tsa.arima.model import ARIMA

    arma_high = ARIMA(data['High'],order = (p_high,0,q_high))
    res_high = arma_high.fit()
    fore_high = res_high.forecast(horizon)
    fore_high = fore_high.to_frame()
    fore_high.columns = ['Forecast_High']
    pred_high = res_high.predict(start=data.index[0], end=data.index[-1])
    smap_high = round(smape(data['High'], pred_high), 3)

    arma_low = ARIMA(data['Low'],order = (p_low,0,q_low))
    res_low = arma_low.fit()
    fore_low = res_low.forecast(horizon)
    fore_low = fore_low.to_frame()
    fore_low.columns = ['Forecast_Low']
    pred_low = res_low.predict(start=data.index[0], end=data.index[-1])
    smap_low = round(smape(data['Low'], pred_low), 3)

    data_final = pd.concat([data, pred_low, pred_high, fore_high, fore_low], axis=1)
    data_final.loc[data.index[-1], 'Forecast_High'] = data_final.loc[data.index[-1], 'High']
    data_final.loc[data.index[-1], 'Forecast_Low'] = data_final.loc[data.index[-1], 'Low']

    return [data_final,smap_high,smap_low]

def ARIMA_model(data,horizon,p_high,p_low,q_high,q_low,i_high,i_low):
    from statsmodels.tsa.arima.model import ARIMA

    arima_high = ARIMA(data['High'], order=(p_high, i_high, q_high))
    res_high = arima_high.fit()
    fore_high = res_high.forecast(horizon)
    fore_high = fore_high.to_frame()
    fore_high.columns = ['Forecast_High']
    pred_high = res_high.predict(start=data.index[0], end=data.index[-1])
    smap_high = round(smape(data['High'], pred_high), 3)

    arima_low = ARIMA(data['Low'], order=(p_low, i_low, q_low))
    res_low = arima_low.fit()
    fore_low = res_low.forecast(horizon)
    fore_low = fore_low.to_frame()
    fore_low.columns = ['Forecast_Low']
    pred_low = res_low.predict(start=data.index[0], end=data.index[-1])
    smap_low = round(smape(data['Low'], pred_low), 3)

    data_final = pd.concat([data, pred_low, pred_high, fore_high, fore_low], axis=1)
    data_final.loc[data.index[-1], 'Forecast_High'] = data_final.loc[data.index[-1], 'High']
    data_final.loc[data.index[-1], 'Forecast_Low'] = data_final.loc[data.index[-1], 'Low']

    return [data_final, smap_high, smap_low]

def Auto_Arima(data,horizon):
    from pmdarima import auto_arima
    index = pd.bdate_range(start=data.index[-1], periods=(horizon+1))
    model_high = auto_arima(data['High'])
    fore_high = model_high.predict(horizon)
    fore_high = np.insert(fore_high, 0, data['High'][-1])
    fore_high = pd.DataFrame(fore_high, index=index)
    fore_high.columns = ['Forecast_High']

    model_low = auto_arima(data['Low'])
    fore_low = model_low.predict(horizon)
    fore_low = np.insert(fore_low, 0, data['Low'][-1])
    fore_low = pd.DataFrame(fore_low, index=index)
    fore_low.columns = ['Forecast_Low']

    data_final = pd.concat([data,fore_high,fore_low],axis = 1)
    return data_final