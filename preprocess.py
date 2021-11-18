import pandas as pd
import datetime
import numpy as np
def smape(y_true, y_pred):
  numerator = np.abs(y_true - y_pred)
  denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
  ratio = numerator / denominator
  return (ratio.mean())

def preprocessing(data,interval):
    if interval=='1d':
        data = data.asfreq('B') #set frequency as Business Day
        data.ffill(inplace=True)

    elif interval == '1wk':
        data.dropna(inplace = True) # weekly/monthly data has days on which dividends are paid, while all other values as NA
        if int(str((data.index[-1]-data.index[-2]))[0])<7: #if the last data point is of today and today is not monday
            data = data.iloc[:-1,]
        data = data.asfreq('W-MON') #week  monday
    elif interval == '1mo':
        data.dropna(inplace=True)
        if data.index[-1].day!=1: #if the last data point is of today not the first of month
            data = data.iloc[:-1,]
        data = data.asfreq('MS') #month start
    elif interval == '3mo':
        data.dropna(inplace=True)
        if data.index[-2].month-data.index[-1].month <3:  # if the last data point is of today not the first of month
            data = data.iloc[:-1,]
        freq = 'QS-'+data.index[-1].month_name()[0:3].upper()
        data = data.asfreq(freq)  # quarter start
    return data

#def seasonal(interval,period):
 #   if interval== '1d':
  #      if period=='1mo' or period=='3mo' or period=='6mo':
      #      season = 7

def process_high(data,res_high ,fore_high):
    fore_high = fore_high.to_frame()
    fore_high.columns = ['Forecast_High']
    pred_high = res_high.predict(start=data.index[0], end=data.index[-1])
    smap_high = round(smape(data['High'], pred_high), 3)
    return [fore_high,pred_high,smap_high]

def process_low(data,res_low ,fore_low):
    fore_low = fore_low.to_frame()
    fore_low.columns = ['Forecast_Low']
    pred_low = res_low.predict(start=data.index[0], end=data.index[-1])
    smap_low = round(smape(data['Low'], pred_low), 3)
    return [fore_low,pred_low,smap_low]