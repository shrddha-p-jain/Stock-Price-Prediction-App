import streamlit as st
import pandas as pd
import sqlite3 as sq
import datetime
import yfinance as yf
from preprocess import preprocessing
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

"# Stock Price Prediction"
"This is an app to predict the High and Low of the given Stock. You can select different stocks, intervals, periods from the sidebar. Feel free to experiment with different models"
"The Github repository of the app is available [here](https://github.com/shrddha-p-jain/Stock-Price-Prediction-App).Feel free to contact me on [LinkedIn](https://www.linkedin.com/in/shraddha-p-jain/)\
        or via [e-mail](mailto:shrddhapjain@gmail.com)."

db = sq.connect('stocks.db')

# get country
query = "SELECT DISTINCT(Country) FROM tkrinfo;"
country = pd.read_sql_query(query, db)
choice_country = st.sidebar.selectbox("Pick country", country)

# get exchange
query = "SELECT DISTINCT(Exchange) FROM tkrinfo WHERE Country = '" + choice_country + "'"
exchange = pd.read_sql_query(query, db)
choice_exchange = st.sidebar.selectbox("Pick exchange", exchange, index = 1)

# get stock name
query = "SELECT DISTINCT(Name) FROM tkrinfo WHERE Exchange = '" + choice_exchange + "'"
name = pd.read_sql_query(query, db)
choice_name = st.sidebar.selectbox("Pick the Stock", name)

# get stock tickr
query = "SELECT DISTINCT(Ticker) FROM tkrinfo WHERE Exchange = '" + choice_exchange + "'" + "and Name = '" + choice_name + "'"
ticker_name = pd.read_sql_query(query, db)
ticker_name = ticker_name.loc[0][0]

# st.write("This is a nice country  ", choice_country)
# st.write("It has exchange:,",choice_exchange)
# st.write(choice_name)

# get start date
#start_date = st.sidebar.date_input("Start Date", value=datetime.date.today() - datetime.timedelta(days=30))
#st.write(start_date)

# get end date
#end_date = st.sidebar.date_input("End Date", value=datetime.date.today())
#st.write(end_date)
#st.write(str(ticker_name))

# get interval
interval = st.sidebar.selectbox("Interval", ['1d', '1wk', '1mo', '3mo'])

#get period
period = st.sidebar.selectbox("Period",['1mo','3mo','6mo','1y','2y','5y','10y','max'],index = 2)

# get stock data
stock = yf.Ticker(str(ticker_name))
#data = stock.history(interval=interval, start=start_date, end=end_date)
data = stock.history(interval=interval, period=period)

if len(data)==0:
    st.write("Unable to retrieve data.This ticker may no longer be in use. Try some other stock")
else:

    #preprocessing
    data = preprocessing(data,interval)

    if period == '1mo' or period == '3mo':
        horizon = st.sidebar.slider("Forecast horizon",1,15,5)
    else:
        if interval == '1d' or interval == '1wk':
            horizon = st.sidebar.slider("Forecast horizon", 1, 30, 5)
        else:
            horizon = st.sidebar.slider("Forecast horizon", 1, 15, 5)

    model = st.selectbox('Model',['Simple Exponential Smoothing','Halt Model','Holt-Winter Model','Auto Regressive Model',
                                  'Moving Average Model','ARMA Model', 'ARIMA Model','AutoARIMA',
                                  'Linear Regression','Random Forest', 'Gradient Boosting','Support Vector Machines',
                                  ])

    if model=='Simple Exponential Smoothing':
        col1,col2 = st.columns(2)
        with col1:
            alpha_high = st.slider("Alpha_high",0.0,1.0,0.20)
        with col2:
            alpha_low = st.slider("Alpha_low",0.0,1.0,0.25)
        from SES import SES_model
        data_final, smap_low, smap_high, optim_alpha_high, optim_alpha_low = SES_model(data,horizon,alpha_high,alpha_low)

#data_final
        st.line_chart(data_final[['High','Forecast_High','Low','Forecast_Low']])
        col1,col2 = st.columns(2)
        with col1:
            st.write("SMAPE for High: {}".format(smap_high))
            st.write("Optimal Alpha for High : {} ".format(optim_alpha_high))
        with col2:
            st.write("SMAPE for Low: {}".format(smap_low))
            st.write("Optimal Alpha for Low: {} ".format(optim_alpha_low))

    elif model == 'Halt Model':
        col1, col2,col3,col4 = st.columns(4)
        with col1:
            level_high = st.slider("Level High", 0.0, 1.0, 0.20)
        with col2:
            trend_high = st.slider("Trend high", 0.0, 1.0, 0.20)
        with col3:
            level_low = st.slider("Level low", 0.0, 1.0, 0.20)
        with col4:
            trend_low = st.slider("Trend Low", 0.0, 1.0, 0.20)
        from SES import Holt_model
        data_final,smap_low,smap_high,optim_level_high,optim_level_low,optim_trend_high,optim_trend_low = Holt_model(data,horizon
                                                                        ,level_high,level_low,trend_high,trend_low)
        st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        col1, col2 = st.columns(2)
        with col1:
            st.write("SMAPE for High: {}".format(smap_high))
            st.write("Optimal Level for High : {} ".format(optim_level_high))
            st.write("Optimal Trend for High : {} ".format(optim_trend_high))
        with col2:
            st.write("SMAPE for Low: {}".format(smap_low))
            st.write("Optimal Level for Low: {} ".format(optim_level_low))
            st.write("Optimal Trend for Low: {} ".format(optim_trend_low))


    elif model == 'Holt-Winter Model':
        col1, col2 = st.columns(2)
        with col1:
            level_high = st.slider("Level High", 0.0, 1.0, 0.20)
            trend_high = st.slider("Trend high", 0.0, 1.0, 0.20)
            season_high = st.slider("Seasonal high", 0.0, 1.0, 0.20)
        with col2:
            level_low = st.slider("Level low", 0.0, 1.0, 0.20)
            trend_low = st.slider("Trend Low", 0.0, 1.0, 0.20)
            season_low = st.slider("Seasonal Low", 0.0, 1.0, 0.20)
        from SES import Holt_Winter_Model
        data_final, smap_low, smap_high, optim_level_high, optim_level_low, optim_trend_high, optim_trend_low, optim_season_high, optim_season_low = Holt_Winter_Model(data,horizon, level_high, level_low,trend_high,trend_low,season_high,season_low)

        st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        col1, col2 = st.columns(2)
        with col1:
            st.write("SMAPE for High: {}".format(smap_high))
            st.write("Optimal Level for High : {} ".format(optim_level_high))
            st.write("Optimal Trend for High : {} ".format(optim_trend_high))
            st.write("Optimal Seasonal smoothing for high: {}".format(optim_season_high))
        with col2:
            st.write("SMAPE for Low: {}".format(smap_low))
            st.write("Optimal Level for Low: {} ".format(optim_level_low))
            st.write("Optimal Trend for Low: {} ".format(optim_trend_low))
            st.write("Optimal Seasonal smoothing for Low: {}".format(optim_season_low))

    elif model == 'Auto Regressive Model':
        col1, col2 = st.columns(2)
        with col1:
            p_high = st.slider("Order of High", 1, 30, 1)
        with col2:
            p_low = st.slider("Order of Low", 1, 30, 1)
        from SES import AR_model

        data_final, smap_high, smap_low = AR_model(data,horizon,p_high,p_low)
        st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        col1, col2 = st.columns(2)
        with col1:
            st.write("SMAPE of High: {}".format(smap_high))
        with col2:
            st.write("SMAPE of Low : {}".format(smap_low))

    elif model == 'Moving Average Model':
        col1, col2 = st.columns(2)
        with col1:
            q_high = st.slider("Order of High", 1, 30, 1)
        with col2:
            q_low = st.slider("Order of Low", 1, 30, 1)
        from SES import AR_model
        data_final, smap_high, smap_low = AR_model(data, horizon, q_high, q_low)
        st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        col1, col2 = st.columns(2)
        with col1:
            st.write("SMAPE of High: {}".format(smap_high))
        with col2:
            st.write("SMAPE of Low : {}".format(smap_low))

    elif model == 'ARMA Model':
        col1, col2 = st.columns(2)
        with col1:
            p_high = st.slider("Order of AR High", 1, 30, 1)
            q_high = st.slider("Order of MA High", 1, 30, 1)
        with col2:
            p_low = st.slider("Order of AR Low", 1, 30, 1)
            q_low = st.slider("Order of MA Low", 1, 30, 1)
        from SES import ARMA_model
        data_final, smap_high, smap_low = ARMA_model(data,horizon,p_high,p_low,q_high,q_low)
        st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        col1, col2 = st.columns(2)
        with col1:
            st.write("SMAPE of High: {}".format(smap_high))
        with col2:
            st.write("SMAPE of Low : {}".format(smap_low))

    elif model == 'ARIMA Model':
        col1, col2 = st.columns(2)
        with col1:
            p_high = st.slider("Order of AR High", 1, 30, 1)
            q_high = st.slider("Order of MA High", 1, 30, 1)
            i_high = st.slider("Order of Differencing High" , 0,10,0)
        with col2:
            p_low = st.slider("Order of AR Low", 1, 30, 1)
            q_low = st.slider("Order of MA Low", 1, 30, 1)
            i_low = st.slider("Order of Differencing Low", 0, 10, 0)
        from SES import ARIMA_model
        data_final, smap_high, smap_low = ARIMA_model(data,horizon,p_high,p_low,q_high,q_low,i_high,i_low)
        st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])
        col1, col2 = st.columns(2)
        with col1:
            st.write("SMAPE of High: {}".format(smap_high))
        with col2:
            st.write("SMAPE of Low : {}".format(smap_low))
    elif model == 'AutoARIMA':
        from SES import Auto_Arima
        st.write("Note: This model may take some time to fit")
        data_final = Auto_Arima(data,horizon)
        st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])

    else:
        from ML_models import forecast
        #data_final = forecast(data,horizon,model)
        data_final, smape_high, smape_low = forecast(data,horizon,model)
        st.line_chart(data_final[['High', 'Forecast_High', 'Low', 'Forecast_Low']])

        col1, col2 = st.columns(2)
        with col1:
            st.write("SMAPE of High: {}".format(smape_high))
        with col2:
            st.write("SMAPE of Low : {}".format(smape_low))

db.close()
