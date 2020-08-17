# Price Prediction and Investment Prototype
"""
Author: Rohaan Ahmed
Start Date: October 23, 2019
"""

# Importing libraries #
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import datetime, date, timedelta
from textblob import TextBlob
from newsapi import NewsApiClient
from sklearn.metrics import mean_squared_error, r2_score
import alpaca_trade_api as tradeapi
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn  
import yfinance as yf 
from math import sqrt
import time

#Constants Declaration #
HIGH_CONST = 13
LOW_CONST = 0
ETF = "GE"
Company = "General Electric"
HISTORICAL_DAYS = 500
HISTORICAL_MINUTES = 1000
NEWSAPI_KEY = '979932a04d094470a69cc59911137065'
API_KEY = "PKT3AR641ELLEDEMEKTX"
API_SECRET = "zVSz24oiN/ffUJPcdy3WGEv66oDkNd02bzZT7OU2"
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"
TIME_LOOP_DELAY = 60 #in seconds
time_loop_delay_in_mins = int(round(TIME_LOOP_DELAY/60)) - 1
ROUND_TO = 3
TEST_SIZE = 25

# Declaring dates
#startdate = (date.today() - timedelta(HISTORICAL_DAYS)).strftime('%Y-%m-%d')
#print("Historical data starting at:", startdate)
today = date.today().strftime('%Y-%m-%d')

last_state = 'sell'
dummy_investment = 0
#last_sell_price = 0
#last_buy_price = 0
original_investment = 0
profit = 0
last_buy_price = 0
last_sell_price = 0
#buysell = 0

alpaca_api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, 'v2')

while True:
        
        # Historical data - MINUTES  
    
    barset = alpaca_api.get_barset(ETF, '1Min', HISTORICAL_MINUTES)
    df2 = barset[ETF].df
    df2 = df2[['open', 'close', 'volume']]
    df2 = df2.dropna()  
    df2['PrevClose'] = df2['close'].shift(1)
    df2['5MinAvg'] = df2['close'].shift(1).rolling(window=5).mean()  
    df2['10MinAvg'] = df2['close'].shift(1).rolling(window=10).mean()  
    df2 = df2.dropna()   
    
    X = df2[['open', 'volume', 'PrevClose','5MinAvg','10MinAvg']]
    X.head()
    
    y = df2['close']
    y.head()
    
    # Splitting Test and Training Sets
    t=0.95
    t = int(t*len(df2))  
    t = len(df2) - TEST_SIZE
    # Train dataset  
    X_train = X[:t]  
    y_train = y[:t]  
    # Test dataset  
    X_test = X[t:]  
    y_test = y[t:]  
    
    df2.close[(round(len(df2)/4))*3:].plot(figsize=(6,5))
    plt.ylabel("Price")
    plt.show()
    
    actual_trend = np.asarray(y_test)[len(y_test) - 1] - np.asarray(y_test)[len(y_test) - (2 + time_loop_delay_in_mins)]
    actual_trend = round(actual_trend, ROUND_TO)
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.values.reshape(-1,1))
    
    # Creating Regressors
    
    """Linear Regression"""
    linear = LinearRegression()
    linear.fit(X_train,y_train) 
    """Random Forest Regression"""
    randomforest = RandomForestRegressor(n_estimators = 500)
    randomforest.fit(X_train, y_train) 
    """Polynomial Regression"""
    lin_reg = PolynomialFeatures(degree = 4)
    X_poly = lin_reg.fit_transform(X_train)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y_train)
    """Support Vector Regression Regression"""
    svr = SVR(kernel = 'rbf') #Using gaussian kernel
    svr.fit(X_train, y_train)
    
    # Generating Predictions
    predicted_price_lin = sc_y.inverse_transform(linear.predict(X_test))
    predicted_price_lin = pd.DataFrame(predicted_price_lin,index=y_test.index,columns = ['LIN price']) 
    predicted_price_rfr = sc_y.inverse_transform(randomforest.predict(X_test))
    predicted_price_rfr = pd.DataFrame(predicted_price_rfr,index=y_test.index,columns = ['RFR price']) 
    
    predicted_price_poly = lin_reg.fit_transform(X_test)
    predicted_price_poly = poly_reg.predict(predicted_price_poly)
    predicted_price_poly = sc_y.inverse_transform(predicted_price_poly)
    predicted_price_poly = pd.DataFrame(predicted_price_poly,index=y_test.index,columns = ['Poly price']) 
    
    predicted_price_svr = sc_y.inverse_transform(svr.predict(X_test))
    predicted_price_svr = pd.DataFrame(predicted_price_svr,index=y_test.index,columns = ['SVR price']) 
    
    # Trading Strategy
    """
    Buy and Keep conditions:
        - Predicted slope > 0
        - Actual slope  > 0
        - Mean Sentiment Polarity for 7 days > 0
    In all other cases, Sell 
    """
    
    plt.plot(predicted_price_lin, color = 'blue')#, label = 'predicted_price_lin')
    #plt.scatter(y_test.index,predicted_price_lin, color = 'blue', label = 'predicted_price_lin')
    #plt.plot(predicted_price_rfr, color = 'green', label = 'predicted_price_rfr')
    #plt.plot(predicted_price_poly, color = 'orange', label = 'predicted_price_poly')
    #plt.plot(predicted_price_svr, color = 'brown', label = 'predicted_price_svr')
    #plt.plot(buysell, color = 'purple', label = 'buy or sell')
    #plt.plot(investment, color = 'black', label = 'investment')
    #plt.scatter(y_test.index,investment, color = 'black', label = 'investment')
    plt.plot(y_test, color = 'red')#, label = 'actual')
    #plt.scatter(time.now(), buysell, color = 'purple')
    
    plt.ylabel("Closing Price " + ETF) 
    plt.grid(which='major', linewidth='0.5', color='black')
    #plt.legend()
    plt.pause(0.05)
    
    # Accuracy Measurements
    rms_lin = sqrt(mean_squared_error(y_test, predicted_price_lin))
    rms_rfr = sqrt(mean_squared_error(y_test, predicted_price_rfr))
    rms_poly = sqrt(mean_squared_error(y_test, predicted_price_poly))
    rms_svr = sqrt(mean_squared_error(y_test, predicted_price_svr))
    
    r2_lin = r2_score(y_test, predicted_price_lin)
    r2_rfr = r2_score(y_test, predicted_price_rfr)
    r2_poly = r2_score(y_test, predicted_price_poly)
    r2_svr = r2_score(y_test, predicted_price_svr)
    
    # Trading
    
    prediction_trend_long = np.asarray(predicted_price_lin)[len(predicted_price_lin) - 1] - np.asarray(predicted_price_lin)[len(predicted_price_lin) - (2 + time_loop_delay_in_mins)]
    #round(prediction_trend_long, ROUND_TO)

    current_price = y_test[len(y_test) - 1]
    previous_price = y_test[len(y_test) - 2]
    
    print('Actual Trend: ' + str(actual_trend))
    print('Predicted Trend: ' + str(prediction_trend_long))
    print('Last State: ' + last_state)
    print('Current Price: ' + str(current_price))
    print('Previous Price: ' + str(previous_price))
    
#    if (prediction_trend_long < 0 or actual_trend < 0) and last_state == 'buy':
    if (prediction_trend_long < 0) and last_state == 'buy':
        """Indicate Sell if the either slope is greater than 0 (Price Decrease)"""
        print('SELL CONDITION')
        dummy_investment = dummy_investment + (dummy_investment*(current_price - previous_price))/previous_price
        last_state = 'sell'
        last_sell_price = current_price
        profit = profit + (current_price - last_buy_price)
        order = alpaca_api.submit_order(symbol = ETF, qty = 1, side = 'sell', type = 'market', time_in_force = 'gtc')
        order = alpaca_api.cancel_all_orders()
#     elif (prediction_trend_long > 0 and actual_trend > 0) and last_state == 'sell':
    elif (prediction_trend_long > 0) and last_state == 'sell':
        print('BUY CONDITION')
        """Indicate Buy if both slopes are greater than 0 (Price Increase)"""
        if dummy_investment == 0:
            original_investment = current_price
            dummy_investment = current_price
        dummy_investment = dummy_investment + (dummy_investment*(current_price - previous_price))/previous_price
        last_state = 'buy'
        last_buy_price = current_price
        order = alpaca_api.submit_order(symbol = ETF, qty = 1, side = 'buy', type = 'market', time_in_force = 'gtc')
    else:
        print('KEEP LAST CONDITION')
        if last_state == 'buy':
            dummy_investment = dummy_investment + (dummy_investment*(current_price - previous_price))/previous_price
            profit = profit + (current_price - last_buy_price)
    
    #dummy_investment - original_investment
    
    plt.show() 
    print('Dummy Investment: ' + str(dummy_investment))
    print('Original Investment: ' + str(original_investment))
    print('Profit: ' + str(profit))
    print('Time: ' + str(datetime.now()))
    print('---------------------------')
    time.sleep(TIME_LOOP_DELAY)