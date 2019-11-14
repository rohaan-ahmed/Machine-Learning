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
API_KEY = "PKL0IPYLD9ZZ7WYFYJDX"
API_SECRET = "rb/duMv5dyOqS6bpQ/R/7lHMG9L72NC9bgmTlwFW"
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"
TIME_LOOP_DELAY = 60

# Declaring dates
startdate = (date.today() - timedelta(HISTORICAL_DAYS)).strftime('%Y-%m-%d')
print("Historical data starting at:", startdate)
today = date.today().strftime('%Y-%m-%d')

last_state = 'sell'
dummy_investment = 0
last_sell_price = 0
last_buy_price = 0
original_investment = 0
profit = 0
buysell = 0

while True:
    # Sentiment Analysis using News Data
    
    #newsapi = NewsApiClient(api_key = NEWSAPI_KEY)
    #
    #"""News Data from last week"""
    #day = (date.today() - timedelta(7))
    #iteration = 0
    #day_sentiment = [0,0]
    #sentiment_polarity_array = []
    #sentiment_objectivity_array = []
    #date_array = []
    #
    #if day.weekday() == 5:
    #    day = day + timedelta(2)
    #elif day.weekday() == 6:
    #    day = day + timedelta(1)
    #
    #while day <= date.today(): 
    #    """Getting news articles"""
    #    date_string = day.strftime('%Y-%m-%d')
    #    articles = newsapi.get_everything(q= Company, \
    #                                      from_param = date_string, \
    #                                      to = date_string, \
    #                                      language='en', \
    #                                      sort_by='relevancy', \
    #                                      page_size=10, \
    #                                      page=1)
    #    temp1 = 0
    #    sentence = ""
    #    """Extracting titles and performing sentiment analysis"""
    #    while temp1 < len(articles['articles']):
    #        sentence = sentence + articles['articles'][temp1]['title']
    #        sentence = sentence + " " 
    #        temp1 = temp1 + 1
    #        
    #        sentence = TextBlob(sentence)
    #        day_sentiment[0] = day_sentiment[0] + sentence.sentiment[0]
    #        day_sentiment[1] = day_sentiment[1] + sentence.sentiment[1]
    #        
    #        sentence = ""
    #    
    #    day_sentiment[0] = day_sentiment[0] / len(articles['articles'])
    #    day_sentiment[1] = day_sentiment[1] / len(articles['articles'])
    #    
    #    sentiment_polarity_array.append(day_sentiment[0])
    #    sentiment_objectivity_array.append(day_sentiment[1])
    #    date_array.append(date_string)
    #    
    #    day_sentiment = [0,0]
    #    if day.weekday() == 4:
    #        day = day + timedelta(3)
    #    elif day.weekday() == 5:
    #        day = day + timedelta(2)
    #    elif day.weekday() == 6:
    #        day = day + timedelta(1)
    #    else:
    #        day = day + timedelta(1)
    
    
    ## Historical data - DAILY  
    #df1 = yf.download(tickers = ETF, start = startdate, end = (date.today() + timedelta(1)).strftime('%Y-%m-%d'))
    ##df1 = yf.download(tickers = 'GLD', period = '5yr')
    #
    #df1 = df1[['Open','Close', 'Volume']]
    ##df1['polarity'] = sentiment_polarity_array
    ##df1['objectivity'] = sentiment_objectivity_array
    #df1 = df1.dropna()  
    ##df1.Close.plot(figsize=(10,5))  
    ##plt.ylabel("Price")
    ##plt.show()
    #
    #df1['PrevDay'] = df1['Close'].shift(-1)
    #df1['3DayAvg'] = df1['Close'].shift(1).rolling(window=3).mean()  
    #df1['9DayAvg'] = df1['Close'].shift(1).rolling(window=9).mean()   
    #df1= df1.dropna()
    #
    ##X = df1[['PrevDay','3DayAvg','9DayAvg','polarity','objectivity']]
    #X = df1[['Open','Volume','PrevDay','3DayAvg','9DayAvg',]]
    #X.head()
    #
    #y = df1['Close']
    #y.head()
    
    # Historical data - MINUTES  
    alpaca_api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, 'v2')
    barset = alpaca_api.get_barset(ETF, '1Min', HISTORICAL_MINUTES)
    df2 = barset[ETF].df
    
    df2 = df2[['open', 'close', 'volume']]
    df2 = df2.dropna()  
    df2['PrevMin'] = df2['close'].shift(-1)
    df2['5MinAvg'] = df2['close'].shift(1).rolling(window=5).mean()  
    df2['15MinAvg'] = df2['close'].shift(1).rolling(window=15).mean()  
    df2 = df2.dropna()   
    #df2.close.plot(figsize=(10,5))  
    #plt.ylabel("Price")
    #plt.show()
    
    X = df2[['open', 'volume','PrevMin','5MinAvg','15MinAvg',]]
    X.head()
    
    y = df2['close']
    y.head()
    
    # Splitting Test and Training Sets
    t=0.95
    t = int(t*len(df2))  
    # Train dataset  
    X_train = X[:t]  
    y_train = y[:t]  
    # Test dataset  
    X_test = X[t:]  
    y_test = y[t:]  
    
    actual_trend = np.asarray(y_test)[len(y_test) - 1] - np.asarray(y_test)[len(y_test) - (1 + TIME_LOOP_DELAY)]
    
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
    #"""comment from here"""
    #
    ##avg_sentiment = sum(sentiment_polarity_array) / len(sentiment_polarity_array)
    #
    #slope_predicted_lin = np.asarray(predicted_price_lin)[1:] - np.asarray(predicted_price_lin)[:-1]
    #slope_actual = np.asarray(y_test)[1:] - np.asarray(y_test)[:-1]
    #i = 1
    #slope_predicted_lin[0] = 0
    #slope_actual[0] = 0
    #
    #while i < len(slope_predicted_lin):
    #    if slope_predicted_lin[i] < 0 :
    #        """Indicate Sell if the predicted slope is < 0 (Price Decrease)"""
    #        slope_predicted_lin[i] = LOW_CONST
    #    elif slope_predicted_lin[i] > 0:
    #        """Indicate Buy if the predicted slope is > 0 (Price Increase)"""
    #        slope_predicted_lin[i] = HIGH_CONST
    #    else:
    #        """If slope is 0 (no change in price), keep last Buy/Sell state"""
    #        slope_predicted_lin[i] = slope_predicted_lin[i-1]
    #    i+=1
    #    
    #slope_predicted_lin = np.insert(slope_predicted_lin, 0, 0)
    #slope_actual = np.insert(slope_actual, 0, 0)
    #
    #buysell = np.empty(len(slope_predicted_lin))
    #buysell[:] = 0
    #
    #i = 1
    #while i < len(slope_predicted_lin):
    #    """Buy Condition"""
    #    if slope_actual[i] > 0 and slope_predicted_lin[i] == HIGH_CONST:
    #        buysell[i] = HIGH_CONST
    #    else:
    #        """Sell Condition"""
    #        buysell[i] = LOW_CONST
    #    i+=1
    #
    ##Testing Investment
    #investment = np.empty(len(slope_predicted_lin))
    #investment[0] = y_test[0]
    #i = 1
    #while i < len(slope_predicted_lin):
    #    if buysell[i] == HIGH_CONST:
    #        #current_price = np.asarray(predicted_price_lin)[i]
    #        current_price = np.asarray(y_test)[i]
    #        previous_price = np.asarray(y_test)[i-1]
    #        investment[i] = investment[i-1] + (investment[i-1]*(current_price - previous_price))/previous_price
    #    else:
    #        investment[i] = investment[i-1]
    #    i+=1
    #
    #buysell = pd.DataFrame(buysell,index=y_test.index,columns = ['buy or sell']) 
    #investment = pd.DataFrame(investment,index=y_test.index,columns = ['buy or sell']) 
    #"""comment until here"""
    
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
    
    prediction_trend_long = np.asarray(predicted_price_lin)[len(predicted_price_lin) - 1] - np.asarray(predicted_price_lin)[len(predicted_price_lin) - (2 + TIME_LOOP_DELAY)]
    prediction_trend_short = np.asarray(predicted_price_lin)[len(predicted_price_lin) - 1] - np.asarray(predicted_price_lin)[len(predicted_price_lin) - (2 + TIME_LOOP_DELAY)/2]
    
    print('Actual Trend: ' + str(actual_trend))
    print('Predicted Trend: ' + str(prediction_trend_long))
    print('Last State: ' + last_state)

    if (prediction_trend_long > 0 and actual_trend > 0) and last_state == 'sell':
        """Indicate Buy if the slopes are > 0 (Price Increase)"""
        print('BUY CONDITION')
        last_state = 'buy'
        last_buy_price = y_test[len(y_test) - 1]
        buysell = HIGH_CONST
        if dummy_investment == 0:
            dummy_investment = last_buy_price
            original_investment = last_buy_price
        order = alpaca_api.submit_order(symbol = ETF, qty = 1, side = 'buy', type = 'market', time_in_force = 'gtc')
    elif (prediction_trend_long < 0 or actual_trend < 0) and last_state == 'buy':
        """Indicate Buy if the slopes are < 0 (Price Decrease)"""
        print('SELL CONDITION')
        last_state = 'sell'
        last_sell_price = y_test[len(y_test) - 1]
        buysell = LOW_CONST
        dummy_investment = last_sell_price
        profit = profit + (last_sell_price - last_buy_price)
        order = alpaca_api.submit_order(symbol = ETF, qty = 1, side = 'sell', type = 'market', time_in_force = 'gtc')
        order = alpaca_api.cancel_all_orders()
        alpaca_api.get_asset(ETF)
    else:
        """If slopes are 0 (no change in price), keep last Buy/Sell state"""
        print('KEEP LAST CONDITION')
        last_state = last_state
        current_price = y_test[len(y_test) - 1]
        if last_state == 'buy':
            dummy_investment = y_test[len(y_test) - 1]
            profit = profit + (current_price - last_buy_price)
        else:
            dummy_investment = dummy_investment

    r = alpaca_api.list_orders()
    plt.show() 
    print('Dummy Investment: ' + str(dummy_investment))
   # print('Original Investment: ' + str(original_investment))
    print('Profit: ' + str(profit))
    print('Time: ' + str(datetime.now()))
    print('---------------------------')
    time.sleep(TIME_LOOP_DELAY)