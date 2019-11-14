# Price Prediction and Investment Prototype
"""
Author: Rohaan Ahmed
Start Date: October 23, 2019
"""

# Machine learning libraries 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import date, timedelta

#Constant declaration
HIGH_CONST = 150
LOW_CONST = 0
ETF = "GLD"
HISTORICAL_DAYS = 1000

# pandas and numpy are used for data manipulation  
import pandas as pd  
import numpy as np  
# matplotlib and seaborn are used for plotting graphs  
import matplotlib.pyplot as plt  
import seaborn  
# yfinance is used to fetch historic data  
import yfinance as yf 

startdate = (date.today() - timedelta(HISTORICAL_DAYS)).strftime('%Y-%m-%d')
print("Start Date is:", startdate)
today = date.today() 
print("Today's date:", today)

# Read in historical data  
"""Df = yf.download('GLD','2008-01-01','2017-12-31')"""
Df = yf.download(tickers = ETF, start = startdate, end = today)
#Df = yf.download(tickers = 'MAXR', start = startdate, end = today)
#Df = yf.download(tickers = 'GLD', period = '5yr')
#Df = yf.download(tickers = 'MSFT', start = startdate, end = today)

# Only keep the closing price column
Df=Df[['Close']]  
# Drop rows with missing values  
Df= Df.dropna()  
# Plot the closing price  
#Df.Close.plot(figsize=(10,5))  
#plt.ylabel("Price")
#plt.show()

Df['PrevDay'] = Df['Close'].shift(-1)
Df['3DayAvg'] = Df['Close'].shift(1).rolling(window=3).mean()  
Df['9DayAvg']= Df['Close'].shift(1).rolling(window=9).mean()  
Df= Df.dropna()
X = Df[['PrevDay','3DayAvg','9DayAvg']]
X.head()

y = Df['Close']
y.head()

# Splitting Test and Training Sets
t=.75 
t = int(t*len(Df))  
# Train dataset  
X_train = X[:t]  
y_train = y[:t]  
# Test dataset  
X_test = X[t:]  
y_test = y[t:]  

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.values.reshape(-1,1))

#Creating Regressors
linear = LinearRegression()
linear.fit(X_train,y_train) 

randomforest = RandomForestRegressor(n_estimators = 500)
randomforest.fit(X_train, y_train) 

lin_reg = PolynomialFeatures(degree = 4)
X_poly = lin_reg.fit_transform(X_train)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train)

svr = SVR(kernel = 'rbf') #Using gaussian kernel
svr.fit(X_train, y_train)

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

#Future Predictions
"""
Trading Strategy: 
If predicted slope and actual slope of the price are both >0 Execute  buy
In all other cases, Execute sell
"""

slope_predicted_lin = np.asarray(predicted_price_lin)[1:] - np.asarray(predicted_price_lin)[:-1]
slope_actual = np.asarray(y_test)[1:] - np.asarray(y_test)[:-1]
i = 1
slope_predicted_lin[0] = 0
slope_actual[0] = 0

while i < len(slope_predicted_lin):
    if slope_predicted_lin[i] < 0 :
        """Indicate Sell if the predicted slope is greater than 0 (Price Decrease)"""
        slope_predicted_lin[i] = LOW_CONST
    elif slope_predicted_lin[i] > 0:
        """Indicate Buy if the predicted slope is greater than 0 (Price Increase)"""
        slope_predicted_lin[i] = HIGH_CONST
    else:
        """If slope is 0 (no change in price), keep last Buy/Sell state"""
        slope_predicted_lin[i] = slope_predicted_lin[i-1]
        #slope_predicted_lin[i] = LOW_CONST
    i+=1
slope_predicted_lin = np.insert(slope_predicted_lin, 0, 0)
slope_actual = np.insert(slope_actual, 0, 0)

buysell = np.empty(len(slope_predicted_lin))
buysell[:] = 0

i = 1
while i < len(slope_predicted_lin):
    #if slope_predicted_lin[i-1] == HIGH_CONST and slope_predicted_lin[i] == HIGH_CONST:
    if slope_actual[i] > 0 and slope_predicted_lin[i] == HIGH_CONST:
        #Indicate Buy if Predicted and Actual Slopes are > 0
        buysell[i] = HIGH_CONST
    else:
        buysell[i] = LOW_CONST
    i+=1


#Testing Investment
investment = np.empty(len(slope_predicted_lin))
investment[0] = y_test[0]
i = 1
while i < len(slope_predicted_lin):
    if buysell[i] == HIGH_CONST:
        #current_price = np.asarray(predicted_price_lin)[i]
        current_price = np.asarray(y_test)[i]
        previous_price = np.asarray(y_test)[i-1]
        investment[i] = investment[i-1] + (investment[i-1]*(current_price - previous_price))/previous_price
    else:
        investment[i] = investment[i-1]
    i+=1

buysell = pd.DataFrame(buysell,index=y_test.index,columns = ['buy or sell']) 
investment = pd.DataFrame(investment,index=y_test.index,columns = ['buy or sell']) 

plt.plot(predicted_price_lin, color = 'blue', label = 'predicted_price_lin')
plt.scatter(y_test.index,predicted_price_lin, color = 'blue', label = 'predicted_price_lin')
plt.plot(predicted_price_rfr, color = 'green', label = 'predicted_price_rfr')
plt.plot(predicted_price_poly, color = 'orange', label = 'predicted_price_poly')
plt.plot(predicted_price_svr, color = 'brown', label = 'predicted_price_svr')
plt.plot(buysell, color = 'purple', label = 'buy or sell')
plt.plot(investment, color = 'black', label = 'investment')
#plt.scatter(y_test.index,investment, color = 'black', label = 'investment')
plt.plot(y_test, color = 'red', label = 'actual')

plt.ylabel("Closing Price") 
plt.legend()
plt.show() 

from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

rms_lin = sqrt(mean_squared_error(y_test, predicted_price_lin))
rms_rfr = sqrt(mean_squared_error(y_test, predicted_price_rfr))
rms_poly = sqrt(mean_squared_error(y_test, predicted_price_poly))
rms_svr = sqrt(mean_squared_error(y_test, predicted_price_svr))

r2_lin = r2_score(y_test, predicted_price_lin)
r2_rfr = r2_score(y_test, predicted_price_rfr)
r2_poly = r2_score(y_test, predicted_price_poly)
r2_svr = r2_score(y_test, predicted_price_svr)

import alpaca_trade_api as tradeapi
API_KEY = "PKL0IPYLD9ZZ7WYFYJDX"
API_SECRET = "rb/duMv5dyOqS6bpQ/R/7lHMG9L72NC9bgmTlwFW"
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"

alpaca_api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, 'v2')

symbol_bars1 = yf.download(tickers = 'MAXR', period = '5d', interval = '1d')
symbol_bars2 = alpaca_api.get_barset('MAXR', 'day', 5).df
symbol_price = symbol_bars2['MAXR']['close']

order = alpaca_api.submit_order(symbol = 'MAXR', qty = 1, side = 'buy', type = 'market', time_in_force = 'gtc')
order = alpaca_api.cancel_all_orders()
r = alpaca_api.list_orders()


#r2_score_lin = linear.score(X[t:],y[t:])*100 
#float("{0:.2f}".format(r2_score)) 