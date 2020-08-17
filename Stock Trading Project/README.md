Project for small-funds investment for short-trading of specific ETFs (stock) using open source data.
The followind data is used:
- Monthly and hourly stock data from Yahoo Finance, Google Finance, and Alpaca, for prediction
- News headlines from various sources, for Sentiment Analysis

Several methods were evaluated, including:
- Linear Regression and RNNs for Stock Price prediction at the next timestep
- XGBoost, Logistic Regression, and K-NEarest Neighbour Classification techniques for Buy/Sell/Keep classification (code not uploaded)

Ultimately, Linear Regression gave the best results, expecially for inferring prices that were "outside" of historical data ranges

Disclaimer: This was a passion project, never intended for use in real stock trading with large sums. It was tested in the Alpaca sandbox enviroment, and consistely made a WHOPPING 2% profit.
For details, please contact.
