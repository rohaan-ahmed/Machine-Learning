Project for small-funds investment for short-trading of specific ETFs (stock) using open source data.
The following data is used:
- Monthly and hourly stock data from Yahoo Finance, Google Finance, and Alpaca, for prediction
- News headlines from various sources, for Sentiment Analysis

Several methods were evaluated, including:
- Linear Regression and Recurrent Neural Networks for Stock Price prediction at the next timestep
- XGBoost, Logistic Regression, and K-NEarest Neighbour Classification techniques for Buy/Sell/Keep classification (code not uploaded)

Ultimately, Linear Regression gave the best results, expecially for inferring prices that were "outside" of historical data ranges

Disclaimer: This was a personal passion project, made after conversations with a few friends, and never intended for use in real stock trading with large sums. 
It was tested in the Alpaca sandbox enviroment, and consistently lost money! The issue was not prediction, but the speed at which trading took place. By the time an order was placed and filled, the predicted stock value was no longer relevant. This could possibly be resolved with delay risk assessment.

For details, please contact.
