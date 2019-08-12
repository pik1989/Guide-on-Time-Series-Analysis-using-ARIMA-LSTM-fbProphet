# Time-Series-Forecasting-using-LSTM-ARIMA-fbProphet
A small guide on Time Series.

What is TS Analysis?

Time Series: 
Set of observations taken at a specified time usually at equal intervals. 
It is used to predict future values based on previous observed values.

Considering a graph, when x is time & if the dependent variable depends on time parameter then it’s time series analysis.


Components of TS Analysis:

Trend
Seasonality
Irregularity
Cyclic

![Test Image 7](https://github.com/pik1989/Time-Series-Forecasting-using-LSTM-ARIMA-fbProphet/blob/master/Images/Component-of-Time-Series-Data.jpg)

When not to use TS Analysis?

1. If the values are constant
2. If the values are in form of a function

![Test Image 7](https://github.com/pik1989/Time-Series-Forecasting-using-LSTM-ARIMA-fbProphet/blob/master/Images/pic1.jpg)
![Test Image 7](https://github.com/pik1989/Time-Series-Forecasting-using-LSTM-ARIMA-fbProphet/blob/master/Images/pic2.gif)

Stationarity?

TS data can be stationary by removing:

1. Trend - Varying over time
2. Seasonality - Variations at specific time

How?

1. Constant mean - Average
2. Constant variance - Distance from mean
3. Auto covariance that does not depend on time

How to test stationarity?

Rolling Statistics: Visual technique

Augmented Dickey Fuller Test: Here the null hypothesis is that the TS is non-stationary.If ‘Test Statistics’ < ‘Critical Value’, then we can reject the hypothesis & conclude that TS is stationary.

ARIMA:

Auto Regression: Auto Regressive lags, If there’s a correlation between t & t-5, then that’s an autoregressive model
If p is 5, then predictors of x(t) = x(t-1)….x(t-5)

![Test Image 7](https://github.com/pik1989/Time-Series-Forecasting-using-LSTM-ARIMA-fbProphet/blob/master/Images/ACF_PACF.png)



Moving Average: Lagged forecast errors in prediction.
If q is 2, predictors of x(t) will be e(t-1)..e(t-2)
e(i) is the difference between moving average of ith instance & actual value

ACF: Auto Correlation function, it’s the measure of correlation between TS and the lagged value of itself.
PACF: Partial Auto Correlation function, it’s the correlation of TS with a lagged value of itself but after removing variation.


LSTM (Long Short Term Memory)

Recurrent Neural Network:

Vector to Sequence – I/P (Image)  Describes an image
	Example: Image Captioning
Sequence to Vector – I/P(Product Reviews)  O/P is in form of a vector [0.9 0.1] of positive: negative
	Example: Sentiment Analysis
Sequence to Sequence – I/P(Sequence)  O/P(Sequence)
	It’s based on Encoder-Decoder Architecture
	Example: Translation
  
1. TS data is actually sequences.
2. When dealing with weather data  Precipitation, Rain, Temperature etc.
3. Where some of the features can be relevant for forecasting, weather entirely is treated as a vector & is i/p to the neural network.
4. TS can be modelled as a sequence to sequence problem

![Test Image 7](https://github.com/pik1989/Time-Series-Forecasting-using-LSTM-ARIMA-fbProphet/blob/master/Images/LSTM.png)

