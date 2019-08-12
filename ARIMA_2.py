import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from datetime import datetime as dt
from math import sqrt
import numpy as np
from operator import add
from pandas import read_csv
from pandas import Series
import numpy as np


dataset = pd.read_csv('shampoo-sales_ARIMA.csv')
new_df = pd.DataFrame(dataset)
new_df.fillna(10,inplace=True)

new_df["Month"] = pd.to_datetime(new_df["Month"])
new_df = new_df.set_index('Month')

from math import log, exp

#new_df['Diff'] = new_df['Outbound Utilization (%)'] - new_df['Outbound Utilization (%)'].shift(1)
new_df["Logarithmic"] = np.log(new_df["Sales"])

new_df["DoubleLogarithmic"] = np.log(new_df["Logarithmic"])

#new_df["BacktoOrig"] = np.exp(new_df["Logarithmic"])

new_df["DoubleLogarithmic"].fillna(0.001,inplace=True)
ts_values = new_df["DoubleLogarithmic"].values
ts_log = np.log(new_df['Logarithmic'])
#Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(ts_values)
#new_df.to_csv("C:/Python/x.csv")

X = ts_values
size = int(len(X) * 0.667)
train, test = X[0:size], X[size:len(X)]


'''
X = new_df["Outbound Utilization (%)"]
size = int(len(X) * 0.995)
train, test = X[0:size], X[size:len(X)]


X = ts.values
size = int(len(X) * 0.995)
train, test = X[0:size], X[size:len(X)]


from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
plot_acf(train, lags=20)
pyplot.show()

#Partial AutoCorrelation
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(train, lags=20)
pyplot.show()
'''




#AR Model
#training will be 66%, test will be 33% as per our model
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from numpy.linalg import LinAlgError
import warnings
warnings.filterwarnings("ignore")
history = [x for x in train]
predictions = list()
#test.reset_index()
for t in range(len(test)):
    try:
        model = ARIMA(history, order=(1,1,1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    except (ValueError, LinAlgError):
        pass
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
rmse = mean_squared_error(test, predictions)**0.5
print('Test MSE: %.3f' % rmse)



#abc = np.vstack(predictions)

from math import sqrt
rms = sqrt(mean_squared_error(test, predictions))

# plot
pyplot.plot(test, color = 'blue')   
pyplot.plot(predictions, color='red')
pyplot.show()






pred_df = pd.DataFrame(predictions)


#pred_df.to_csv("C:/Python/xx.csv")

new_df = pd.read_csv("Predictions.csv")

new_df = new_df[len(train): len(new_df)]

new_df["1StepExpo"] = np.exp(new_df["Prediction"])
new_df["FinalPrediction"] = np.exp(new_df["1StepExpo"])
#new_df.to_csv("C:/Python/x.csv")
#new_dataset = pd.read_csv("C:/Python/Logarithmic.csv")


#new_dataset = new_dataset[len(train): len(new_dataset)]

pyplot.plot(new_df["FinalPrediction"], label="predict")
pyplot.plot(new_df["Sales"], label="actual")
pyplot.legend()
pyplot.show()

#rmse = mean_squared_error(new_df["FinalPrediction"].iloc[24:], new_df["Sales"].iloc[24:])**0.5
rmse = mean_squared_error(new_df["FinalPrediction"], new_df["Sales"])**0.5
print('Test MSE: %.3f' % rmse)


##########################################################################
##########################################################################
######################DETERMINE PDQ VALUES - DONOT RUN####################
##########################################################################
##########################################################################
import warnings
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.8)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')



import datetime
print(datetime.datetime.now())
p_values = [1,2,3,4,5]
d_values = [0,1]
q_values = [1,2,3]
warnings.filterwarnings("ignore")
evaluate_models(train, p_values, d_values, q_values)
print(datetime.datetime.now())