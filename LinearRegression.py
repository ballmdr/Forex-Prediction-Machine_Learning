import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns

# Read data from csv 
data_train = pd.read_csv('Uniqlo_test_2017_day.csv')

# setting index to datetime
data_train = data_train.reindex(index=data_train.index[::-1])
data_train['Date'] = pd.to_datetime(data_train['Date'], format='%Y-%m-%d')
data_train.index = data_train['Date']
del data_train['Date']

sns.pairplot(data_train)

# setting index to datetime
data_test = pd.read_csv('Uniqlo(FastRetailing) 2017 Test - stocks2017.csv')
data_test = data_test.reindex(index=data_test.index[::-1])
data_test['Date'] = pd.to_datetime(data_test['Date'], format='%Y-%m-%d')
data_test.index = data_test['Date']
del data_test['Date']

# plotting dickey-fuller analysis 
decomposition = sm.tsa.seasonal_decompose(data_train['Stock Trading'],freq=1)
fig1 = decomposition.plot()
plt.show(fig1)

fig2 = data_train['Open'].plot()
plt.show(fig2)

fig3 = data_train['High'].plot()
plt.show(fig3)

fig4 = data_train['Low'].plot()
plt.show(fig4)

fig5 = data_train['Close'].plot()
plt.show(fig5)

fig6 = data_train['Volume'].plot()
plt.show(fig6)

fig7 = data_train['Stock Trading'].plot()
plt.show(fig7)

Y_train = data_train['Stock Trading']
Y_test = data_test['Stock Trading']
del data_train['Stock Trading']
del data_test['Stock Trading']
Y_train = Y_train.values
Y_test = Y_test.values
X_train = data_train.values
X_test = data_test.values

model = LinearRegression()
model.fit(X_train, Y_train)
Y_hypo = model.predict(X_test)
print("RMSE : %s" % str(math.sqrt(metrics.mean_squared_error(Y_hypo, Y_test))))
plt.plot([1,2,3,4,5,6,7], Y_test, 'r-')
plt.plot([1,2,3,4,5,6,7], Y_hypo, 'b-')
