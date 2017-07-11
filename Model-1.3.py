import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing

def cleanToPandas(filePath):
    cleaned = pd.read_csv(filePath, index_col=None, na_values=['.'])
    cleaned = cleaned.fillna(method='ffill')
    return cleaned

def volatilityByDay(series, depth):
    x = np.matrix(series.index.values).T
    y = np.matrix(series.values).T
    regr = linear_model.LinearRegression()
    values = np.zeros(len(series))
    for i in range(depth+1, len(y)):
        regr.fit(x[i - depth:i], y[i - depth:i])
        m = regr.coef_[0]
        b = regr.intercept_
        est = m * i + b
        error = abs(est - y[i])
        values[i] = error.item(0)
    return values

def rollingAverage(data, depth):
    rolling = pd.rolling_mean(pd.DataFrame(data), depth)
    rolling = rolling.fillna(0)
    rolling = rolling[0].values
    return rolling

def bigPlot(xVals, yVals, labels, xTicks, title):
    plt.figure(figsize=(10, 5))
    for i in range(len(xVals)):
        plt.plot(xVals[i], yVals[i], label=labels[i])
    plt.xticks(xVals[0][::365], xTicks[::365])
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Dollars', fontsize=12)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()

def mergeByDate(dfs):
    result = dfs[0]
    for i in range(1, len(dfs)):
        result = pd.merge(result, dfs[i], on='DATE', how='left')
    result = result.fillna(method='ffill')
    result = result.fillna(method='bfill')
    return result

def pullAndMerge(*args):
    data = [None] * len(args)
    for i in range(0, len(args)):
        data[i] = cleanToPandas('/Users/josephokeefe/Desktop/ResearchProject/Data/%s.csv' % args[i])
    return mergeByDate(data)


data = pullAndMerge('SP500', 'BAMLH0A0HYM2', 'DGS10', 'FEDFUNDS', 'T10Y2Y', 'USROA', 'USROE')
sp500 = data.filter(items=['DATE', 'SP500'])
data = data.drop(['DATE', 'SP500'], axis=1)

volt = volatilityByDay(sp500['SP500'], 150)
voltAv = rollingAverage(volt, 100)
norm = preprocessing.scale(voltAv)

X = data.as_matrix()
y = norm

X_train, X_test = X[:-90], X[:-90]
y_train, y_test = y[90:], y[90:]

X_train, X_test = X_train[200:1800], X_test[1800:]
y_train, y_test = y_train[200:1800], y_test[1800:]

est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
pred1 = np.concatenate([np.zeros(90), est.predict(X)])
error1 = mean_squared_error(y_test, est.predict(X_test))
print("Error: %s" % error1)
x = range(0, len(pred1))


neigh = KNeighborsRegressor(n_neighbors=3, weights='distance').fit(X_train, y_train)
pred2 = np.concatenate([np.zeros(90), neigh.predict(X)])
error2 = mean_squared_error(y_test, neigh.predict(X_test))
print("Error: %s" % error2)

xVals = [x[:2481], x, x]
yVals = [norm, pred1, pred2]
labels = ['ref', 'Gradient Boosting Model', 'KNN', 'Volatility']
bigPlot(xVals, yVals, labels, sp500['DATE'], 'S&P 500 Volatility\nMeasured as the Predictive Error of 150 Day Regressions')
