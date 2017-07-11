import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn import tree as tree1
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

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
        if labels[i] == 'Conglomerate Regression':
            plt.plot(xVals[i], yVals[i], label=labels[i], linewidth=2.0, color='blue', zorder=5)
        elif labels[i] == 'Volatility':
            plt.plot(xVals[i], yVals[i], label=labels[i], linewidth=2.0, color='red', zorder=4)
        elif labels[i] == 'Mean':
            plt.plot(xVals[i], yVals[i], label=labels[i], linestyle='--', linewidth=2.0, color='black', zorder=3)
        elif labels[i] == 'Lasso':
            plt.plot(xVals[i], yVals[i], label='Machine Learning Regressions', linewidth=.75, color='grey')
        else:
            plt.plot(xVals[i], yVals[i], label='_nolegend_', linewidth=.75, color='grey')
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

data = pullAndMerge('SP500', 'BAMLH0A0HYM2', 'DGS10', 'FEDFUNDS', 'T10Y2Y', 'USROA', 'USROE', 'T10YFF', 'PSAVERT', 'BAMLEMHYHYLCRPIUSOAS')
sp500 = data.filter(items=['DATE', 'SP500'])
print(data.head())
data = data.drop(['DATE', 'SP500'], axis=1)

volt = volatilityByDay(sp500['SP500'], 150)
voltAv = rollingAverage(volt, 100)
norm = preprocessing.scale(voltAv)

X = data.as_matrix()
y = norm
y1 = norm

X_train, X_test = X[:-90], X[:-90]
y_train, y_test = y[90:], y[90:]

X_train, X_test = X_train[200:2200], X_test[2200:]
y_train, y_test = y_train[200:2200], y_test[2200:]

def algo(function):
    model = function.fit(X_train, y_train)
    pred = np.concatenate([np.zeros(90), model.predict(X)])
    error = mean_squared_error(y_test, model.predict(X_test))
    print('Error: %s' % error)
    return pred

gradBoost = algo(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls'))
knn = algo(KNeighborsRegressor(n_neighbors=3, weights='distance').fit(X_train, y_train))
lasso = algo(linear_model.Lasso(alpha=0.1))
elasticNet = algo(linear_model.ElasticNet(alpha=0.1))
bayRidge = algo(linear_model.BayesianRidge())
SGDReg = algo(SGDRegressor())
tree = algo(tree1.DecisionTreeRegressor())
randForest = algo(RandomForestRegressor(n_estimators=10))
extraTrees = algo(ExtraTreesRegressor(n_estimators=10))
ridge = algo(Ridge(alpha=1.0))
neuralMLP = algo(MLPRegressor())

newDf = pd.DataFrame({'gradBoost' : gradBoost, 'knn' : knn, 'lasso' : lasso, 'elasticNet' : elasticNet, 'bayRidge' : bayRidge, 'SGDReg' : SGDReg, 'tree' : tree, 'randForest' : randForest, 'extraTrees' : extraTrees, 'ridge' : ridge, 'neuralMLP' : neuralMLP})
y = np.mean(newDf.as_matrix(), axis=1)
y_test = y1[2200:]
error = mean_squared_error(y_test, y[2200:-90])
print('Supper Error: %s' % error)
super = y

x = range(0, len(gradBoost))
y10 = np.zeros(len(x[250:]))

xVals = [x[250:], x[250:2481], x[250:], x[250:], x[250:], x[250:], x[250:], x[250:], x[250:], x[250:], x[250:], x[250:], x[250:], x[250:]]
yVals = [y10, norm[250:], gradBoost[250:], knn[250:], lasso[250:], elasticNet[250:], bayRidge[250:], SGDReg[250:], tree[250:], randForest[250:], extraTrees[250:], ridge[250:], neuralMLP[250:], super[250:]]
labels = ['Mean', 'Volatility', 'Gradient Boosting', 'K Nearest Neighbors', 'Lasso', 'Elastic Net', 'Bayesian Ridge', 'Stochastic Gradient Descent', 'Decision Tree', 'Random Forest', 'Extra Trees', 'Ridge Regression', 'Multi-layer Perceptron', 'Conglomerate Regression']
bigPlot(xVals, yVals, labels, sp500['DATE'], 'S&P 500 Volatility\nMeasured as the Predictive Error of 150 Day Regressions')
