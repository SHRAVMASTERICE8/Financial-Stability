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
from sklearn.model_selection import train_test_split

def cleanToPandas(filePath):
    cleaned = pd.read_csv(filePath, index_col=None, na_values=['.'])
    cleaned = cleaned.fillna(method='ffill')
    cleaned = cleaned.fillna(method='bfill')
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
        data[i] = cleanToPandas('/Users/josephokeefe/Desktop/ResearchProject/Data2/%s.csv' % args[i])
    return mergeByDate(data)

def getFinalVolt(stuff, regDepth, smoothDepth):
    return preprocessing.scale(rollingAverage(volatilityByDay(stuff, regDepth), smoothDepth))

def setLag(indicators, volatility, lag):
    return indicators[:-lag], volatility[lag:]

def algosPlot(algos, ticks):
    plt.figure(figsize=(10, 5))
    for algo in algos:
        pred = algos[algo]
        if algo == 'Volatility':
            plt.plot(range(0, len(pred)), pred, label=algo, linewidth=1.5, color='black')
        else:
            plt.plot(range(0, len(pred)), pred, label=algo, linewidth=.75)
    plt.xticks(range(0, len(ticks))[::365], ticks[::365])
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Dollars', fontsize=12)
    plt.title('MODEL')
    plt.legend(loc='upper right')
    plt.show()

def errors(algos, xTest, yTest):
    errors = {}
    for algo in algos:
        errors[algo] = round(mean_squared_error(algos[algo].predict(xTest), yTest), 2)
    return errors

def train(algos, xTrain, yTrain):
    for algo in algos:
        algos[algo].fit(xTrain, yTrain)

def populate(algos, X, lag):
    new = {}
    for algo in algos:
        model = algos[algo]
        pred = np.concatenate([np.zeros(lag), model.predict(X)])
        new[algo] = pred
    return new

algos = {
    'gradBoost': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls'),
    'knn': KNeighborsRegressor(n_neighbors=3, weights='distance'),
    'lasso': linear_model.Lasso(alpha=0.1),
    'elasticNet': linear_model.ElasticNet(alpha=0.1),
    'bayRidge': linear_model.BayesianRidge(),
    'SGDReg': SGDRegressor(),
    'tree': tree1.DecisionTreeRegressor(),
    'randForest': RandomForestRegressor(n_estimators=10, min_samples_leaf=2, max_depth=5),
    'extraTrees': ExtraTreesRegressor(n_estimators=10, min_samples_leaf=2, max_depth=5),
    'ridge': Ridge(alpha=1.0),
    'neuralMLP': MLPRegressor()
}

data = pullAndMerge('SP500', 'BAMLH0A0HYM2', 'DGS10', 'FEDFUNDS', 'T10Y2Y', 'T10YFF', 'USROA', 'USROE')

dates = data['DATE']

volatility = getFinalVolt(data['SP500'], regDepth=175, smoothDepth=105)[200:]

indicators = data.drop(['DATE', 'SP500'], axis=1).as_matrix()[200:]

X, y = setLag(indicators, volatility, lag=90)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=33)

# X_train, X_test = X[:2000], X[2000:]
# y_train, y_test = y[:2000], y[2000:]

train(algos, X_train, y_train)

errors = errors(algos, X_test, y_test)
print('Errors: %s' % errors)
print('\nError Average: %s' % np.mean(list(errors.values())))

filled = populate(algos, indicators, lag=90)

filled['Volatility'] = volatility

algosPlot(filled, dates)


#
#
# def opt():
#
#     regRange = range(100, 200, 5)
#     smoothRange = range(70, 110, 5)
#
#     t = len(regRange)*len(smoothRange)
#     result = np.zeros((t, 3))
#
#     i = 0
#     for reg in regRange:
#         for smooth in smoothRange:
#             print('\nG-SET\n')
#             print('Working reg=%s smooth=%s' % (reg, smooth))
#             volatility = getFinalVolt(data['SP500'], regDepth=reg, smoothDepth=smooth)[200:]
#             X, y = setLag(indicators, volatility, lag=90)
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#             train(algos, X_train, y_train)
#             error = np.mean(list(errors(algos, X_test, y_test).values()))
#             print('\t%s' % error)
#             print('\n G-WHAT?\n')
#             result[i, 0] = reg
#             result[i, 1] = smooth
#             result[i, 2] = error
#             i = i+1
#
#     return result
#
# result = opt()
# print(result[np.where(result[:, 2] == result[:, 2].min()), :])
