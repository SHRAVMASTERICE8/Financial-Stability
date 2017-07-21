import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

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
        data[i] = cleanToPandas('/Users/josephokeefe/Desktop/ResearchProject/Data3/%s.csv' % args[i])
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
            plt.plot(range(0, len(pred)), pred, label=algo, linewidth=1.75, color='black')
        else:
            plt.plot(range(0, len(pred)), pred, label=algo, linewidth=0.75)
    plt.xticks(range(0, len(ticks))[::1500], ticks[::1500])
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

def complexErrors(algos, X, y, trainDepth, predictionDepth, frequency):
    n_records = len(X) - predictionDepth
    results = np.zeros((len(algos), len(range(trainDepth, n_records, frequency))))
    b = 0
    for i in range(trainDepth, n_records, frequency):
        print('%s/%s' % (i, n_records))
        X_train, X_test = X[i-trainDepth:i], X[i:i + 90]
        y_train, y_test = y[i-trainDepth:i], y[i:i + 90]
        train(algos, X_train, y_train)
        errorList = errors(algos, X_test, y_test)
        g = 0
        for algo in algos:
            results[g, b] = errorList[algo]
            g = g + 1
        b = b + 1
    finalResult = {}
    g = 0
    for algo in algos:
        finalResult[algo] = np.mean(results[g, :])
        g = g + 1
    return finalResult

algos = {
    'elasticNet': linear_model.ElasticNet(alpha=0.9, l1_ratio=.1)
}

data = pullAndMerge('SP500', 'TWEXB', 'CPIHOSNS', 'A191RL1Q225SBEA', 'PSAVERT', 'T10YFF', 'DGS10', 'BAMLH0A0HYM2', 'T10Y2Y', 'FEDFUNDS', 'USROA', 'USROE', 'USSTHPI', 'STLFSI', 'NCBCMDPMVCE', 'MPRIME', 'CILACBQ158SBOG', 'INTDSRUSM193N', 'TERMCBAUTO48NS', 'TOTLL', 'BAMLH0A0HYM2EY', 'BAMLC0A0CM', 'RU3000VTR', 'TOTBKCR')

print(data.head())

dates = data['DATE']
num = data['SP500']

volatility = getFinalVolt(data['SP500'], regDepth=200, smoothDepth=200)[400:]

indicators = data.drop(['DATE', 'SP500'], axis=1).as_matrix()[400:]

X, y = setLag(indicators, volatility, lag=90)

print(complexErrors(algos, X, y, trainDepth=300, predictionDepth=90, frequency=300))
#
# def opt():
#
#     aRange = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     lRange = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#
#     t = len(aRange)*len(lRange)
#
#     result = np.zeros((3, t))
#
#     i = 0
#     for a in aRange:
#         for l in lRange:
#             print('%s %s' % (a, l))
#             algos = {
#                 'elasticNet': linear_model.ElasticNet(alpha=a, l1_ratio=l)
#             }
#
#             error = complexErrors(algos, X, y, 300, predictionDepth=90, frequency=300)
#             result[0, i] = a
#             result[1, i] - l
#             result[2, i] = error['elasticNet']
#             i = i + 1
#
#     return result
#
# result = opt()
#
# print(result[:, np.where(result[2, :] == result[2, :].min())])
