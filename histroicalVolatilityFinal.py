import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


def cleanToPandas(filePath):
    cleaned = pd.read_csv(filePath, index_col=None, na_values=['.'])
    cleaned = cleaned.fillna(method='ffill')
    return cleaned


def formatForRegression(df):
    x = df.index.values
    x = np.matrix(x).T
    y = df['SP500'].values
    y = np.matrix(y).T
    return x, y


def volatilityByDay(df, depth):
    x, y = formatForRegression(df)
    regr = linear_model.LinearRegression()
    values = np.zeros(len(y))

    for i in range(depth+1, len(y)):
        regr.fit(x[i - depth:i], y[i - depth:i])
        m = regr.coef_[0]
        b = regr.intercept_
        est = m * i + b
        error = abs(est - y[i])
        values[i] = error.item(0) * 10

    return values


def rollingAverage(data, depth):
    rolling = pd.rolling_mean(pd.DataFrame(data), depth)
    rolling = rolling.fillna(0)
    rolling = rolling[0].values
    return rolling


def plotAll(original, volt, voltAv):
    x, y = formatForRegression(original)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='r', label='S&P 500 (Market Value)')
    plt.plot(x, volt, label='Volatility (Residual of 150 Day Regression)')
    plt.plot(x, voltAv, linewidth=3.0, label='Volatility (100 Day Moving Average)')
    plt.xticks(x[::365], original['DATE'][::365])
    plt.title('S&P 500 Volatility\nMeasured as the Predictive Error of 150 Day Regressions')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Dollars', fontsize=12)
    plt.legend(loc='upper right')
    plt.ylim([0, 3200])
    plt.show()


sp = cleanToPandas('/Users/josephokeefe/Desktop/SP500.csv')
volatility = volatilityByDay(sp, 150)
volatilityAverage = rollingAverage(volatility, 100)

plotAll(sp, volatility, volatilityAverage)
