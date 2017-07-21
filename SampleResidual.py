import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

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

        if i == 3679:
            m1 = m
            b1 = b

        est = m * i + b
        error = abs(est - y[i])
        values[i] = error.item(0)
    return values, m1, b1

data = cleanToPandas('/Users/josephokeefe/Desktop/ResearchProject/Data3/SP500.csv')

sp = data['SP500']
dates = data['DATE']
x = range(0, len(sp))

volt, m, b = volatilityByDay(sp, 200)



plt.figure(figsize=(6, 5))
plt.plot(x, sp, label='S&P 500', color='C1')
plt.plot(x[3479:3679], m*x[3479:3679]+b, label='Sample Regression')
plt.vlines(3679, sp[3679], m*x[3679]+b, color='black',
           linewidth=1, label='Sample Residual', linestyles='--')
plt.plot(x[3679], sp[3679], 'ro', color='black', markersize=2)
plt.plot(x[3679], m*x[3679]+b, 'ro', color='black', markersize=2)
plt.xticks(x[::200], dates[::200])
plt.xlabel('Date', fontsize=12)
plt.ylabel('Index Value', fontsize=12)
plt.title('Calculating Daily Volatility\nMeasured as the Residuals of 200 Day Regressions')
plt.legend(loc='upper left')
plt.show()
