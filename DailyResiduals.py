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
        est = m * i + b
        error = abs(est - y[i])
        values[i] = error.item(0)
    return values

data = cleanToPandas('/Users/josephokeefe/Desktop/ResearchProject/Data3/SP500.csv')

sp = data['SP500']
dates = data['DATE']
volt = volatilityByDay(sp, 200)

sp = sp[400:]
dates = dates[400:]
volt = volt[400:]

x = range(0, len(sp))

plt.figure(figsize=(10, 5))
plt.plot(x, volt*5, label='Volatility')
plt.plot(x, sp, label='S&P 500')
plt.xticks(x[::900], dates[::900])
plt.xlabel('Date', fontsize=12)
plt.ylabel('Index Value', fontsize=12)
plt.title('Daily Volatility in the S&P 500\nMeasured as the Residuals of 200 Day Regressions')
plt.legend(loc='upper right')
plt.show()
