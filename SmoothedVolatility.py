import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing

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


data = cleanToPandas('/Users/josephokeefe/Desktop/ResearchProject/Data3/SP500.csv')

sp = data['SP500']
dates = data['DATE']
volt = volatilityByDay(sp, 200)
betterVolt = preprocessing.scale(rollingAverage(volt, 200))

sp = sp[400:]
dates = dates[400:]
volt = volt[400:]
betterVolt = betterVolt[400:]

x = range(0, len(sp))
#
# plt.figure(figsize=(10, 5))
# plt.plot(x, volt*5, label='Volatility')
# plt.plot(x, betterVolt*200+350, label='Smoothed and Standardized Volatility')
# plt.plot(x, sp, label='S&P 500')
# plt.xticks(x[::900], dates[::900])
# plt.xlabel('Date', fontsize=12)
# plt.ylabel('Index Value', fontsize=12)
# plt.title('Daily Volatility in the S&P 500\nMeasured as the Residuals of 200 Day Regressions')
# plt.legend(loc='upper right')
# plt.show()
#
#


def two_scales(ax1, time, data1, data2, c1, c2):
    """

    Parameters
    ----------
    ax : axis
        Axis to put two scales on

    time : array-like
        x-axis values for both datasets

    data1: array-like
        Data for left hand scale

    data2 : array-like
        Data for right hand scale

    c1 : color
        Color for line 1

    c2 : color
        Color for line 2

    Returns
    -------
    ax : axis
        Original axis
    ax2 : axis
        New twin axis
    """
    ax2 = ax1.twinx()

    ax1.plot(time, data1, color=c1, label='Volatility')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Standardized Volatiltiy')

    ax2.plot(time, data2, color=c2, label='S&P 500')
    ax2.set_ylabel('S&P 500 Index Value')
    plt.legend(loc='upper right')
    return ax1, ax2


# Create axes
fig, ax = plt.subplots()
ax1, ax2 = two_scales(ax, x, betterVolt, sp, 'C0', 'C1')
plt.xticks(x[::1000], dates[::1000])


# Change color of each axis
def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None

color_y_axis(ax1, 'C0')
color_y_axis(ax2, 'C1')
plt.title('Smoothed Volatility Training Data\nMeasured by the Residuals of 200 Day Regressions')
plt.show()
