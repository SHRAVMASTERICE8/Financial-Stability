import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 5, 5

def cleanToPandas(filePath):
    cleaned = pd.read_csv(filePath, index_col=None, na_values=['.'])
    cleaned = cleaned.fillna(method='ffill')
    cleaned = cleaned.fillna(method='bfill')
    return cleaned

def percentChange(series):
    values = np.zeros(len(series))
    for i in range(1, len(series)):
        values[i] = abs((series[i]-series[i-1])/(series[i-1]))

    return values


data = cleanToPandas('/Users/josephokeefe/Desktop/ResearchProject/Data3/SP500.csv')

sp = data['SP500']
dates = data['DATE']
x = range(0, len(sp))

pChange = percentChange(sp)


plt.figure(figsize=(5, 5))
plt.plot(x, pChange, label='Smoothed Volatility')
plt.plot(x, sp, label='S&P 500')
plt.xticks(x[::300], dates[::300])
plt.xlabel('Date', fontsize=12)
plt.ylabel('Percent', fontsize=12)
plt.title('S&P 500 Volatility\nMeasured in terms of Daily Percent Change')
plt.legend(loc='upper left')
plt.show()




#
# def two_scales(ax1, time, data1, data2, c1, c2):
#     """
#
#     Parameters
#     ----------
#     ax : axis
#         Axis to put two scales on
#
#     time : array-like
#         x-axis values for both datasets
#
#     data1: array-like
#         Data for left hand scale
#
#     data2 : array-like
#         Data for right hand scale
#
#     c1 : color
#         Color for line 1
#
#     c2 : color
#         Color for line 2
#
#     Returns
#     -------
#     ax : axis
#         Original axis
#     ax2 : axis
#         New twin axis
#     """
#     ax2 = ax1.twinx()
#
#     ax1.plot(time, data1, color=c1)
#     ax1.set_xlabel('Date')
#     ax1.set_ylabel('Absolute Value of Daily Percent Change')
#
#     ax2.plot(time, data2, color=c2)
#     ax2.set_ylabel('S&P 500 Index Value')
#     return ax1, ax2
#
#
# # Create axes
# fig, ax = plt.subplots()
# ax1, ax2 = two_scales(ax, x, pChange, sp, 'C0', 'C1')
# plt.xticks(x[::900], dates[::900])
#
# # Change color of each axis
# def color_y_axis(ax, color):
#     """Color your axes."""
#     for t in ax.get_yticklabels():
#         t.set_color(color)
#     return None
#
# color_y_axis(ax1, 'C0')
# color_y_axis(ax2, 'C1')
# plt.title('S&P 500 Volatility\n Measured as the Absolute Value of Daily Percent Change')
# plt.show()
