import fred
import pandas as pd
import numpy as np

class Data:

    def __init__(self, obsStart, obsEnd, *args):
        self.tags = []
        self.obsStart = obsStart
        self.obsEnd = obsEnd
        fred.key('4cfe54d7bef3609a5f3eae6e5d87f790')
        for i in range(0, len(args)):
            self.tags.append(args[i])

    @classmethod
    def generateDates(cls, start, end):
        dateRange = pd.date_range(start, end, freq="D").values
        list = []
        for date in dateRange:
            list.append(str(date.astype('M8[D]')))
        return pd.DataFrame(list, columns=['DATE'])

    def download(self):
        self.dfList = []
        for tag in self.tags:
            self.obsList = []
            obsRaw = fred.observations(tag[0], observation_start=self.obsStart, observation_end=self.obsEnd, units=tag[1])
            for obs in obsRaw['observations']:
                self.obsList.append((obs['date'], obs['value']))
            df =  pd.DataFrame(self.obsList, columns=['DATE', tag[0]])
            self.dfList.append(df)
        self.data = self.mergeByDate(self.dfList)
        self.data = self.data.apply(pd.to_numeric, errors='ignore')

    def mergeByDate(self, dfs):
        result = self.generateDates(self.obsStart, self.obsEnd)
        for i in range(0, len(dfs)):
            result = pd.merge(result, dfs[i], on='DATE', how='left')
        result = result.replace('.', np.NaN)
        result = result.fillna(method='ffill')
        result = result.fillna(method='bfill')
        return result