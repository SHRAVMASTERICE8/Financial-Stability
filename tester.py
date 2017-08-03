from Data import Data

ids = [('SP500', 'lin'),
       ('DGS10', 'lin'),
       ('DEXUSEU', 'lin'),
       ('PAYEMS', 'lin'),
       ('VXVCLS', 'lin')]

load = Data('2010-08-03', '2015-08-02', *ids)

load.download()

data = load.data

print(data)

print()

print(type(data['SP500'][5]))
