import pandas as pd
nifty = pd.read_csv('NIFTY.csv')
niftyetf = pd.read_csv('ICICINIFTY.NS.csv')

print(nifty.head())
print(niftyetf.head())
