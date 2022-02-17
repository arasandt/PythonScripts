import pandas as pd
df = pd.read_csv('D:\Arasan\Misc\GitHub\Others\input\yob2014.txt',header=None, names=['Name','Gender','Count'])
print(df.head())
print(df[df['Gender']=='M'].head(10))
print(df[df['Gender']=='F'].head(10))