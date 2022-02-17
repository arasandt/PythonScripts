import sys
import numpy as np
from datetime import datetime, timedelta
from dateutil import tz
import pandas as pd


files = ['Juice Bar 20fps.lnr','Lenel_video_LNR_Format.lnr','Met1MDFEntrance.lnr']

def getFiletime(dt):
    microseconds = int(dt, 16) / 10
    seconds, microseconds = divmod(microseconds, 1000000)
    days, seconds = divmod(seconds, 86400)
    return datetime(1601, 1, 1) + timedelta(days, seconds, microseconds)

# =============================================================================
# def printStartTimestamp(f):
#     f.seek(32, 1)
#     data = f.read(8).hex()
#     utc = ''.join([data[i:i+2] for i in range(0, len(data), 2)][::-1])
#     from_zone = tz.gettz('UTC')
#     to_zone = tz.gettz('America/New_York')    
#     utc = getFiletime(utc).replace(tzinfo=from_zone)
#     est = utc.astimezone(to_zone)
#     print(est.strftime('%m/%d/%Y %I:%M:%S %p %Z'))
# 
# =============================================================================
def printStartTimestamp(f):
    data = f[:16]
    print(data)
    utc = ''.join([data[i:i+2] for i in range(0, len(data), 2)][::-1])
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/New_York')    
    utc = getFiletime(utc).replace(tzinfo=from_zone)
    est = utc.astimezone(to_zone)
    return est.strftime('%m/%d/%Y %I:%M:%S %p %Z')


tbcol=['Line#',*files,'Match']

df = pd.DataFrame(columns=tbcol)

fileseek=[]                           
for i in list(map(lambda x: '..\\VideoCapture\\' + x, files)):
    binary_file = open(i, "rb")
    fileseek.append(binary_file)

for line in range(13):
    rowvalue = list()
    rowvalue.append(line+1)
    for i in range(len(files)):
        rowvalue.append(fileseek[i].read(16).hex())
    if len(set(rowvalue)) > 2:
        rowvalue.append('False')
    else:
        rowvalue.append('True')
    #print(rowvalue)
    df = df.append(dict(zip(tbcol,rowvalue)),ignore_index=True)
pd.set_option('display.max_columns', 7)
df = df[df['Match'] == 'False'].iloc[:,:-1]
print(df.iloc[1].head())
df['StartTimestamp'] = df.iloc[1].apply(printStartTimestamp)
print(df.head())

    #printStartTimestamp(binary_file)
    
    
    
for i in fileseek:
    i.close()
        
        
        



        
