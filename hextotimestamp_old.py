from datetime import datetime, timedelta
from dateutil import tz
import os

data = ['1d4add6e1d9139c']


def getFiletime(dt):
        microseconds = int(dt, 16) / 10
        seconds, microseconds = divmod(microseconds, 1000000)
        days, seconds = divmod(seconds, 86400)
        return datetime(1601, 1, 1) + timedelta(days, seconds, microseconds)


from_zone = tz.gettz('UTC')
to_zone = tz.gettz('America/New_York')    
filetime = getFiletime(data[0])
utc = filetime
utc = utc.replace(tzinfo=from_zone)
est = utc.astimezone(to_zone)
print(est.strftime('%m/%d/%Y %I:%M:%S %p %Z'))