#from pytube import Playlist
#pl = Playlist("https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN")
#pl.download_all()

"""
from pytube import YouTube
YouTube('http://youtube.com/watch?v=9bZkp7q19f0').streams.filter(progressive=True).first().download()
"""

import pandas as pd
from pytube import YouTube

df = pd.read_excel('MLPlaylist.xlsx')
df.drop(['Published Date','Title','Channel'], axis=1, inplace=True)
for index, row in df.iterrows():
    video = row['Video URL']
    print('Downloading {0} of {1} --> {2}'.format(index,len(df),video))
    YouTube(video).streams.filter(progressive=True).first().download()