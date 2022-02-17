import os

path = 'C:\\ProgramData\\Anaconda3\\Lib\\site-packages\\tensorflow\\models\\images'
counter = 1
for f in os.listdir(path):
    suffix = f.split('.')[-1]
    if suffix == 'jpg' or suffix == 'png':
        new = '{}.{}'.format(str(counter), suffix)
        os.rename(path + f, path + new)
        counter = int(counter) + 1