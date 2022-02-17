import shutil
import os
from multiprocessing import Pool 
from PIL import Image
import time as time

SIZE = (75,75)
SAVE_DIRECTORY = 'thumbs'

def get_image_paths(folder):
  return (os.path.join(folder, f) for f in os.listdir(folder) if 'jpg' in f)

def create_thumbnail(filename): 
  im = Image.open(filename)
  im.thumbnail(SIZE, Image.ANTIALIAS)
  base, fname = os.path.split(filename) 
  save_path = os.path.join(base, SAVE_DIRECTORY, fname)
  im.save(save_path)

if __name__ == '__main__':
  folder = os.path.abspath('D:\Arasan\Misc\GitHub\Others\input\ImagesFolder')
  if os.path.exists(os.path.join(folder, SAVE_DIRECTORY)):
      shutil.rmtree(os.path.join(folder, SAVE_DIRECTORY))
  else:
      os.mkdir(os.path.join(folder, SAVE_DIRECTORY))

  images = get_image_paths(folder)
  #print(images)
  t = time.time()
# =============================================================================
#   for image in images: 
#     create_thumbnail(image)
# =============================================================================
  
  pool = Pool(12)
  pool.map(create_thumbnail, images)
  pool.close() 
  pool.join()
  print(time.time()-t)