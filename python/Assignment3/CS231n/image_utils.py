"""
Platform: win10 64-bits
python: 3.6
"""
import urllib3
import os, tempfile 
import numpy as np 
from scipy.misc import imread 
from CS231n.fast_layers import conv_forward_fast 

"""
Utility functions used for viewing and processing images. 
"""
def blur_image(x):
  """
  Image bluring operation, to be used as a regularizer for image generation. 
  
  Inputs:
    - x: Image data of shape (N, 3, H, W)
  Returns:
    - X_blur: blurred version of x, of shape (N, 3, H, W)
  """
  W_blur = np.zeros((3, 3, 3, 3)) # filter number, channel number, height, width
  b_blur = np.zeros(3)            # filter number
  blur_param = {'stride':1, 'pad': 1}
  for i in range(3):
    w_blur[i,i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
  
  w_blur /= 200.0
  return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]

def preprocess_image(img, mean_img, mean='image'):
  """
  Convert to float, transepose, and subtract mean pixel
  
  Input:
  - img: (H, W, 3)
  
  Returns:
  - (1, 3, H, W)
  """
  if mean == 'image':
    mean = mean_img
  elif mean == 'pixel':
    mean = mean_img.mean(axis=(1, 2), keepdims=True)
  elif mean == 'none':
    mean = 0
  else:
    raise ValueError('mean must be image or pixel or none')
  return img.astype(np.float32).transpose(2, 0, 1)[None] - mean

def image_from_url(url):
  """
  Read an image from a URL. Return a numpy array with the pixel data. 
  We write the image to a temporary file then read it back. 
  """
  http = urllib3.PoolManager()
  try:
    r = http.request('GET', url)
    _, fname = tempfile.mkstemp()
    with open(fname, 'wb') as ff:
      ff.write(r.data)
    img = imread(fname)
    os.remove(fname)
    return img
  except urllib2.URLError as e:
    print('URL Error: ', e.reason, url)
  except urllib2.HTTPError as e:
    print('HTTP Error: ', e.code, url)