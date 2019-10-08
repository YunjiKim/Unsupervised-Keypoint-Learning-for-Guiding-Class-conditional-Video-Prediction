from __future__ import division

import os.path as osp
import os
import tensorflow as tf
from scipy.io import loadmat
import random
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import math


def load_dataset(data_dir, subset):
  load_subset = 'train' if subset in ['train', 'val'] else 'test'
  with open(os.path.join(data_dir, 'penn_' + load_subset + '_set_add.txt'), 'r') as f:
    images = f.read().splitlines()

#   if subset in ['train', 'val']:
#     # put the last 10 percent of the training aside for validation
#     n_validation = int(round(0.1 * len(images)))
#     if subset == 'train':
#       images = images[:-n_validation]
#     elif subset == 'val':
#       images = images[-n_validation:]
#     else:
#       raise ValueError()

  return images
  

class PENNDataset(object):
  
  N_LANDMARKS = 13
  N_ACTION = 12

  def __init__(self, data_dir, subset, max_samples=None,
               image_size=[128, 128], order_stream=False, landmarks=False,
               tps=False, vertical_points=10, horizontal_points=10,
               rotsd=[0.0, 5.0], scalesd=[0.0, 0.1], transsd=[0.1, 0.1],
               warpsd=[0.001, 0.005, 0.001, 0.01],
               name='CelebADataset'):

    self._data_dir = data_dir
    self._order_stream = order_stream
    self._max_samples = max_samples
    self._images = load_dataset(self._data_dir, subset)
    print(subset+'set : ', len(self._images))


  def _get_sample_dtype(self):
    d =  {'image': tf.float32,
          'landmarks': tf.float32,
          'future_image': tf.float32,
          'future_landmarks': tf.float32,
          # 'box': tf.float32,
#           'box_': tf.float32
          }
    return d


  def _get_sample_shape(self):
    d = {'image': [128, 128, 3],
         'landmarks': [self.N_LANDMARKS, 2],
         'future_image': [128, 128, 3],
         'future_landmarks': [self.N_LANDMARKS, 2],
        #  'box': [128, 128, 3],
#          'box_': [4]
         }
    return d


  def random_filter(self, im, future_im):
    r_id = random.randint(0,9)
    if r_id == 0:
        im = im.filter(ImageFilter.DETAIL)
        future_im = future_im.filter(ImageFilter.DETAIL)
        return np.asarray(im), np.asarray(future_im)
    elif r_id == 1:
        im = im.filter(ImageFilter.EDGE_ENHANCE)
        future_im = future_im.filter(ImageFilter.EDGE_ENHANCE)
        return np.asarray(im), np.asarray(future_im)
    elif r_id == 2:
        im = im.filter(ImageFilter.SMOOTH)
        future_im = future_im.filter(ImageFilter.SMOOTH)
        return np.asarray(im), np.asarray(future_im)
    elif r_id == 3:
        im = im.filter(ImageFilter.SMOOTH_MORE)
        future_im = future_im.filter(ImageFilter.SMOOTH_MORE)
        return np.asarray(im), np.asarray(future_im)
    elif r_id == 4:
        im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
        future_im = future_im.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return np.asarray(im), np.asarray(future_im)
    elif r_id == 5:
        im = im.filter(ImageFilter.BLUR)
        future_im = future_im.filter(ImageFilter.BLUR)
        return np.asarray(im), np.asarray(future_im)
    elif r_id == 6:
        r_val = random.randint(0,50)
        im = ImageEnhance.Sharpness(im).enhance(r_val*0.1)
        future_im = ImageEnhance.Sharpness(future_im).enhance(r_val*0.1)
        return np.asarray(im), np.asarray(future_im)
    elif r_id == 7:
        r_val = random.randint(7,20)
        im = ImageEnhance.Brightness(im).enhance(r_val*0.1)
        future_im = ImageEnhance.Brightness(future_im).enhance(r_val*0.1)
        return np.asarray(im), np.asarray(future_im)
    elif r_id == 8:
        r_val = random.randint(0,50)
        im = ImageEnhance.Color(im).enhance(r_val*0.1)
        future_im = ImageEnhance.Color(future_im).enhance(r_val*0.1)
        return np.asarray(im), np.asarray(future_im)
    else:
        r_val = random.randint(7,30)
        im = ImageEnhance.Contrast(im).enhance(r_val*0.1)
        future_im = ImageEnhance.Contrast(future_im).enhance(r_val*0.1)
        return np.asarray(im), np.asarray(future_im)


  def _proc_im_pair(self, inputs):
    with tf.name_scope('proc_im_pair'):
      inputs = {'image': inputs['image']*2.0-1.0, 'future_image': inputs['future_image']*2.0-1.0,
                'landmarks': (inputs['landmarks']-64.0)/64.0, 'future_landmarks': (inputs['future_landmarks']-64.0)/64.0}
    return inputs


  def _get_image_test(self, idx):
        
    img_path, n_act = self._images[idx].split()
    file_len = len(os.listdir(osp.join(self._data_dir, img_path)))

    image = Image.open(osp.join(self._data_dir, img_path, '{:06d}'.format(1)+'.jpg'))
    future_image = Image.open(osp.join(self._data_dir, img_path, '{:06d}'.format(11)+'.jpg'))
        
    mat_ = loadmat(osp.join(self._data_dir, img_path.replace('frames', 'labels') + '.mat'))
    keypoints = np.concatenate([np.expand_dims(mat_['x'], axis=-1), np.expand_dims(mat_['y'], axis=-1)], axis=-1)
    landmarks = keypoints[0, :, :]
    future_landmarks = keypoints[10, :, :]
    # box = mat_['bbox'][10, :]
    
    w,h = image.size
    if w>h:
      w_larger = 1
    else:
      w_larger = 0

    if w_larger:

      ## resize
      ratio = h/128.0
        
      image = image.resize([int(w/ratio), int(h/ratio)])
      future_image = future_image.resize([int(w/ratio), int(h/ratio)])
    
      ## centercrop
      ox, oy = image.size
      ox /= 2.0
    
      image = image.crop((ox-64, 0, ox+64, 128))
      future_image = future_image.crop((ox-64, 0, ox+64, 128))
    
      image = np.asarray(image)
      future_image = np.asarray(future_image)
    
      landmarks = landmarks/ratio
      future_landmarks = future_landmarks/ratio
    
      ## point position calibration and apply horizontal flip
      landmarks[:,0] = landmarks[:,0] - (ox-64)
      future_landmarks[:,0] = future_landmarks[:,0] - (ox-64)
        
      # left, top, right, bottom = box/ratio
      # left -= (ox-64)
      # right -= (ox-64)
        
      # left_, right_ = np.min(future_landmarks[:,0]), np.max(future_landmarks[:,0])
      # top_, bottom_ = np.min(future_landmarks[:,1]), np.max(future_landmarks[:,1])
        
    else:

      ## resize
      ratio = w/128.0
        
      image = image.resize([int(w/ratio), int(h/ratio)])
      future_image = future_image.resize([int(w/ratio), int(h/ratio)])
    
      ## centercrop
      ox, oy = image.size
      oy /= 2.0        
        
      image = image.crop((0, oy-64, 128, oy+64))
      future_image = future_image.crop((0, oy-64, 128, oy+64))
        
      image = np.asarray(image)
      future_image = np.asarray(future_image)
        
      landmarks = landmarks/ratio
      future_landmarks = future_landmarks/ratio
    
      landmarks[:,1] = landmarks[:,1] - (oy-64)
      future_landmarks[:,1] = future_landmarks[:,1] - (oy-64)
    
    #   left, top, right, bottom = box/ratio
    #   top -= (oy-64)
    #   bottom -= (oy-64)
        
    #   left_, right_ = np.min(future_landmarks[:,0]), np.max(future_landmarks[:,0])
    #   top_, bottom_ = np.min(future_landmarks[:,1]), np.max(future_landmarks[:,1])

    # left, top, right, bottom = np.clip(np.asarray([left, top, right, bottom]), 0, 127)

    # box_filter = np.zeros((128, 128, 3), dtype = np.float32)
    # box_filter[int(top):int(bottom)+1, int(left):int(right)+1, :] = 1.0
        
    inputs = {'image': image/255.0, 'future_image': future_image/255.0, \
              'landmarks': landmarks, 'future_landmarks': future_landmarks}
    return inputs


  def _get_image(self, idx):

    img_path, n_act = self._images[idx].split()
    file_len = len(os.listdir(osp.join(self._data_dir, img_path)))

    im_idx = random.randint(0, file_len-1)
    rand_interval = random.randint(8,11)
    fu_im_idx = (im_idx+rand_interval)%file_len

    rand1 = random.randrange(-10,11)

    ## load images and add randomness
    image = Image.open(osp.join(self._data_dir, img_path, '{:06d}'.format(im_idx+1)+'.jpg'))
    future_image = Image.open(osp.join(self._data_dir, img_path, '{:06d}'.format(fu_im_idx+1)+'.jpg'))

    ## load joint and box information
    mat_ = loadmat(osp.join(self._data_dir, img_path.replace('frames', 'labels') + '.mat'))
    keypoints = np.concatenate([np.expand_dims(mat_['x'], axis=-1), np.expand_dims(mat_['y'], axis=-1)], axis=-1)
    landmarks = keypoints[im_idx, :, :]
    future_landmarks = keypoints[fu_im_idx, :, :]
    # box = mat_['bbox'][fu_im_idx, :]
    
    ## add randomness
    w,h = image.size
    if w>h:
      w_larger = 1
    else:
      w_larger = 0
    
    ## rotate image and the points
    image = image.rotate(rand1)
    future_image = future_image.rotate(rand1)
    
    ox, oy = image.size
    ox /= 2.0
    oy /= 2.0    
    
    qx = ox + math.cos(math.radians(-rand1)) * (landmarks[:,0] - ox) - math.sin(math.radians(-rand1)) * (landmarks[:,1] - oy)
    qy = oy + math.sin(math.radians(-rand1)) * (landmarks[:,0] - ox) + math.cos(math.radians(-rand1)) * (landmarks[:,1] - oy)
    landmarks = np.concatenate([np.expand_dims(qx, -1),np.expand_dims(qy,-1)], axis=-1)
    
    qx = ox+math.cos(math.radians(-rand1))*(future_landmarks[:,0]-ox)-math.sin(math.radians(-rand1))*(future_landmarks[:,1]-oy)
    qy = oy+math.sin(math.radians(-rand1))*(future_landmarks[:,0]-ox)+math.cos(math.radians(-rand1))*(future_landmarks[:,1]-oy)
    future_landmarks = np.concatenate([np.expand_dims(qx, -1),np.expand_dims(qy,-1)], axis=-1)
    
    
    if w_larger:

      ratio = h/128.0

      image = image.resize([int(w/ratio), int(h/ratio)])
      future_image = future_image.resize([int(w/ratio), int(h/ratio)])
    
      landmarks = landmarks/ratio
      future_landmarks = future_landmarks/ratio
    
      min_ = min(np.min(landmarks[:,0]), np.min(future_landmarks[:,0]))

      rand1 = random.randint(0, int(max(0,min(min_, w/ratio-128))))
      rand2 = random.randint(0,1)    
        
      image = image.crop((rand1, 0, rand1+128, 128))
      future_image = future_image.crop((rand1, 0, rand1+128, 128))

      if rand2:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        future_image = future_image.transpose(Image.FLIP_LEFT_RIGHT)

      image, future_image = self.random_filter(image, future_image)

      ## point position calibration and apply horizontal flip
      landmarks[:,0] = landmarks[:,0] - rand1
      future_landmarks[:,0] = future_landmarks[:,0] - rand1

      if rand2:
        landmarks[:,0] = 128.0 - landmarks[:,0]
        future_landmarks[:,0] = 128.0 - future_landmarks[:,0]
        
      ## resize bbox coordinate
      # left, top, right, bottom = box/ratio
      # left -= rand1
      # right -= rand1
        
      # if rand2:
      #   tem = left
      #   left = 128.0 - right
      #   right = 128.0 - tem
        
      # left_, right_ = np.min(future_landmarks[:,0]), np.max(future_landmarks[:,0])
      # top_, bottom_ = np.min(future_landmarks[:,1]), np.max(future_landmarks[:,1])

    else:

      ratio = w/128.0

      image = image.resize([int(w/ratio), int(h/ratio)])
      future_image = future_image.resize([int(w/ratio), int(h/ratio)])

      ## resize points
      landmarks = landmarks/ratio
      future_landmarks = future_landmarks/ratio
        
      min_ = min(np.min(landmarks[:,1]), np.min(future_landmarks[:,1]))
    
      ## add randomness [crop, rotate]
      rand1 = random.randint(0, int(max(0,min(min_, h/ratio-128))))
      rand2 = random.randint(0,1)
    
      image = image.crop((0, rand1, 128, rand1+128))
      future_image = future_image.crop((0, rand1, 128, rand1+128))

      if rand2:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        future_image = future_image.transpose(Image.FLIP_LEFT_RIGHT)

      image, future_image = self.random_filter(image, future_image)

      ## point position calibration and apply horizontal flip
      landmarks[:,1] = landmarks[:,1] -rand1
      future_landmarks[:,1] = future_landmarks[:,1] - rand1

      if rand2:
        landmarks[:,0] = 128.0 - landmarks[:,0]
        future_landmarks[:,0] = 128.0 - future_landmarks[:,0]
        
      ## resize bbox coordinate
      # left, top, right, bottom = box/ratio
      # top -= rand1
      # bottom -= rand1
        
      # if rand2:
      #   tem = left
      #   left = 128.0 - right
      #   right = 128.0 - tem
        
      # left_, right_ = np.min(future_landmarks[:,0]), np.max(future_landmarks[:,0])
      # top_, bottom_ = np.min(future_landmarks[:,1]), np.max(future_landmarks[:,1])


    # left, top, right, bottom = np.clip(np.asarray([left, top, right, bottom]), 0, 127)

    # box_filter = np.zeros((128, 128, 3), dtype = np.float32)
    # box_filter[int(top):int(bottom)+1, int(left):int(right)+1, :] = 1.0
      
    # n_act_enc = np.zeros(self.N_ACTION)
    # n_act_enc[int(n_act)] = 1
      
    inputs = {'image': image/255.0, 'future_image': future_image/255.0,\
              'landmarks': landmarks, 'future_landmarks': future_landmarks}
    return inputs


  def _get_random_image(self):
    idx = np.random.randint(len(self._images))
    return self._get_image(idx)


  def _get_ordered_stream(self):
    for i in range(len(self._images)):
      yield self._get_image_test(i)


  def sample_image_pair(self):
    f_sample = self._get_random_image
    if self._order_stream:
      g = self._get_ordered_stream()
      f_sample = lambda: next(g)
    max_samples = float('inf')
    if self._max_samples is not None:
      max_samples = self._max_samples
    i_samp = 0
    while i_samp < max_samples:
      yield f_sample()
      if self._max_samples is not None:
          i_samp += 1


  def get_dataset(self, batch_size, repeat=False, shuffle=False,
                  num_preprocess_threads=12, keep_aspect=True, prefetch=True):
    """
    Returns a tf.Dataset object which iterates over samples.
    """
    def sample_generator():
      return self.sample_image_pair()

    sample_dtype = self._get_sample_dtype()
    sample_shape = self._get_sample_shape()
    dataset = tf.data.Dataset.from_generator(sample_generator, sample_dtype, sample_shape)
    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(2000)

    dataset = dataset.map(self._proc_im_pair, num_parallel_calls=num_preprocess_threads)
    dataset = dataset.batch(batch_size)

    if prefetch:
      dataset = dataset.prefetch(1)
    return dataset
