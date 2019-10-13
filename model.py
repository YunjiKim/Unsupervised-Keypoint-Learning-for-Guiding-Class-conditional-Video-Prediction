from __future__ import division

import tensorflow as tf
import numpy as np
import time
from collections import defaultdict

import copy
from datetime import datetime
import os.path as osp
import os


#######################

def get_random_color(pastel_factor = 0.5):
  return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
  return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
  max_distance = None
  best_color = None
  for i in range(0,100):
    color = get_random_color(pastel_factor = pastel_factor)
    if not existing_colors:
      return color
    best_distance = min([color_distance(color,c) for c in existing_colors])
    if not max_distance or best_distance > max_distance:
      max_distance = best_distance
      best_color = color
  return best_color

def get_n_colors(n, pastel_factor=0.9):
  colors = []
  for i in range(n):
    colors.append(generate_new_color(colors,pastel_factor = 0.9))
  return colors

def colorize_landmark_maps(self, maps):
  n_maps = maps.shape.as_list()[-1]
  # get n colors:
  if self.colors == None:
    self.colors = get_n_colors(n_maps, pastel_factor=0.0)
  hmaps = [tf.expand_dims(maps[..., i], axis=3) * np.reshape(self.colors[i], [1, 1, 1, 3])
          for i in range(n_maps)]
  return tf.reduce_max(hmaps, axis=0)

def get_gaussian_maps(mu, shape_hw, inv_std):
  mu_x, mu_y = mu[:, :, 0:1], mu[:, :, 1:2]
  y = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[0]))
  x = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[1]))
  mu_y, mu_x = tf.expand_dims(mu_y, -1), tf.expand_dims(mu_x, -1)
  y = tf.reshape(y, [1, 1, shape_hw[0], 1])
  x = tf.reshape(x, [1, 1, 1, shape_hw[1]])
  g_y = tf.square(y - mu_y)
  g_x = tf.square(x - mu_x)
  dist = (g_y + g_x) * inv_std**2
  g_yx = tf.transpose(tf.exp(-dist), perm=[0, 2, 3, 1])
  return g_yx

def get_coord(other_axis, axis_size):
  # get "x-y" coordinates:
  g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
  g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
  coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size)) # W
  coord_pt = tf.reshape(coord_pt, [1, axis_size, 1])
  g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
  return g_c, g_c_prob

#######################

def conv(x, channels, kernel=4, stride=2, pad=0, use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        x = tf.layers.conv2d(inputs=x, filters=channels, padding='same',
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)
        return x

def batch_norm(x, train_mode, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        epsilon=1e-05,
                                        center=True, scale=True,
                                        scope=scope, is_training=train_mode)

def lstm_model(self, layers):
  lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(units, state_is_tuple=True) for units in layers]
  lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0) for cell in lstm_cells]
  stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=True)
  return stacked_lstm

def linear(input_,
           output_size,
           name,
           stddev=0.02,
           bias_start=0.0,
           reuse=False,
           with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name, reuse=reuse):
    matrix = tf.get_variable(
        "Matrix", [shape[1], output_size],
        tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable(
        "bias", [output_size], initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
    
#######################

def image_encoder(self, x):
  with tf.variable_scope('image_encoder'):
    filters = self.config.n_filters
    block_features = [x]
    x = conv(x, filters, kernel=7, stride=1, scope='conv_1')
    x = batch_norm(x, self.train_mode, scope='b_norm_1')
    x = tf.nn.relu(x)
    x = conv(x, filters, kernel=3, stride=1, scope='conv_2')
    x = batch_norm(x, self.train_mode, scope='b_norm_2')
    x = tf.nn.relu(x)
    block_features.append(x)
    for i in range(3):
      filters *= 2
      x = conv(x, filters, kernel=3, stride=2, scope='conv_%d'%(i*2+3))
      x = batch_norm(x, self.train_mode, scope='b_norm_%d'%(i*2+3))
      x = tf.nn.relu(x)
      x = conv(x, filters, kernel=3, stride=1, scope='conv_%d'%(i*2+4))
      x = batch_norm(x, self.train_mode, scope='b_norm_%d'%(i*2+4))
      x = tf.nn.relu(x)
      block_features.append(x)
    return block_features

def pose_encoder(self, x, filters=32, final_res=128):
  with tf.variable_scope('pose_encoder', reuse=tf.AUTO_REUSE):
    block_features = self.encoder(x)
    x = block_features[-1]
    filters = self.config.n_filters /2 * 8
    size = x.shape.as_list()[1:3]
    conv_id = 1
    for i in range(4):
      if i > 0:
          x = conv(tf.concat([x, block_features[-1*(i+1)]], axis=-1), filters, kernel=3, stride=1, scope='conv_%d_0'%conv_id)
      else:
          x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_0'%conv_id)
      x = batch_norm(x, self.train_mode, scope='b_norm_%d_0'%conv_id)
      x = tf.nn.relu(x)
      x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_1'%conv_id)
      x = batch_norm(x, self.train_mode, scope='b_norm_%d_1'%conv_id)
      x = tf.nn.relu(x)
      if size[0]==final_res:
        x = conv(x, self.config.n_maps, kernel=1, stride=1)
        break
      else:
        x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_0'%(conv_id+1))
        x = batch_norm(x, self.train_mode, scope='b_norm_%d_0'%(conv_id+1))
        x = tf.nn.relu(x)
        x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_1'%(conv_id+1))
        x = batch_norm(x, self.train_mode, scope='b_norm_%d_1'%(conv_id+1))
        x = tf.nn.relu(x)
        x = tf.image.resize_images(x, [2 * s for s in size])
      size = x.shape.as_list()[1:3]
      conv_id += 2
      if filters >= 8: filters /= 2
    xshape = x.shape.as_list()
    gauss_y, gauss_y_prob = get_coord(2, xshape[1])  # B,NMAP
    gauss_x, gauss_x_prob = get_coord(1, xshape[2])  # B,NMAP
    gauss_mu = tf.stack([gauss_x, gauss_y], axis=2)
    return gauss_mu

def translator(self, x, final_res=128):
  with tf.variable_scope('translator'):
    filters = self.config.n_filters * 8
    size = x.shape.as_list()[1:3]
    conv_id = 1
    while size[0] <= final_res:
      x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_0'%conv_id)
      x = batch_norm(x, self.train_mode, scope='b_norm_%d_0'%conv_id)
      x = tf.nn.relu(x)
      x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_1'%conv_id)
      x = batch_norm(x, self.train_mode, scope='b_norm_%d_1'%conv_id)
      x = tf.nn.relu(x)
      if size[0]==final_res:
        crude_output = conv(x, 3, kernel=3, stride=1, scope='conv_%d_0'%(conv_id+1))
        mask = conv(x, 1, kernel=3, stride=1, scope='conv_%d_1'%(conv_id+1))
        mask = tf.nn.sigmoid(mask)
        break
      else:
        x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_0'%(conv_id+1))
        x = batch_norm(x, self.train_mode, scope='b_norm_%d_0'%(conv_id+1))
        x = tf.nn.relu(x)
        x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_1'%(conv_id+1))
        x = batch_norm(x, self.train_mode, scope='b_norm_%d_1'%(conv_id+1))
        x = tf.nn.relu(x)
        x = tf.image.resize_images(x, [2 * s for s in size])
      size = x.shape.as_list()[1:3]
      conv_id += 2
      if filters >= 8: filters /= 2
  return crude_output, mask

def vae_encoder(self, x, f_pt, act_code):     
  with tf.variable_scope('vae_encoder'):
    cell = self.lstm_model(self.cell_info)
    state = cell.zero_state(tf.shape(x)[0], tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(cell, x, initial_state=state, dtype=tf.float32)
    logit = tf.contrib.layers.fully_connected(tf.concat([outputs[:,-1,:], f_pt, act_code], axis = -1), self.vae_dim*2)
    mu = logit[:, :self.vae_dim]
    stddev = logit[:, self.vae_dim:]
    return mu, stddev

def decoder(self, input_, name='decoder'):
  out = 
  return 
  
def vae_decoder(self, x, f_pt, act_code):
  with tf.variable_scope('vae_decoder', reuse=tf.AUTO_REUSE):
    cell = self.lstm_model(self.cell_info)
    decoder = tf.group(linear(input_, self.config.n_maps*2), tf.tanh(out))
    state = cell.zero_state(tf.shape(x)[0], tf.float32)
    input_ = tf.contrib.layers.fully_connected(tf.concat([x, f_pt, act_code], axis = -1), 32)
    empty_input = tf.zeros_like(input_)
    outputs = []
    output, state = cell(input_, state)
    outputs.append(tf.expand_dims(decoder(output), axis = 1))
    for i in range(31):
      output, state = cell(empty_input, state)
      outputs.append(tf.expand_dims(decoder(output), axis = 1))
    outputs = tf.concat(outputs, axis = 1) 
    return outputs

def seq_discr(self, x):
  with tf.variable_scope('seq_discr', reuse=tf.AUTO_REUSE):
    cell = self.lstm_model([1024,1024])
    state = cell.zero_state(tf.shape(x)[0], tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(cell, x, initial_state=state, dtype=tf.float32)
    logit = tf.contrib.layers.fully_connected(outputs, 1)
    return logit[:,-1,:]  
 
def img_discr(self, x):
  with tf.variable_scope('img_discr', reuse=tf.AUTO_REUSE):
    channel = 64
    x = conv(x, channel, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_0')
    x = tf.nn.leaky_relu(x, 0.01)
    for i in range(1, 6):
      x = conv(x, channel * 2, kernel=4, stride=2, pad=1, use_bias=True, scope='conv_' + str(i))
      x = tf.nn.leaky_relu(x, 0.01)
      channel = channel * 2
    logit = conv(x, channels=1, kernel=3, stride=1, pad=1, use_bias=False, scope='D_logit')
    return logit

#######################
  
class Vgg19:
  def __init__(self, vgg19_npy_path=None):
    if vgg19_npy_path is None:
      path = os.path.join('/workspace/imm/vgg19.npy')
      vgg19_npy_path = path
    self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()

  def build(self, rgb):
    start_time = time.time()
    print("build model started")
    # Convert RGB to BGR
    VGG_MEAN = [103.939, 116.779, 123.68]
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
    bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]])
    self.conv1_1 = self.conv_layer(bgr, "conv1_1")
    self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
    self.pool1 = self.max_pool(self.conv1_2, 'pool1')
    self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
    self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
    self.pool2 = self.max_pool(self.conv2_2, 'pool2')
    self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
    self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
    self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
    self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
    self.pool3 = self.max_pool(self.conv3_4, 'pool3')
    self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
    self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
    self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
    self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
    self.pool4 = self.max_pool(self.conv4_4, 'pool4')
    self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
    self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
    self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
    self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
    self.pool5 = self.max_pool(self.conv5_4, 'pool5')
    self.data_dict = None
    print(("building vgg model finished: %ds" % (time.time() - start_time)))
    return [self.conv1_2, self.conv2_2, self.conv3_4, self.conv4_4, self.conv5_4]

  def max_pool(self, bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
  
  def conv_layer(self, bottom, name):
    with tf.variable_scope(name):
      filt = self.get_conv_filter(name)
      conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
      conv_biases = self.get_bias(name)
      bias = tf.nn.bias_add(conv, conv_biases)
      relu = tf.nn.relu(bias)
      return relu
    
  def get_conv_filter(self, name):
    return tf.constant(self.data_dict[name][0], name="filter")
  
  def get_bias(self, name):
    return tf.constant(self.data_dict[name][1], name="biases")

