from __future__ import division

import tensorflow as tf
import numpy as np
import time
from collections import defaultdict

import copy
from datetime import datetime
import os.path as osp
import os

from ..models.base_model import *
from ..utils import utils as utils
from ..tf_utils.op_utils import dev_wrap
from ..tf_utils import op_utils

from ..utils.colorize import *


def get_gaussian_maps(mu, shape_hw, inv_std, mode='ankush'):

  mu_x, mu_y = mu[:, :, 0:1], mu[:, :, 1:2]
  y = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[0]))
  x = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[1]))

  mu_y, mu_x = tf.expand_dims(mu_y, -1), tf.expand_dims(mu_x, -1)
  y = tf.reshape(y, [1, 1, shape_hw[0], 1])
  x = tf.reshape(x, [1, 1, 1, shape_hw[1]])
  g_y = tf.square(y - mu_y)
  g_x = tf.square(x - mu_x)
  dist = (g_y + g_x) * inv_std**2

  if mode == 'rot':
    g_yx = tf.exp(-dist)
  else:
    g_yx = tf.exp(-tf.pow(dist + 1e-5, 0.25))
  g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])

  return g_yx

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = os.path.join('/workspace/imm/vgg19.npy')
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):

        start_time = time.time()
        print("build model started")

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

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
        
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

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

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


class IMMModel(object):

  def __init__(self, sess, config, cell_info, vae_dim, global_step, training, dtype=tf.float32, name='IMMModel'):
    # super(IMMModel, self).__init__(dtype, name)
    self.config = config.model
    self.train_config = config.training
    self._global_step = global_step
    self.sess = sess
    self.cell_info = cell_info
    self.vae_dim = vae_dim
    self.train_mode = training
    self.colors = None
    
  def colorize_landmark_maps(self, maps):
    n_maps = maps.shape.as_list()[-1]
    # get n colors:
    if self.colors == None:
      self.colors = utils.get_n_colors(n_maps, pastel_factor=0.0)
    hmaps = [tf.expand_dims(maps[..., i], axis=3) * np.reshape(self.colors[i], [1, 1, 1, 3])
            for i in range(n_maps)]
    return tf.reduce_max(hmaps, axis=0)


  def perceptual_loss(self, gt_image, pred_image):

    vgg = Vgg19()

    with tf.variable_scope("content_vgg"):
      ims = tf.concat([gt_image, pred_image], axis=0)
      feats = vgg.build(ims)
      feat_gt, feat_pred = zip(*[tf.split(f, 2, axis=0) for f in feats])

      losses = []
      for k in range(len(feats)):
        l = tf.abs(feat_gt[k] - feat_pred[k])
        l = tf.reduce_mean(l)
        losses.append(l)

      loss = tf.reduce_mean(losses)
    return loss


  def simple_renderer(self, x, final_res=128):
    with tf.variable_scope('renderer'):

      filters = self.config.n_filters * 8
    
      size = x.shape.as_list()[1:3]
      conv_id = 1
      while size[0] <= final_res:
        x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_0'%conv_id)
        x = batch_norm(x, self.train_mode, scope='b_norm_%d_0'%conv_id)
        x = relu(x)
        x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_1'%conv_id)
        x = batch_norm(x, self.train_mode, scope='b_norm_%d_1'%conv_id)
        x = relu(x)
        if size[0]==final_res:
          crude_output = conv(x, 3, kernel=3, stride=1, scope='conv_%d_0'%(conv_id+1))
          mask = conv(x, 1, kernel=3, stride=1, scope='conv_%d_1'%(conv_id+1))
          mask = tf.nn.sigmoid(mask)
          break
        else:
          x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_0'%(conv_id+1))
          x = batch_norm(x, self.train_mode, scope='b_norm_%d_0'%(conv_id+1))
          x = relu(x)
          x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_1'%(conv_id+1))
          x = batch_norm(x, self.train_mode, scope='b_norm_%d_1'%(conv_id+1))
          x = relu(x)
          x = tf.image.resize_images(x, [2 * s for s in size])
        size = x.shape.as_list()[1:3]
        conv_id += 2
        if filters >= 8: filters /= 2
    return crude_output, mask


  def encoder(self, x):
    with tf.variable_scope('encoder'):
      filters = self.config.n_filters

      block_features = []

      x = conv(x, filters, kernel=7, stride=1, scope='conv_1')
      x = batch_norm(x, self.train_mode, scope='b_norm_1')
      x = relu(x)
      x = conv(x, filters, kernel=3, stride=1, scope='conv_2')
      x = batch_norm(x, self.train_mode, scope='b_norm_2')
      x = relu(x)
      block_features.append(x)

      for i in range(3):
        filters *= 2
        x = conv(x, filters, kernel=3, stride=2, scope='conv_%d'%(i*2+3))
        x = batch_norm(x, self.train_mode, scope='b_norm_%d'%(i*2+3))
        x = relu(x)
        x = conv(x, filters, kernel=3, stride=1, scope='conv_%d'%(i*2+4))
        x = batch_norm(x, self.train_mode, scope='b_norm_%d'%(i*2+4))
        x = relu(x)
        block_features.append(x)

      return block_features


  def image_encoder(self, x, filters=64):
    
    with tf.variable_scope('image_encoder'):
      block_features = self.encoder(x)
      # add input image to supply max resulution features
      block_features = [x] + block_features

      return block_features


  def pose_encoder(self, x, filters=32, map_sizes=None):

    with tf.variable_scope('pose_encoder', reuse=tf.AUTO_REUSE):

      block_features = self.encoder(x)
      x = block_features[-1]
      x = conv(x, self.config.n_maps, kernel=1, stride=1)

      def get_coord(other_axis, axis_size):
        # get "x-y" coordinates:
        g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
        g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
        coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size)) # W
        coord_pt = tf.reshape(coord_pt, [1, axis_size, 1])
        g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
        return g_c, g_c_prob

      xshape = x.shape.as_list()
      gauss_y, gauss_y_prob = get_coord(2, xshape[1])  # B,NMAP
      gauss_x, gauss_x_prob = get_coord(1, xshape[2])  # B,NMAP
      gauss_mu = tf.stack([gauss_x, gauss_y], axis=2)

      return gauss_mu
    
    
  def pose_encoder_large(self, x, filters=32, final_res=128, map_sizes=None):

    with tf.variable_scope('pose_encoder_large', reuse=tf.AUTO_REUSE):

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
        x = relu(x)
        x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_1'%conv_id)
        x = batch_norm(x, self.train_mode, scope='b_norm_%d_1'%conv_id)
        x = relu(x)
        if size[0]==final_res:
          x = conv(x, self.config.n_maps, kernel=1, stride=1)
          break
        else:
          x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_0'%(conv_id+1))
          x = batch_norm(x, self.train_mode, scope='b_norm_%d_0'%(conv_id+1))
          x = relu(x)
          x = conv(x, filters, kernel=3, stride=1, scope='conv_%d_1'%(conv_id+1))
          x = batch_norm(x, self.train_mode, scope='b_norm_%d_1'%(conv_id+1))
          x = relu(x)
          x = tf.image.resize_images(x, [2 * s for s in size])
        size = x.shape.as_list()[1:3]
        conv_id += 2
        if filters >= 8: filters /= 2
    
      def get_coord(other_axis, axis_size):
        # get "x-y" coordinates:
        g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
        g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
        coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size)) # W
        coord_pt = tf.reshape(coord_pt, [1, axis_size, 1])
        g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
        return g_c, g_c_prob

      xshape = x.shape.as_list()
      gauss_y, gauss_y_prob = get_coord(2, xshape[1])  # B,NMAP
      gauss_x, gauss_x_prob = get_coord(1, xshape[2])  # B,NMAP
      gauss_mu = tf.stack([gauss_x, gauss_y], axis=2)

      return gauss_mu    


  def decoder(self, input_, name='decoder'):
    out = linear(input_, self.config.n_maps*2, name='dec_fc2')
    return tanh(out)


  def lstm_model(self, layers):
    lstm_cells = [
        tf.nn.rnn_cell.BasicLSTMCell(units, state_is_tuple=True)
        for units in layers
    ]
    lstm_cells = [
        tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0)
        for cell in lstm_cells
    ]
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=True)
    return stacked_lstm


  def vae_encoder(self, x, f_pt, act_code):
        
    with tf.variable_scope('vae_encoder'):

      cell = self.lstm_model(self.cell_info)
      state = cell.zero_state(tf.shape(x)[0], tf.float32)
      outputs, _ = tf.nn.dynamic_rnn(cell, x, initial_state=state, dtype=tf.float32)
    
      logit = tf.contrib.layers.fully_connected(tf.concat([outputs[:,-1,:], f_pt, act_code], axis = -1), self.vae_dim*2)
      mu = logit[:, :self.vae_dim]
      stddev = logit[:, self.vae_dim:]

      return mu, stddev
    

  def vae_decoder(self, x, f_pt, act_code):
        
    with tf.variable_scope('vae_decoder', reuse=tf.AUTO_REUSE):

      cell = self.lstm_model(self.cell_info)
      state = cell.zero_state(tf.shape(x)[0], tf.float32)
      input_ = tf.contrib.layers.fully_connected(tf.concat([x, f_pt, act_code], axis = -1), 32)
      empty_input = tf.zeros_like(input_)
    
      outputs = []
      output, state = cell(input_, state)
      outputs.append(tf.expand_dims(self.decoder(output), axis = 1))

      for i in range(31):
        output, state = cell(empty_input, state)
        outputs.append(tf.expand_dims(self.decoder(output), axis = 1))

      outputs = tf.concat(outputs, axis = 1) 

      return outputs


  def loss(self, pred_seq, gt_seq, mu, stddev, pred_im_seq, real_im_seq):

    l = 1000*tf.abs(pred_seq - gt_seq)
    pred_seq_loss = tf.reduce_mean(l)
    self.seq_los = tf.summary.scalar('pt_recon_loss', pred_seq_loss)

    kl_l = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(stddev) - tf.log(1e-8 + tf.square(stddev)) - 1, 1)
    kl_loss = tf.reduce_mean(kl_l)
    self.kl_los = tf.summary.scalar('kl_loss', kl_loss)
    
    reconstruction_loss = self.perceptual_loss((pred_im_seq+1)/2.0*255.0, (real_im_seq+1)/2.0*255.0)
    self.g_recon = tf.summary.scalar('img_recon_loss', reconstruction_loss)

    self.summary_loss = tf.summary.merge([self.seq_los, self.kl_los, self.g_recon])

    loss = kl_loss + pred_seq_loss

    return loss


  def build(self, inputs, output_tensors=False, build_loss=True):

    gauss_mode = self.config.gauss_mode
    im, im_seq, act_code, first_pt_, pt_seq =\
                inputs['image'], inputs['real_im_seq'], inputs['action_code'], inputs['landmarks'], inputs['real_pt_seq']

    im_ = tf.expand_dims(im, 1)
    im_ = tf.tile(im_, [1, 32, 1, 1, 1])
    im_ = tf.reshape(im_, [-1, 128, 128, 3])

    im_size = im.shape.as_list()[1:3]
    assert im_size[0] == im_size[1]
    im_size = im_size[0]

    # determine the sizes for the renderer
    render_sizes = []
    size = im_size
    stride = self.config.renderer_stride
    while True:
      render_sizes.append(size)
      if size <= self.config.min_res:
        break
      size = size // stride

    embeddings = self.image_encoder(im)
    im_emb_size = embeddings[-2].shape.as_list()[-3:]
    embeddings = tf.expand_dims(embeddings[-2], 1)
    embeddings = tf.tile(embeddings, [1, 32, 1, 1, 1])
    embeddings = tf.reshape(embeddings, [-1]+im_emb_size)

    first_pt = self.pose_encoder_large(im, map_sizes=render_sizes)
    first_pt = tf.reshape(first_pt, [-1, self.config.n_maps*2])

    if self.train_mode == True:
      mu, stddev = self.vae_encoder(tf.reshape(pt_seq, [-1, 32, self.config.n_maps*2]), first_pt, act_code)
      z = mu + stddev*tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
      pred_seq = self.vae_decoder(z, first_pt, act_code)
    else:
      z = tf.random_normal([tf.shape(first_pt)[0]]+[self.vae_dim], 0, 1, dtype=tf.float32)
      pred_seq = self.vae_decoder(z, first_pt, act_code)

    pred_seq = tf.reshape(pred_seq, [-1, 32, self.config.n_maps, 2])

    rd_sz = self.config.render_input_sz
    print('joint_embedding_size : ', rd_sz)

    current_pt_map = get_gaussian_maps(tf.reshape(first_pt, [-1, self.config.n_maps, 2]), \
                        [rd_sz, rd_sz], 1.0 / self.config.gauss_std, mode=gauss_mode)
    emb_size = current_pt_map.shape.as_list()[-3:]
    current_pt_map = tf.expand_dims(current_pt_map, 1)
    current_pt_map = tf.tile(current_pt_map, [1, 32, 1, 1, 1])
    current_pt_map = tf.reshape(current_pt_map, [-1]+emb_size)

    pred_pt_map = get_gaussian_maps(tf.reshape(pred_seq, [-1, self.config.n_maps, 2]), \
                        [rd_sz, rd_sz], 1.0 / self.config.gauss_std, mode=gauss_mode)

    joint_embedding = tf.concat([embeddings, current_pt_map, pred_pt_map], axis = -1)
    crude_output, mask = self.simple_renderer(joint_embedding)
    final_output = im_*mask + crude_output*(1-mask)

    current_landmarks_map = get_gaussian_maps(tf.reshape(first_pt, [-1, self.config.n_maps, 2]),\
                                   [128, 128], 1.0 / self.config.gauss_std, mode=gauss_mode)
    future_landmarks_map = get_gaussian_maps(tf.reshape(pred_seq, [-1, self.config.n_maps, 2]),\
                            [128, 128], 1.0 / self.config.gauss_std, mode=gauss_mode)

    pred_seq_img = []
    for i in range(32):
      gauss_map = get_gaussian_maps(tf.reshape(pred_seq[:,i,::], [-1, self.config.n_maps, 2]), [64, 64], 1.0 / self.config.gauss_std, mode=gauss_mode)
      pred_seq_img.append(self.colorize_landmark_maps(gauss_map))
    pred_seq_img = tf.concat(pred_seq_img, axis = 2)

    real_seq_img = []
    for i in range(32):
      gauss_map = get_gaussian_maps(pt_seq[:,i,::], [64, 64], 1.0 / self.config.gauss_std, mode=gauss_mode)
      real_seq_img.append(self.colorize_landmark_maps(gauss_map))
    real_seq_img = tf.concat(real_seq_img, axis = 2)

    first_pt_map = get_gaussian_maps(first_pt_, [128, 128], 1.0 / self.config.gauss_std, mode=gauss_mode)
    
    pred_im_seq = []
    pred_im_seq_ = tf.reshape(final_output, [-1,32,128,128,3])
    for i in range(32):
        pred_im_seq.append(pred_im_seq_[:,i,::])
#         pred_im_seq.append(tf.image.resize_images(pred_im_seq_[:,i,::], [64,64]))
    pred_im_seq = tf.concat(pred_im_seq, axis = 2)
    
    real_im_seq = []
    for i in range(32):
        real_im_seq.append(im_seq[:,i,::])
#         real_im_seq.append(tf.image.resize_images(im_seq[:,i,::], [64,64]))
    real_im_seq = tf.concat(real_im_seq, axis = 2)    
    
    mask_seq = []
    mask_seq_ = tf.reshape(mask, [-1,32,128,128,1])
    for i in range(32):
        mask_seq.append(mask_seq_[:,i,::])
    mask_seq = tf.concat(mask_seq, axis = 2)
        
    crude_im_seq = []
    crude_im_seq_ = tf.reshape(crude_output, [-1,32,128,128,3])
    for i in range(32):
        crude_im_seq.append(crude_im_seq_[:,i,::])
#         pred_im_seq.append(tf.image.resize_images(pred_im_seq_[:,i,::], [64,64]))
    crude_im_seq = tf.concat(crude_im_seq, axis = 2)
    
    # visualize images:
    self.first_pt = tf.summary.image('first_pt', self.colorize_landmark_maps(first_pt_map), max_outputs=2)
    self.im_sum = tf.summary.image('im', (im+1)/2.0*255.0, max_outputs=2)
    self.pred_p_seq = tf.summary.image('predicted_pose_sequence', self.colorize_landmark_maps(pred_seq_img), max_outputs=2)
    self.real_p_seq = tf.summary.image('real_pose_sequence', self.colorize_landmark_maps(real_seq_img), max_outputs=2)
    self.pred_im_seq = tf.summary.image('predicted_image_sequence', \
                                        tf.clip_by_value((pred_im_seq+1)/2.0*255.0, 0, 255), max_outputs=2)
    self.real_im_seq = tf.summary.image('real_image_sequence', \
                                        tf.clip_by_value((real_im_seq+1)/2.0*255.0, 0, 255), max_outputs=2)
    self.mask_sum = tf.summary.image('mask', mask_seq*255.0, max_outputs=2)
    self.crude_im_seq = tf.summary.image('crude_image_sequence', \
                                        tf.clip_by_value((crude_im_seq+1)/2.0*255.0, 0, 255), max_outputs=2)
    self.image_summary = tf.summary.merge([self.im_sum, self.first_pt, self.pred_p_seq, self.real_p_seq, \
                                           self.pred_im_seq, self.real_im_seq, self.mask_sum, self.crude_im_seq])
    
    self.loss_ = None
    if self.train_mode == True:
      # compute the losses:
      self.loss_ = self.loss(pred_seq, pt_seq, mu, stddev, final_output, tf.reshape(im_seq, [-1,128,128,3]))
      t_vars = tf.trainable_variables()
    
#       vars_to_train = [var for var in t_vars if 'vae' in var.name]
      print(t_vars)

      lr = tf.train.exponential_decay(self.train_config.lr.start_val, self._global_step,\
                                      self.train_config.lr.step, self.train_config.lr.decay)
      self.lr_ = tf.summary.scalar('lr', lr)
      self.train_op_ = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(self.loss_, var_list=t_vars, global_step=self._global_step)

    self.tensor = {'real_im_seq': im_seq, 'im': im, 'pred_im_seq': tf.reshape(final_output, [-1,32,128,128,3]),\
                   'mask': tf.reshape(mask, [-1,32,128,128,1]), 'pred_im_crude': tf.reshape(crude_output, [-1,32,128,128,3]),\
                   'current_points': tf.reshape(self.colorize_landmark_maps(current_landmarks_map), [-1,128,128,3]),\
                   'future_points': tf.reshape(self.colorize_landmark_maps(future_landmarks_map), [-1,32,128,128,3]),
                   'fut_pt_raw': pred_seq}

                   
    # self.tensor = {'current_points': tf.reshape(self.colorize_landmark_maps(current_landmarks_map), [-1,128,128,3]),\
    #                'future_points': tf.reshape(self.colorize_landmark_maps(future_landmarks_map), [-1,32,128,128,3])}
                   

  def train_loop(self, opts, train_dataset, test_dataset, handle_pl, num_steps, checkpoint_fnames, reset_global_step = -1, vars_to_restore='all',\
                ignore_missing_vars=True, exclude_vars=False):
   
    tf.logging.set_verbosity(tf.logging.INFO)

    # define iterators
    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    self.sess.run([global_init,local_init])

    # set up iterators
    train_handle = self.sess.run(train_iterator.string_handle())
    self.sess.run(train_iterator.initializer)
    test_handle = self.sess.run(test_iterator.string_handle())

    # check if we need to restore the model:
    for checkpoint_fname in checkpoint_fnames:
      # restore checkpoint:
      if tf.gfile.Exists(checkpoint_fname) or tf.gfile.Exists(checkpoint_fname + '.index'):
        print('RESTORING MODEL from: ' + checkpoint_fname)
        reader = tf.train.NewCheckpointReader(checkpoint_fname)
        vars_to_restore = tf.global_variables()
        checkpoint_vars = reader.get_variable_to_shape_map().keys()
        vars_ignored = [v.name for v in vars_to_restore if v.name[:-2] not in checkpoint_vars]
        # print(colorize('vars-IGNORED (not restoring):', 'blue', bold=True))
        # print(colorize(', '.join(vars_ignored), 'blue'))
        vars_to_restore = [v for v in vars_to_restore if v.name[:-2] in checkpoint_vars]
        vars_to_restore = [v for v in vars_to_restore if 'global' not in v.name[:-2] \
                                                       and 'Adam' not in v.name[:-2] \
                                                       and 'beta' not in v.name[:-2] \
                                                       and 'b_norm' not in v.name[:-2]]
        vars_to_restore_str = [v.name for v in vars_to_restore]
        print(colorize('vars-to-RESTORE:', 'blue', bold=True))
        print(colorize(', '.join(vars_to_restore_str), 'blue'))
        restorer = tf.train.Saver(var_list=vars_to_restore)
        restorer.restore(self.sess, checkpoint_fname)
      else:
        raise Exception('model file does not exist at: ' + checkpoint_fname)

    # create a summary writer:
    train_writer = tf.summary.FileWriter(osp.join(opts['log_dir'], 'train'), self.sess.graph)
    test_writer = tf.summary.FileWriter(osp.join(opts['log_dir'], 'test'))

    # create a check-pointer:
    #  --> keep ALL the checkpoint files:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # get the value of the global-step:
    start_step = self.sess.run(self._global_step)
    # run the training loop:
    begin_time = time.time()

    for step in range(int(start_step), num_steps):
        start_time = time.time()
        if step % opts['n_summary'] == 0:
          feed_dict = {handle_pl: train_handle}
          loss_value, _, ls_sum, im_sum, lr_sum = self.sess.run([self.loss_, self.train_op_, self.summary_loss, self.image_summary, self.lr_], feed_dict=feed_dict)
          train_writer.add_summary(ls_sum, step)
          train_writer.add_summary(im_sum, step)
          train_writer.add_summary(lr_sum, step)
          train_writer.flush()
        else:
          feed_dict = {handle_pl: train_handle}
          loss_value, _ = self.sess.run([self.loss_, self.train_op_], feed_dict=feed_dict)
        
        duration = time.time() - start_time

        # make sure that we have non NaNs:
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 250 == 0:
          # print stats for this batch:
          examples_per_sec = opts['batch_size'] / float(duration)
          format_str = '%s: step %d, loss = %.4f (%.1f examples/sec) %.3f sec/batch %d'
          tf.logging.info(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration,\
                                        self.sess.run(self._global_step)))

        # periodically test on test set
        if test_dataset and step % opts['n_test'] == 0:
          feed_dict = {handle_pl: test_handle}
          self.sess.run(test_iterator.initializer)
          test_iter = 0
          while True:
            try:
              start_time = time.time()
              if test_iter == 0:
                loss_value, ls_sum, im_sum = self.sess.run([self.loss_, self.summary_loss, self.image_summary], feed_dict=feed_dict)
                test_writer.add_summary(ls_sum, step)
                test_writer.add_summary(im_sum, step)
                test_writer.flush()
              else:
                loss_value = self.sess.run(self.loss_, feed_dict=feed_dict)
              duration = time.time() - start_time

              examples_per_sec = opts['batch_size'] / float(duration)
              format_str = 'test: %s: step %d, loss = %.4f (%.1f examples/sec) %.3f sec/batch'
              tf.logging.info(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration))
            except tf.errors.OutOfRangeError:
              print('iteration through test set finished')
              break
            test_iter += 1

        # periodically checkpoint:
        if step % opts['n_checkpoint'] == 0:
          checkpoint_path = osp.join(opts['log_dir'],'model.ckpt')
          saver.save(self.sess, checkpoint_path, global_step=step)

    total_time = time.time()-begin_time
    samples_per_sec = opts['batch_size'] * num_steps / float(total_time)
    print('Avg. samples per second %.3f'%samples_per_sec)



  def evaluate(self, opts, train_dataset, test_dataset, handle_pl, checkpoint_fnames, vars_to_restore=None, reset_global_step=True):
   
    tf.logging.set_verbosity(tf.logging.INFO)

    # define iterators
    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    self.sess.run([global_init,local_init])

    # set up iterators
    train_handle = self.sess.run(train_iterator.string_handle())
    self.sess.run(train_iterator.initializer)
    test_handle = self.sess.run(test_iterator.string_handle())

    for checkpoint_fname in checkpoint_fnames:
      # restore checkpoint:
      if tf.gfile.Exists(checkpoint_fname) or tf.gfile.Exists(checkpoint_fname + '.index'):
        print('RESTORING MODEL from: ' + checkpoint_fname)
        reader = tf.train.NewCheckpointReader(checkpoint_fname)
        vars_to_restore = tf.global_variables()
        checkpoint_vars = reader.get_variable_to_shape_map().keys()
        vars_ignored = [v.name for v in vars_to_restore if v.name[:-2] not in checkpoint_vars]
        # print(colorize('vars-IGNORED (not restoring):', 'blue', bold=True))
        # print(colorize(', '.join(vars_ignored), 'blue'))
        vars_to_restore = [v for v in vars_to_restore if v.name[:-2] in checkpoint_vars]
        vars_to_restore_str = [v.name for v in vars_to_restore]
        print(colorize('vars-to-RESTORE:', 'blue', bold=True))
        print(colorize(', '.join(vars_to_restore_str), 'blue'))
        restorer = tf.train.Saver(var_list=vars_to_restore)
        restorer.restore(self.sess, checkpoint_fname)
      else:
        raise Exception('model file does not exist at: ' + checkpoint_fname)

    ############### compute output ###############

    tensors_names = self.tensor.keys()
    outputs = []
    
    ########## compute output from trainset ##########
    tensors_results = {k: [] for k in tensors_names}

    feed_dict = {handle_pl: train_handle}
    self.sess.run(train_iterator.initializer)

    while True:
      try:
        tensors_values = self.sess.run(self.tensor, feed_dict=feed_dict)
        for name in tensors_names:
          tensors_results[name].append(tensors_values[name])
      except tf.errors.OutOfRangeError:
        print('iteration through train set finished')
        break

    for name in tensors_names:
      tensors_results[name] = np.concatenate(tensors_results[name], axis = 0)

    outputs.append(tensors_results)

    ########## compute output from testset ##########
    tensors_results = {k: [] for k in tensors_names}

    feed_dict = {handle_pl: test_handle}
    self.sess.run(test_iterator.initializer)

    while True:
      try:
        tensors_values = self.sess.run(self.tensor, feed_dict=feed_dict)
        for name in tensors_names:
          tensors_results[name].append(tensors_values[name])
      except tf.errors.OutOfRangeError:
        print('iteration through test set finished')
        break

    for name in tensors_names:
      tensors_results[name] = np.concatenate(tensors_results[name], axis = 0)

    outputs.append(tensors_results)

    return outputs



  def evaluate_kpts(self, opts, train_dataset, test_dataset, handle_pl, checkpoint_fnames, vars_to_restore=None, reset_global_step=True):
   
    tf.logging.set_verbosity(tf.logging.INFO)

    # define iterators
    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    self.sess.run([global_init,local_init])

    # set up iterators
    train_handle = self.sess.run(train_iterator.string_handle())
    self.sess.run(train_iterator.initializer)
    test_handle = self.sess.run(test_iterator.string_handle())

    for checkpoint_fname in checkpoint_fnames:
      # restore checkpoint:
      if tf.gfile.Exists(checkpoint_fname) or tf.gfile.Exists(checkpoint_fname + '.index'):
        print('RESTORING MODEL from: ' + checkpoint_fname)
        reader = tf.train.NewCheckpointReader(checkpoint_fname)
        vars_to_restore = tf.global_variables()
        checkpoint_vars = reader.get_variable_to_shape_map().keys()
        vars_ignored = [v.name for v in vars_to_restore if v.name[:-2] not in checkpoint_vars]
        print(colorize('vars-IGNORED (not restoring):', 'blue', bold=True))
        print(colorize(', '.join(vars_ignored), 'blue'))
        vars_to_restore = [v for v in vars_to_restore if v.name[:-2] in checkpoint_vars]
        restorer = tf.train.Saver(var_list=vars_to_restore)
        restorer.restore(self.sess, checkpoint_fname)
      else:
        raise Exception('model file does not exist at: ' + checkpoint_fname)

    for i in range(100):
        
        print(i)
        
        ############### compute output ###############

        tensors_names = self.tensor.keys()

        ########## compute output from trainset ##########
        tensors_results = {k: [] for k in tensors_names}

        feed_dict = {handle_pl: train_handle}
        self.sess.run(train_iterator.initializer)

        step = 0
        while True:
          try:
            tensors_values = self.sess.run(self.tensor, feed_dict=feed_dict)
            for name in tensors_names:
              tensors_results[name].append(tensors_values[name])
          except tf.errors.OutOfRangeError:
            print('iteration through train set finished')
            break

        for name in tensors_names:
          tensors_results[name] = np.concatenate(tensors_results[name], axis = 0)

        np.save('./data/kpts_std/'+str(i)+'_img.npy', tensors_results['pred_im_seq'])

    return
