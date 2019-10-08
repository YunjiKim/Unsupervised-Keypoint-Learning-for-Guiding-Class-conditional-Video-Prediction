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

def colorize_landmark_maps(self, maps):
  n_maps = maps.shape.as_list()[-1]
  # get n colors:
  if self.colors == None:
    self.colors = utils.get_n_colors(n_maps, pastel_factor=0.0)
  hmaps = [tf.expand_dims(maps[..., i], axis=3) * np.reshape(self.colors[i], [1, 1, 1, 3])
          for i in range(n_maps)]
  return tf.reduce_max(hmaps, axis=0)

def translator(self, x, final_res=128):
  with tf.variable_scope('translator'):
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

def get_coord(other_axis, axis_size):
  # get "x-y" coordinates:
  g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
  g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
  coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size)) # W
  coord_pt = tf.reshape(coord_pt, [1, axis_size, 1])
  g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
  return g_c, g_c_prob

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
    xshape = x.shape.as_list()
    gauss_y, gauss_y_prob = get_coord(2, xshape[1])  # B,NMAP
    gauss_x, gauss_x_prob = get_coord(1, xshape[2])  # B,NMAP
    gauss_mu = tf.stack([gauss_x, gauss_y], axis=2)
    return gauss_mu

def decoder(self, input_, name='decoder'):
  out = linear(input_, self.config.n_maps*2, name='dec_fc2')
  return tanh(out)

def lstm_model(self, layers):
  lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(units, state_is_tuple=True) for units in layers]
  lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0) for cell in lstm_cells]
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
    outputs.append(tf.expand_dims(decoder(output), axis = 1))
    for i in range(31):
      output, state = cell(empty_input, state)
      outputs.append(tf.expand_dims(decoder(output), axis = 1))
    outputs = tf.concat(outputs, axis = 1) 
    return outputs

def Seq_discriminator(self, x):
  with tf.variable_scope('seq_discr', reuse=tf.AUTO_REUSE):
    cell = self.lstm_model([1024,1024])
    state = cell.zero_state(tf.shape(x)[0], tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(cell, x, initial_state=state, dtype=tf.float32)
    logit = tf.contrib.layers.fully_connected(outputs, 1)
    return logit[:,-1,:]  
 
def Img_discriminator(self, x):
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

  
class KD_IT(object):
  def __init__(self, sess, config, global_step, training, dtype=tf.float32, name='KD_IT'):
    self.config = config.model
    self.train_config = config.training
    self._global_step = global_step
    self.sess = sess
    self.train_mode = training
    self.colors = None
  
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
  
  def loss_D(self, future_im_pred, future_im, future_landmarks):
    real_ = self.Img_discriminator(future_im)
    fake_ = self.Img_discriminator(future_im_pred)
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_), logits=real_))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_), logits=fake_))
    self.d_real_ = tf.summary.scalar('D_real', real_loss)
    self.d_fake_ = tf.summary.scalar('D_fake', fake_loss)
    self.d_summary_loss = tf.summary.merge([self.d_real_, self.d_fake_])
    return real_loss + fake_loss

  def loss_G(self, future_im_pred, future_crude, future_im, future_landmarks):
    reconstruction_loss = self.perceptual_loss((future_im+1)/2.0*255.0, (future_im_pred+1)/2.0*255.0)
    self.g_recon = tf.summary.scalar('reconstruction_metric', reconstruction_loss)
    fake_ = self.Img_discriminator(future_im_pred)
    adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_), logits=fake_))
    self.g_adv = tf.summary.scalar('G_adv_loss', adv_loss)
    self.g_summary_loss = tf.summary.merge([self.g_recon, self.g_adv])
    return reconstruction_loss + adv_loss

  def build(self, inputs, output_tensors=False, build_loss=True):
    gauss_mode = self.config.gauss_mode
    filters = self.config.n_filters
    
    # generate translated images
    im, future_im = inputs['image'], inputs['future_image']
    future_im_size = future_im.shape.as_list()[1:3]
    assert future_im_size[0] == future_im_size[1]
    future_im_size = future_im_size[0]
    embeddings = self.image_encoder(im)
    current_gauss_pt = self.pose_encoder(im)
    future_gauss_pt = self.pose_encoder(future_im)
    rd_sz = self.config.render_input_sz
    current_pt_map = get_gaussian_maps(current_gauss_pt, [rd_sz, rd_sz], 1.0 / self.config.gauss_std)
    future_pt_map = get_gaussian_maps(future_gauss_pt, [rd_sz, rd_sz], 1.0 / self.config.gauss_std)
    joint_embedding = tf.concat([embeddings[-2], current_pt_map, future_pt_map], axis = -1)
    crude_output, mask = self.translator(joint_embedding)
    final_output = im*mask + crude_output*(1-mask)
    
    # compute loss
    self.loss_G_ = self.loss_D_ = None
    if build_loss:
      # compute the losses:
      self.loss_D_ = self.loss_D(final_output, future_im, future_landmarks_map)
      self.loss_G_ = self.loss_G(final_output, crude_output, future_im, future_landmarks_map)
      t_vars = tf.trainable_variables()
      G_vars = [var for var in t_vars if 'img_discr' not in var.name]
      D_vars = [var for var in t_vars if 'img_discr' in var.name]
      lr = tf.train.exponential_decay(self.train_config.lr.start_val, self._global_step, self.train_config.lr.step, self.train_config.lr.decay)
      self.lr_ = tf.summary.scalar('lr', lr)
      self.train_op_D = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(self.loss_D_, var_list=D_vars)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
          self.train_op_G = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(self.loss_G_, var_list=G_vars, global_step=self._global_step)
 
    # visualize images:
    current_landmarks_map = get_gaussian_maps(current_gauss_pt, [128, 128], 1.0 / self.config.gauss_std)
    future_landmarks_map = get_gaussian_maps(future_gauss_pt, [128, 128], 1.0 / self.config.gauss_std)
    self.future_im_sum = tf.summary.image('future_im', (future_im+1)/2.0*255.0, max_outputs=2)
    self.im_sum = tf.summary.image('im', (im+1)/2.0*255.0, max_outputs=2)
    self.cur_pt_sum = tf.summary.image('current_points', self.colorize_landmark_maps(current_landmarks_map), max_outputs=2)
    self.fut_pt_sum = tf.summary.image('future_points', self.colorize_landmark_maps(future_landmarks_map), max_outputs=2)
    self.crude_im_sum = tf.summary.image('future_im_crude', tf.clip_by_value((crude_output+1)/2.0*255.0, 0, 255), max_outputs=2)
    self.pred_im_sum = tf.summary.image('future_im_pred', tf.clip_by_value((final_output+1)/2.0*255.0, 0, 255), max_outputs=2)
    self.mask_sum = tf.summary.image('mask', mask*255.0, max_outputs=2)
    self.image_summary = tf.summary.merge([self.future_im_sum, self.im_sum, self.cur_pt_sum, self.fut_pt_sum, self.crude_im_sum, self.pred_im_sum, self.mask_sum])
    self.tensor = {'future_im': future_im, 'im': im, 'future_im_pred': final_output, 'mask': mask, 'future_im_crude': crude_output, 'gauss_pt': future_gauss_pt,
                   'current_points': self.colorize_landmark_maps(current_landmarks_map), 'future_points': self.colorize_landmark_maps(future_landmarks_map)}

  def train_loop(self, opts, train_dataset, test_dataset, training_pl, handle_pl, num_steps, checkpoint_fname, reset_global_step = -1, vars_to_restore='all',\
                ignore_missing_vars=False, exclude_vars=False):
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
    if tf.gfile.Exists(checkpoint_fname) or tf.gfile.Exists(checkpoint_fname+'.index'):
      print(colorize('RESTORING MODEL from: '+checkpoint_fname, 'blue', bold=True))
      if not isinstance(vars_to_restore,list):
        if vars_to_restore == 'all':
          vars_to_restore = tf.global_variables()
        elif vars_to_restore == 'model':
          vars_to_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
      if reset_global_step >= 0:
        print(colorize('Setting global-step to %d.'%reset_global_step,'red',bold=True))
        var_names = [v.name for v in vars_to_restore]
        reset_vid = [i for i in range(len(var_names)) if 'global_step' in var_names[i]]
        if reset_vid:
          vars_to_restore.pop(reset_vid[0])
      print(colorize('vars-to-be-restored:','green',bold=True))
      print(colorize(', '.join([v.name for v in vars_to_restore]),'green'))
      if ignore_missing_vars:
        reader = tf.train.NewCheckpointReader(checkpoint_fname)
        checkpoint_vars = reader.get_variable_to_shape_map().keys()
        vars_ignored = [v.name for v in vars_to_restore if v.name[:-2] not in checkpoint_vars]
        print(colorize('vars-IGNORED (not restoring):','blue',bold=True))
        print(colorize(', '.join(vars_ignored),'blue'))
        vars_to_restore = [v for v in vars_to_restore if v.name[:-2] in checkpoint_vars]
      if exclude_vars:
        for exclude_var_name in exclude_vars:
          var_names = [v.name for v in vars_to_restore]
          reset_vid = [i for i in range(len(var_names)) if exclude_var_name in var_names[i]]
          if reset_vid:
            vars_to_restore.pop(reset_vid[0])
      restorer = tf.train.Saver(var_list=vars_to_restore)
      restorer.restore(self.sess,checkpoint_fname)

    # create a summary writer:
    train_writer = tf.summary.FileWriter(osp.join(opts['log_dir'], 'train'), self.sess.graph)
    test_writer = tf.summary.FileWriter(osp.join(opts['log_dir'], 'test'))
    # create a check-pointer:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    # run the training loop:
    start_step = self.sess.run(self._global_step)
    begin_time = time.time()
    for step in range(int(start_step), num_steps):
      start_time = time.time()
      if step % opts['n_summary'] == 0:
        feed_dict = {handle_pl: train_handle, training_pl: True}
        loss_value_D, _, d_ls_sum = self.sess.run([self.loss_D_, self.train_op_D, self.d_summary_loss], feed_dict=feed_dict)
        loss_value_G, _, g_ls_sum, im_sum, lr_sum = self.sess.run([self.loss_G_, self.train_op_G, self.g_summary_loss, self.image_summary, self.lr_], feed_dict=feed_dict)
        train_writer.add_summary(d_ls_sum, step)
        train_writer.add_summary(g_ls_sum, step)
        train_writer.add_summary(im_sum, step)
        train_writer.add_summary(lr_sum, step)
        train_writer.flush()
      else:
        feed_dict = {handle_pl: train_handle, training_pl: True}
        loss_value_D, _ = self.sess.run([self.loss_D_, self.train_op_D], feed_dict=feed_dict)
        loss_value_G, _ = self.sess.run([self.loss_G_, self.train_op_G], feed_dict=feed_dict)
      duration = time.time() - start_time
      # make sure that we have non NaNs:
      assert not np.isnan(loss_value_D), 'D_Model diverged with loss = NaN'
      assert not np.isnan(loss_value_G), 'G_Model diverged with loss = NaN'
      if step % 250 == 0:
        # print stats for this batch:
        examples_per_sec = opts['batch_size'] / float(duration)
        format_str = '%s: step %d, loss_D = %.4f, loss_G = %.4f (%.1f examples/sec) %.3f sec/batch'
        tf.logging.info(format_str % (datetime.now(), step, loss_value_D, loss_value_G,
                        examples_per_sec, duration))
      # periodically test on test set
      if test_dataset and step % opts['n_test'] == 0:
        feed_dict = {handle_pl: test_handle, training_pl: False}
        self.sess.run(test_iterator.initializer)
        test_iter = 0
        while True:
          try:
            start_time = time.time()
            if test_iter == 0:
              loss_value_D, loss_value_G, d_ls_sum, g_ls_sum, im_sum = \
                self.sess.run([self.loss_D_, self.loss_G_, self.d_summary_loss, self.g_summary_loss, self.image_summary], feed_dict=feed_dict)
              test_writer.add_summary(d_ls_sum, step)
              test_writer.add_summary(g_ls_sum, step)
              test_writer.add_summary(im_sum, step)
              test_writer.flush()
            else:
              loss_value_D, loss_value_G = self.sess.run([self.loss_D_, self.loss_G_], feed_dict=feed_dict)
            duration = time.time() - start_time
            examples_per_sec = opts['batch_size'] / float(duration)
            format_str = 'test: %s: step %d, loss_D = %.4f, loss_G = %.4f (%.1f examples/sec) %.3f sec/batch'
            tf.logging.info(format_str % (datetime.now(), step, loss_value_D, loss_value_G,
                            examples_per_sec, duration))
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

  
class MOGEN(object):
  def __init__(self, sess, config, cell_info, vae_dim, global_step, training, dtype=tf.float32, name='MOGEN'):
    self.config = config.model
    self.train_config = config.training
    self._global_step = global_step
    self.sess = sess
    self.cell_info = cell_info
    self.vae_dim = vae_dim
    self.train_mode = training
    self.colors = None
    
  def loss_D(self, pred_seq, gt_seq):
    real_ = self.Seq_discriminator(gt_seq)
    fake_ = self.Seq_discriminator(pred_seq)
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_), logits=real_))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_), logits=fake_))
    loss = real_loss + fake_loss
    self.d_real_ = tf.summary.scalar('D_real', real_loss)
    self.d_fake_ = tf.summary.scalar('D_fake', fake_loss)
    self.d_summary_loss = tf.summary.merge([self.d_real_, self.d_fake_])
    return loss
    
  def loss_G(self, pred_seq, gt_seq, mu, stddev):
    l = 1000*tf.abs(pred_seq - gt_seq)
    pred_seq_loss = tf.reduce_mean(l)
    self.seq_los = tf.summary.scalar('recon_loss', pred_seq_loss)
    kl_l = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(stddev) - tf.log(1e-8 + tf.square(stddev)) - 1, 1)
    kl_loss = tf.reduce_mean(kl_l)
    self.kl_los = tf.summary.scalar('kl_loss', kl_loss)
    fake_ = self.Seq_discriminator(pred_seq)
    adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_), logits=fake_))
    self.g_adv = tf.summary.scalar('G_adv_loss', adv_loss)
    self.g_summary_loss = tf.summary.merge([self.seq_los, self.kl_los, self.g_adv])
    return kl_loss + pred_seq_loss + adv_loss

  def build(self, inputs, output_tensors=False):
    # generate samples
    im, landmarks, real_seq, act_code = inputs['image'], inputs['landmarks'], inputs['real_seq'], inputs['action_code']
    first_pt = tf.reshape(landmarks, [-1, self.config.n_maps*2])
    if self.train_mode == True:
      mu, stddev = self.vae_encoder(tf.reshape(real_seq, [-1, 32, self.config.n_maps*2]), first_pt, act_code)
      z = mu + stddev*tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
      pred_seq = self.vae_decoder(z, first_pt, act_code)
    else:
      z = tf.random_normal([tf.shape(first_pt)[0]]+[self.vae_dim], 0, 1, dtype=tf.float32)
      pred_seq = self.vae_decoder(z, first_pt, act_code)
        
    pred_seq_img = []
    for i in range(32):
      gauss_map = get_gaussian_maps(tf.reshape(pred_seq[:,i,::], [-1, self.config.n_maps, 2]), [64, 64], 1.0 / self.config.gauss_std)
      pred_seq_img.append(self.colorize_landmark_maps(gauss_map))
    pred_seq_img = tf.concat(pred_seq_img, axis = 2)
    real_seq_img = []
    for i in range(32):
      gauss_map = get_gaussian_maps(real_seq[:,i,::], [64, 64], 1.0 / self.config.gauss_std)
      real_seq_img.append(self.colorize_landmark_maps(gauss_map))
    real_seq_img = tf.concat(real_seq_img, axis = 2)
    first_pt_map = get_gaussian_maps(tf.reshape(landmarks, [-1, self.config.n_maps, 2]), [128, 128], 1.0 / self.config.gauss_std)

    # compute loss
    self.loss_G_ = self.loss_D_ = None
    if self.train_mode == True:
      # compute the losses:
      self.loss_D_ = self.loss_D(pred_seq, tf.reshape(real_seq, [-1, 32, self.config.n_maps*2]))
      self.loss_G_ = self.loss_G(pred_seq, tf.reshape(real_seq, [-1, 32, self.config.n_maps*2]), mu, stddev)
      t_vars = tf.trainable_variables()
      G_vars = [var for var in t_vars if 'discr' not in var.name]
      D_vars = [var for var in t_vars if 'discr' in var.name]
      lr = tf.train.exponential_decay(self.train_config.lr.start_val, self._global_step, self.train_config.lr.step, self.train_config.lr.decay)
      self.lr_ = tf.summary.scalar('lr', lr)
      self.train_op_D = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(self.loss_D_, var_list=D_vars)
      self.train_op_G = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(self.loss_G_, var_list=G_vars, global_step=self._global_step)
    
    # visualize images:
    self.first_pt = tf.summary.image('first_pt', self.colorize_landmark_maps(first_pt_map), max_outputs=2)
    self.im_sum = tf.summary.image('im', (im+1)/2.0*255.0, max_outputs=2)
    self.pred_p_seq = tf.summary.image('predicted_pose_sequence', self.colorize_landmark_maps(pred_seq_img), max_outputs=2)
    self.real_p_seq = tf.summary.image('real_pose_sequence', self.colorize_landmark_maps(real_seq_img), max_outputs=2)
    self.image_summary = tf.summary.merge([self.im_sum, self.first_pt, self.pred_p_seq, self.real_p_seq])
    pred_seq_map = []
    for i in range(32):
      gauss_map = get_gaussian_maps(tf.reshape(pred_seq[:,i,::], [-1, self.config.n_maps, 2]), [128, 128], 1.0 / self.config.gauss_std, mode=gauss_mode)
      pred_seq_map.append(tf.expand_dims(self.colorize_landmark_maps(gauss_map),0))
    pred_seq_map = tf.concat(pred_seq_map, axis=0)
    real_seq_map = []
    for i in range(32):
      gauss_map = get_gaussian_maps(real_seq[:,i,::], [128, 128], 1.0 / self.config.gauss_std, mode=gauss_mode)
      real_seq_map.append(tf.expand_dims(self.colorize_landmark_maps(gauss_map),0))
    real_seq_map = tf.concat(real_seq_map, axis=0)
    self.tensor = {'im': im, 'pred_p_seq': pred_seq, 'real_p_seq': real_seq, \
                   'first_pt': self.colorize_landmark_maps(get_gaussian_maps(landmarks, [128, 128], 1.0 / self.config.gauss_std, mode=gauss_mode)),\
                   'real_seq_map': tf.transpose(real_seq_map, perm=[1,0,2,3,4]), 'pred_seq_map': tf.transpose(pred_seq_map, perm=[1,0,2,3,4])}

  def train_loop(self, opts, train_dataset, test_dataset, training_pl, handle_pl, num_steps, checkpoint_fname, reset_global_step = -1, vars_to_restore='all',\
                ignore_missing_vars=False, exclude_vars=False):
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
    if tf.gfile.Exists(checkpoint_fname) or tf.gfile.Exists(checkpoint_fname+'.index'):
      print(colorize('RESTORING MODEL from: '+checkpoint_fname, 'blue', bold=True))
      if not isinstance(vars_to_restore,list):
        if vars_to_restore == 'all':
          vars_to_restore = tf.global_variables()
        elif vars_to_restore == 'model':
          vars_to_restore = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
      if reset_global_step >= 0:
        print(colorize('Setting global-step to %d.'%reset_global_step,'red',bold=True))
        var_names = [v.name for v in vars_to_restore]
        reset_vid = [i for i in range(len(var_names)) if 'global_step' in var_names[i]]
        if reset_vid:
          vars_to_restore.pop(reset_vid[0])
      print(colorize('vars-to-be-restored:','green',bold=True))
      print(colorize(', '.join([v.name for v in vars_to_restore]),'green'))
      if ignore_missing_vars:
        reader = tf.train.NewCheckpointReader(checkpoint_fname)
        checkpoint_vars = reader.get_variable_to_shape_map().keys()
        vars_ignored = [v.name for v in vars_to_restore if v.name[:-2] not in checkpoint_vars]
        print(colorize('vars-IGNORED (not restoring):','blue',bold=True))
        print(colorize(', '.join(vars_ignored),'blue'))
        vars_to_restore = [v for v in vars_to_restore if v.name[:-2] in checkpoint_vars]
      if exclude_vars:
        for exclude_var_name in exclude_vars:
          var_names = [v.name for v in vars_to_restore]
          reset_vid = [i for i in range(len(var_names)) if exclude_var_name in var_names[i]]
          if reset_vid:
            vars_to_restore.pop(reset_vid[0])
      restorer = tf.train.Saver(var_list=vars_to_restore)
      restorer.restore(self.sess,checkpoint_fname)
    # create a summary writer:
    train_writer = tf.summary.FileWriter(osp.join(opts['log_dir'], 'train'), self.sess.graph)
    test_writer = tf.summary.FileWriter(osp.join(opts['log_dir'], 'test'))
    # create a check-pointer:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    # get the value of the global-step:
    start_step = self.sess.run(self._global_step)
    
    # run the training loop:
    begin_time = time.time()
    for step in range(int(start_step), num_steps):
      start_time = time.time()
      if step % opts['n_summary'] == 0:
        feed_dict = {handle_pl: train_handle, training_pl: True}
        loss_value_D, _, d_ls_sum = self.sess.run([self.loss_D_, self.train_op_D, self.d_summary_loss], feed_dict=feed_dict)
        loss_value_G, _, g_ls_sum, im_sum, lr_sum = self.sess.run([self.loss_G_, self.train_op_G, self.g_summary_loss, self.image_summary, self.lr_], feed_dict=feed_dict)
        train_writer.add_summary(d_ls_sum, step)
        train_writer.add_summary(g_ls_sum, step)
        train_writer.add_summary(im_sum, step)
        train_writer.add_summary(lr_sum, step)
        train_writer.flush()
      else:
        feed_dict = {handle_pl: train_handle, training_pl: True}
        loss_value_D, _ = self.sess.run([self.loss_D_, self.train_op_D], feed_dict=feed_dict)
        loss_value_G, _ = self.sess.run([self.loss_G_, self.train_op_G], feed_dict=feed_dict)
      duration = time.time() - start_time

      # make sure that we have non NaNs:
      assert not np.isnan(loss_value_D), 'D_Model diverged with loss = NaN'
      assert not np.isnan(loss_value_G), 'G_Model diverged with loss = NaN'
      if step % 250 == 0:
        # print stats for this batch:
        examples_per_sec = opts['batch_size'] / float(duration)
        format_str = '%s: step %d, loss_D = %.4f, loss_G = %.4f (%.1f examples/sec) %.3f sec/batch'
        tf.logging.info(format_str % (datetime.now(), step, loss_value_D, loss_value_G,
                        examples_per_sec, duration))
      # periodically test on test set
      if test_dataset and step % opts['n_test'] == 0:
        feed_dict = {handle_pl: test_handle, training_pl: False}
        self.sess.run(test_iterator.initializer)
        test_iter = 0
        while True:
          try:
            start_time = time.time()
            if test_iter == 0:
              loss_value_D, loss_value_G, d_ls_sum, g_ls_sum, im_sum = \
                self.sess.run([self.loss_D_, self.loss_G_, self.d_summary_loss, self.g_summary_loss, self.image_summary], feed_dict=feed_dict)
              test_writer.add_summary(d_ls_sum, step)
              test_writer.add_summary(g_ls_sum, step)
              test_writer.add_summary(im_sum, step)
              test_writer.flush()
            else:
              loss_value_D, loss_value_G = self.sess.run([self.loss_D_, self.loss_G_], feed_dict=feed_dict)
            duration = time.time() - start_time

            examples_per_sec = opts['batch_size'] / float(duration)
            format_str = 'test: %s: step %d, loss_D = %.4f, loss_G = %.4f (%.1f examples/sec) %.3f sec/batch'
            tf.logging.info(format_str % (datetime.now(), step, loss_value_D, loss_value_G,
                            examples_per_sec, duration))
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

  
class TEST(object):

  def __init__(self, sess, config, cell_info, vae_dim, global_step, training, dtype=tf.float32, name='TEST'):
    # super(IMMModel, self).__init__(dtype, name)
    self.config = config.model
    self.train_config = config.training
    self._global_step = global_step
    self.sess = sess
    self.cell_info = cell_info
    self.vae_dim = vae_dim
    self.train_mode = training
    self.colors = None
 
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
    embeddings = self.image_encoder(im)
    im_emb_size = embeddings[-2].shape.as_list()[-3:]
    embeddings = tf.expand_dims(embeddings[-2], 1)
    embeddings = tf.tile(embeddings, [1, 32, 1, 1, 1])
    embeddings = tf.reshape(embeddings, [-1]+im_emb_size)

    first_pt = self.pose_encoder(im)
    first_pt = tf.reshape(first_pt, [-1, self.config.n_maps*2])

    z = tf.random_normal([tf.shape(first_pt)[0]]+[self.vae_dim], 0, 1, dtype=tf.float32)
    pred_seq = self.vae_decoder(z, first_pt, act_code)
    pred_seq = tf.reshape(pred_seq, [-1, 32, self.config.n_maps, 2])
    rd_sz = self.config.render_input_sz

    current_pt_map = get_gaussian_maps(tf.reshape(first_pt, [-1, self.config.n_maps, 2]), \
                        [rd_sz, rd_sz], 1.0 / self.config.gauss_std, mode=gauss_mode)
    emb_size = current_pt_map.shape.as_list()[-3:]
    current_pt_map = tf.expand_dims(current_pt_map, 1)
    current_pt_map = tf.tile(current_pt_map, [1, 32, 1, 1, 1])
    current_pt_map = tf.reshape(current_pt_map, [-1]+emb_size)
    pred_pt_map = get_gaussian_maps(tf.reshape(pred_seq, [-1, self.config.n_maps, 2]), \
                        [rd_sz, rd_sz], 1.0 / self.config.gauss_std, mode=gauss_mode)

    joint_embedding = tf.concat([embeddings, current_pt_map, pred_pt_map], axis = -1)
    crude_output, mask = self.translator(joint_embedding)
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

    self.tensor = {'real_im_seq': im_seq, 'im': im, 'pred_im_seq': tf.reshape(final_output, [-1,32,128,128,3]),\
                   'mask': tf.reshape(mask, [-1,32,128,128,1]), 'pred_im_crude': tf.reshape(crude_output, [-1,32,128,128,3]),\
                   'current_points': tf.reshape(self.colorize_landmark_maps(current_landmarks_map), [-1,128,128,3]),\
                   'future_points': tf.reshape(self.colorize_landmark_maps(future_landmarks_map), [-1,32,128,128,3]),
                   'fut_pt_raw': pred_seq}
    

  def evaluate(self, opts, test_dataset, handle_pl, checkpoint_fnames, vars_to_restore=None, reset_global_step=True):
   
    tf.logging.set_verbosity(tf.logging.INFO)
    # define iterators
    test_iterator = test_dataset.make_initializable_iterator()
    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    self.sess.run([global_init,local_init])

    # set up iterators
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
