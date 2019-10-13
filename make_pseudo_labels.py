from __future__ import print_function
from __future__ import absolute_import

import tensorflow.contrib.slim as slim
import tensorflow as tf
import os.path as osp
import argparse
import yaml

from model import EXT_KP
from data_loader import Dataset


config = yaml.load(open('configs/experiments/penn.yaml', 'r'), Loader=yaml.FullLoader)
train_config = config.training
data_dir = train_config.datadir
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_config.gpu_options.allow_growth = True




  
class EXT_KP(object):
  def __init__(self, sess, config, global_step, training, dtype=tf.float32, name='EXT_KP'):
    self.config = config.model
    self.train_config = config.training
    self._global_step = global_step
    self.sess = sess
    self.train_mode = training
    
  def build(self, inputs, output_tensors=False, build_loss=True):
    im, idx, len_ = inputs['image'], inputs['idx'], inputs['len']
    pts = self.pose_encoder(tf.reshape(im, [-1, 128,128,3]))
    self.tensor = {'pts': tf.reshape(pts, [-1, 663, 40, 2]), 'idx': idx, 'len': len_, 'im':im}

  def extract(self, opts, train_dataset, test_dataset, training_pl, handle_pl, checkpoint_fname, vars_to_restore=None, reset_global_step=True):
   
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

    # restore checkpoint:
    if tf.gfile.Exists(checkpoint_fname) or tf.gfile.Exists(checkpoint_fname + '.index'):
      print('RESTORING MODEL from: ' + checkpoint_fname)
      reader = tf.train.NewCheckpointReader(checkpoint_fname)
      vars_to_restore = tf.global_variables()
      checkpoint_vars = reader.get_variable_to_shape_map().keys()
      vars_ignored = [
          v.name for v in vars_to_restore if v.name[:-2] not in checkpoint_vars]
      print(colorize('vars-IGNORED (not restoring):', 'blue', bold=True))
      print(colorize(', '.join(vars_ignored), 'blue'))
      vars_to_restore = [
          v for v in vars_to_restore if v.name[:-2] in checkpoint_vars]
      restorer = tf.train.Saver(var_list=vars_to_restore)
      restorer.restore(self.sess, checkpoint_fname)
    else:
      raise Exception('model file does not exist at: ' + checkpoint_fname)

    ########## compute output from trainset ##########
    feed_dict = {handle_pl: train_handle, training_pl: False}
    self.sess.run(train_iterator.initializer)
    step = 0
    while True:
      try:
        outputs = self.sess.run(self.tensor, feed_dict=feed_dict)
        np.save('/workspace/imm/data/datasets/penn/unsup_pts_new/'+'{:04d}'.format(outputs['idx'][0])+'.npy',\
               outputs['pts'][0,:outputs['len'][0],::])
      except tf.errors.OutOfRangeError:
        print('iteration through train set finished')
        break

    ########## compute output from testset ##########
    feed_dict = {handle_pl: test_handle, training_pl: False}
    self.sess.run(test_iterator.initializer)
    while True:
      try:
        outputs = self.sess.run(self.tensor, feed_dict=feed_dict)
        np.save('/workspace/imm/data/datasets/penn/unsup_pts_new/'+'{:04d}'.format(outputs['idx'][0])+'.npy',\
               outputs['pts'][0,:outputs['len'][0],::])
      except tf.errors.OutOfRangeError:
        print('iteration through test set finished')
        break
  

with tf.Session(config=session_config) as sess:
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # import dataset
    train_dset = Dataset(train_config.datadir, subset='train', order_stream= True)
    train_dset = train_dset.get_dataset(opts['batch_size'], repeat=False, shuffle=False, num_preprocess_threads=12)
    test_dset = Dataset(train_config.datadir, subset='test', order_stream= True, max_samples=1000)
    test_dset = test_dset.get_dataset(opts['batch_size'], repeat=False, shuffle=False, num_preprocess_threads=12)
    
    # set up
    training_pl = tf.placeholder(tf.bool)
    handle_pl = tf.placeholder(tf.string, shape=[])
    base_iterator = tf.data.Iterator.from_string_handle(handle_pl, train_dset.output_types, train_dset.output_shapes)
    inputs = base_iterator.get_next()
    model = EXT_KP(sess, config, global_step, training=False)
    model.build(inputs)

    output_ = model.extract(opts, train_dset, test_dset, training_pl, handle_pl, checkpoint_fname)
