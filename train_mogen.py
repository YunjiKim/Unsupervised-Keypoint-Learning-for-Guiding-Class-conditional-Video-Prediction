from __future__ import print_function
from __future__ import absolute_import

import tensorflow.contrib.slim as slim
import tensorflow as tf
import os.path as osp
import argparse
import yaml

from model import MOGEN
from data_loader import Dataset


parser = argparse.ArgumentParser()
parser.add_argument('--config_root', type=str, required=True, help='path of the configuration file')
args = parser.parse_args()

config = yaml.load(open(args.config_root, 'r'), Loader=yaml.FullLoader)
train_config = config.training
data_dir = train_config.datadir
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_config.gpu_options.allow_growth = True
    
    
# open session
with tf.Session(config=session_config) as sess:
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # import dataset
    train_dset = Dataset(train_config.datadir, subset='train', n_action=opts['n_action'], n_landmark=opts['n_landmark'])
    train_dset = train_dset.get_dataset(opts['batch_size'], repeat=True, shuffle=True, num_preprocess_threads=4)
    test_dset = Dataset(train_config.datadir, subset='test', n_action=opts['n_action'], n_landmark=opts['n_landmark'], order_stream= True, max_samples=1000)
    test_dset = test_dset.get_dataset(opts['batch_size'], repeat=False, shuffle=False, num_preprocess_threads=4)
    
    # set up
    training_pl = tf.placeholder(tf.bool)
    handle_pl = tf.placeholder(tf.string, shape=[])
    base_iterator = tf.data.Iterator.from_string_handle(handle_pl, train_dset.output_types, train_dset.output_shapes)
    inputs = base_iterator.get_next()
    model = MOGEN(sess, config, opts['cell_info'], opts['vae_dim'], global_step, training=True)
    model.build(inputs)

    # start training
    model.train_loop(opts, train_dset, test_dset, training_pl, handle_pl, checkpoint_fname)
