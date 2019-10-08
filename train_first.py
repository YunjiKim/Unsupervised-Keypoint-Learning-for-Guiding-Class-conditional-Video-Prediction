from __future__ import print_function
from __future__ import absolute_import

import tensorflow.contrib.slim as slim
import tensorflow as tf
import os.path as osp

# network definition:
from unsup_img_trans.models.imm_model import IMMModel
from unsup_img_trans.utils.box import Box
from unsup_img_trans.utils.colorize import colorize
from unsup_img_trans.datasets import penn_dataset

import argparse
import metayaml

config = Box(metayaml.read(['configs/experiments/penn-40pts.yaml']))
train_config = config.training
data_dir = train_config.datadir
NUM_STEPS = 30000000

config.model.n_maps = 40
train_config.datadir = '/dataset/imm/penn'

opts = {}
opts['n_summary'] = 500 # number of iterations after which to run the summary-op
opts['n_test'] = 500
opts['n_checkpoint'] = 20000 # number of iteration after which to save the model
opts['batch_size'] = 32

###########################
opts['log_dir'] = 'data/testtt'
checkpoint_fname = ''
###########################

session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_config.gpu_options.allow_growth = True
    
# open session
with tf.Session(config=session_config) as sess:
    
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # dynamic import of a dataset class
    dset_class = penn_dataset.PENNDataset

    train_dset = dset_class(train_config.datadir, subset='train')
    train_dset = train_dset.get_dataset(opts['batch_size'], repeat=True, shuffle=True, num_preprocess_threads=12)

    test_dset = dset_class(train_config.datadir, subset='test', order_stream= True, max_samples=1000)
    test_dset = test_dset.get_dataset(opts['batch_size'], repeat=False, shuffle=False, num_preprocess_threads=12)
    
    # set up inputs
    training_pl = tf.placeholder(tf.bool)
    handle_pl = tf.placeholder(tf.string, shape=[])
    base_iterator = tf.data.Iterator.from_string_handle(handle_pl, train_dset.output_types, train_dset.output_shapes)
    inputs = base_iterator.get_next()
    
    model = IMMModel(sess, config, global_step, training=True)
    model.build(inputs)

    # show network architecture
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    model.train_loop(opts, train_dset, test_dset, training_pl, handle_pl, NUM_STEPS, checkpoint_fname)
    
    
