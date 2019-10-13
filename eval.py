from __future__ import print_function
from __future__ import absolute_import

import tensorflow.contrib.slim as slim
import tensorflow as tf
import os.path as osp
import argparse
import yaml

from model import EVAL
from data_loader import Dataset


config = yaml.load(open('configs/experiments/penn.yaml', 'r'), Loader=yaml.FullLoader)
train_config = config.training
data_dir = train_config.datadir
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_config.gpu_options.allow_growth = True


  
class EVAL(object):

  def __init__(self, sess, config, cell_info, vae_dim, global_step, training, dtype=tf.float32, name='EVAL'):
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



# open session
with tf.Session(config=session_config) as sess:
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # import dataset
    train_dset = dset_class(train_config.datadir, subset='train',\
                            n_action=opts['n_action'], n_landmark=opts['n_landmark'], order_stream= True)
    train_dset = train_dset.get_dataset(opts['batch_size'], repeat=False, shuffle=False, num_preprocess_threads=12)
    test_dset = dset_class(train_config.datadir, subset='test',\
                           n_action=opts['n_action'], n_landmark=opts['n_landmark'], order_stream= True, max_samples=1000)
    test_dset = test_dset.get_dataset(opts['batch_size'], repeat=False, shuffle=False, num_preprocess_threads=12)
    
    # set up inputs
    handle_pl = tf.placeholder(tf.string, shape=[])
    base_iterator = tf.data.Iterator.from_string_handle(handle_pl, train_dset.output_types, train_dset.output_shapes)
    inputs = base_iterator.get_next()
    model = EVAL(sess, config, opts['cell_info'], opts['vae_dim'], global_step, training=False)
    model.build(inputs)

    output_ = model.evaluate(opts, test_dset, test_dset, handle_pl, checkpoint_fname)
