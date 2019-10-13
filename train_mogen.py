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
    real_ = self.seq_discr(gt_seq)
    fake_ = self.seq_discr(pred_seq)
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
    fake_ = self.seq_discr(pred_seq)
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
