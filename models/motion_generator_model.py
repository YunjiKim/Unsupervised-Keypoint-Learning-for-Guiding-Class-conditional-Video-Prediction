import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from .base_model import BaseModel
from . import networks
from utils import model as model_utils

N_FUTURE_FRAMES = 32


class MotionGeneratorModel(BaseModel):
    # name will be used for log management (defined in BaseModel class)
    name = 'motion_generator'

    def __init__(self, config, global_step=None, is_training=True):
        super(MotionGeneratorModel, self).__init__(is_training)

        # configuration variables
        train_config = config['training']
        model_config = config['model']
        paths_config = config['paths']

        if self.is_training:
            self.lr = train_config['lr']
        else:
            self.lr = None
        self.batch_size = train_config['batch_size']
        self.log_dir = paths_config['log_dir']

        # model config variables
        self.n_points = model_config['n_pts']
        self.cell_info = model_config['cell_info']
        self.vae_dim = model_config['vae_dim']

        self.colors = model_utils.get_n_colors(model_config['n_pts'], pastel_factor=0.0)

        # inputs
        self.global_step = global_step
        self.input_im = None
        self.input_landmarks = None
        self.input_real_seq = None
        self.input_action_code = None

        # outputs
        self.current_lr = None
        self.pred_seq = None

        # losses
        self.loss_D_real = None
        self.loss_D_fake = None
        self.loss_D = None
        self.loss_G_recon = None
        self.loss_G_kl = None
        self.loss_G_adv = None
        self.loss_G = None

        # summaries
        self.summary_lr = None
        self.summary_image = None
        self.summary_d_loss = None
        self.summary_g_loss = None
        pass

    def build(self, inputs):
        # input setup
        self.input_im = inputs['image']
        self.input_landmarks = inputs['landmarks']
        self.input_real_seq = inputs['real_seq']
        self.input_action_code = inputs['action_code']

        self._define_forward_pass()
        if self.is_training:
            self._compute_loss()
            self._define_summary()
        pass

    def train_step(self,
                   sess,
                   feed_dict,
                   step,
                   batch_size,
                   should_write_log=False, should_write_summary=False):
        D_ops = [self.loss_D, self.train_op_D]
        G_ops = [self.loss_G, self.train_op_G]

        if should_write_summary:
            D_ops.extend([self.summary_d_loss])
            G_ops.extend([self.summary_g_loss, self.summary_image, self.summary_lr])

        start_time = time.time()
        D_values = sess.run(D_ops, feed_dict=feed_dict)
        G_values = sess.run(G_ops, feed_dict=feed_dict)
        duration = time.time() - start_time

        if should_write_log:
            examples_per_sec = batch_size / float(duration)
            loss_value_D = D_values[0]
            loss_value_G = G_values[0]
            log_format = '%s: step %d, loss_D = %.4f, loss_G = %.4f (%.1f examples/sec) %.3f sec/batch'
            tf.logging.info(log_format % (datetime.now(),
                                          step,
                                          loss_value_D,
                                          loss_value_G,
                                          examples_per_sec,
                                          duration))
        if should_write_summary:
            summary_loss_D = D_values[2]
            summary_loss_G = G_values[2]
            summary_image = G_values[3]
            summary_lr = G_values[4]
            self.train_writer.add_summary(summary_loss_D, step)
            self.train_writer.add_summary(summary_loss_G, step)
            self.train_writer.add_summary(summary_image, step)
            self.train_writer.add_summary(summary_lr, step)
        pass

    def test_step(self, sess, feed_dict, step, test_idx, batch_size):
        ops = [self.loss_D, self.loss_G]
        should_write_summary = test_idx == 0

        if should_write_summary:
            ops.extend([self.summary_d_loss, self.summary_g_loss, self.summary_image])

        start_time = time.time()
        values = sess.run(ops, feed_dict=feed_dict)
        duration = time.time() - start_time

        if should_write_summary:
            summary_loss_D = values[2]
            summary_loss_G = values[3]
            summary_image = values[4]
            self.test_writer.add_summary(summary_loss_D, step)
            self.test_writer.add_summary(summary_loss_G, step)
            self.test_writer.add_summary(summary_image, step)

        loss_value_D = values[0]
        loss_value_G = values[1]

        return loss_value_D, loss_value_G, duration, batch_size

    def collect_test_results(self, results, step):
        average_loss_D = sum([x[0] for x in results]) / len(results)
        average_loss_G = sum([x[1] for x in results]) / len(results)
        total_duration = sum([x[2] for x in results])
        average_duration = total_duration / len(results)
        num_examples = sum([x[3] for x in results])
        examples_per_sec = num_examples / total_duration

        log_format = 'test: %s: step %d, loss_D = %.4f, loss_G = %.4f (%.1f examples/sec) %.3f sec/batch'
        tf.logging.info(log_format % (datetime.now(),
                                      step,
                                      average_loss_D,
                                      average_loss_G,
                                      examples_per_sec,
                                      average_duration))
        pass

    def _define_forward_pass(self):
        landmarks = self.input_landmarks
        real_seq = self.input_real_seq
        action_code = self.input_action_code

        first_pt = tf.reshape(landmarks, [-1, self.n_points * 2])
        if self.is_training:
            mu, stddev = networks.vae_encoder(tf.reshape(real_seq, [-1, N_FUTURE_FRAMES, self.n_points * 2]),
                                              first_pt,
                                              action_code,
                                              self.cell_info,
                                              self.vae_dim)
            z = mu + stddev * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
            pred_seq = networks.vae_decoder(z,
                                            first_pt,
                                            action_code,
                                            self.cell_info,
                                            self.vae_dim,
                                            self.n_points)
        else:
            mu, stddev = None, None
            z = tf.random_normal([tf.shape(first_pt)[0]] + [self.vae_dim], 0, 1, dtype=tf.float32)
            pred_seq = networks.vae_decoder(z,
                                            first_pt,
                                            action_code,
                                            self.cell_info,
                                            self.vae_dim,
                                            self.n_points)

        # outputs
        self.pred_seq = pred_seq
        if self.is_training:
            self.mu = mu
            self.stddev = stddev
        pass

    def _compute_loss(self):
        real_seq = tf.reshape(self.input_real_seq, [-1, N_FUTURE_FRAMES, self.n_points * 2])
        self._compute_loss_D(self.pred_seq, real_seq)
        self._compute_loss_G(self.pred_seq, real_seq, self.mu, self.stddev)

        # optimization
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'discr' not in var.name]
        D_vars = [var for var in t_vars if 'discr' in var.name]
        lr = tf.train.exponential_decay(self.lr['start_val'], self.global_step, self.lr['step'],
                                        self.lr['decay'])

        self.current_lr = lr
        self.train_op_D = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(self.loss_D,
                                                                                      var_list=D_vars)
        self.train_op_G = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(self.loss_G,
                                                                                      var_list=G_vars,
                                                                                      global_step=self.global_step)
        pass

    def _define_summary(self):
        # D loss summary
        summary_d_real = tf.summary.scalar('D_real', self.loss_D_real)
        summary_d_fake = tf.summary.scalar('D_fake', self.loss_D_fake)
        summary_d_loss = tf.summary.merge([summary_d_real, summary_d_fake])

        # G loss summary
        summary_g_kl = tf.summary.scalar('kl_loss', self.loss_G_kl)
        summary_g_recon = tf.summary.scalar('recon_loss', self.loss_G_recon)
        summary_g_adv = tf.summary.scalar('G_adv_loss', self.loss_G_adv)
        summary_g_loss = tf.summary.merge([summary_g_recon, summary_g_kl, summary_g_adv])

        # final summaries to write
        self.summary_lr = tf.summary.scalar('lr', self.current_lr)
        self.summary_image = self._get_image_visualization_summary()
        self.summary_d_loss = summary_d_loss
        self.summary_g_loss = summary_g_loss
        pass

    def _get_image_visualization_summary(self):
        pred_seq = self.pred_seq
        real_seq = self.input_real_seq
        landmarks = self.input_landmarks

        # convert pred_seq to images
        pred_seq_img = []
        for i in range(N_FUTURE_FRAMES):
            gauss_map = model_utils.get_gaussian_maps(tf.reshape(pred_seq[:, i, ::], [-1, self.n_points, 2]),
                                                      [64, 64])
            pred_seq_img.append(model_utils.colorize_landmark_maps(gauss_map, self.colors))
            pass
        pred_seq_img = tf.concat(pred_seq_img, axis=2)

        # convert real_seq to images
        real_seq_img = []
        for i in range(N_FUTURE_FRAMES):
            gauss_map = model_utils.get_gaussian_maps(real_seq[:, i, ::], [64, 64])
            real_seq_img.append(model_utils.colorize_landmark_maps(gauss_map, self.colors))
            pass
        real_seq_img = tf.concat(real_seq_img, axis=2)

        first_pt_map = model_utils.get_gaussian_maps(tf.reshape(landmarks, [-1, self.n_points, 2]), [128, 128])

        # image summary
        summary_im = tf.summary.image('im', (self.input_im + 1) / 2.0 * 255.0, max_outputs=2)
        summary_first_pt = tf.summary.image('first_pt',
                                            model_utils.colorize_landmark_maps(first_pt_map, self.colors),
                                            max_outputs=2)
        summary_pred_p_seq = tf.summary.image('predicted_pose_sequence',
                                              model_utils.colorize_landmark_maps(pred_seq_img, self.colors),
                                              max_outputs=2)
        summary_real_p_seq = tf.summary.image('real_pose_sequence',
                                              model_utils.colorize_landmark_maps(real_seq_img, self.colors),
                                              max_outputs=2)

        return tf.summary.merge([summary_im,
                                 summary_first_pt,
                                 summary_pred_p_seq,
                                 summary_real_p_seq])

    def _compute_loss_D(self, pred_seq, real_seq):
        real_ = networks.seq_discr(real_seq)
        fake_ = networks.seq_discr(pred_seq)
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_), logits=real_)
        )
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_), logits=fake_)
        )
        loss = real_loss + fake_loss

        self.loss_D_real = real_loss
        self.loss_D_fake = fake_loss
        self.loss_D = loss
        pass

    def _compute_loss_G(self, pred_seq, real_seq, mu, stddev):
        l = 1000 * tf.abs(pred_seq - real_seq)
        pred_seq_loss = tf.reduce_mean(l)

        kl_l = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(stddev) - tf.log(1e-8 + tf.square(stddev)) - 1, 1)
        kl_loss = tf.reduce_mean(kl_l)
        fake_ = networks.seq_discr(pred_seq)
        adv_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_), logits=fake_)
        )

        self.loss_G_recon = pred_seq_loss
        self.loss_G_kl = kl_loss
        self.loss_G_adv = adv_loss
        self.loss_G = kl_loss + pred_seq_loss + adv_loss
        pass
