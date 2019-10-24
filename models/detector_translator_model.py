import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from .base_model import BaseModel
from . import networks
from utils import model as model_utils


class DetectorTranslatorModel(BaseModel):

    # name will be used for log management (defined in BaseModel class)
    name = 'detector_translator'

    def __init__(self, config, global_step=None, is_training=True):
        super(DetectorTranslatorModel, self).__init__(is_training)

        # configuration variables
        train_config = config['training']
        model_config = config['model']
        paths_config = config['paths']

        if self.is_training:
            self.lr = train_config['lr']
        else:
            self.lr = None
        self.batch_size = train_config['batch_size']
        self.n_points = model_config['n_pts']
        self.log_dir = paths_config['log_dir']
        self.vgg19_path = paths_config['vggnet']

        self.colors = model_utils.get_n_colors(model_config['n_pts'], pastel_factor=0.0)

        # inputs
        self.global_step = global_step
        self.input_im = None
        self.input_future_im = None

        # outputs
        self.current_lr = None
        self.final_output = None
        self.crude_output = None
        self.mask = None
        self.current_landmarks_map = None
        self.future_landmarks_map = None

        # losses
        self.loss_D_real = None
        self.loss_D_fake = None
        self.loss_D = None
        self.loss_G_recon = None
        self.loss_G_adv = None
        self.loss_G = None

        # ops
        self.train_op_D = None
        self.train_op_G = None

        # summaries
        self.summary_lr = None
        self.summary_image = None
        self.summary_d_loss = None
        self.summary_g_loss = None
        pass

    def build(self, inputs):
        # input setup
        self.input_im = inputs['image']
        self.input_future_im = inputs['future_image']

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
        im = self.input_im
        future_im = self.input_future_im

        # keypoints detection
        embeddings = networks.image_encoder(im, self.is_training)
        current_gauss_pt = networks.pose_encoder(im, self.n_points, self.is_training)
        future_gauss_pt = networks.pose_encoder(future_im, self.n_points, self.is_training)
        current_pt_map = model_utils.get_gaussian_maps(current_gauss_pt, [32, 32])
        future_pt_map = model_utils.get_gaussian_maps(future_gauss_pt, [32, 32])
        joint_embedding = tf.concat([embeddings[-2], current_pt_map, future_pt_map], axis=-1)

        # image translation
        crude_output, mask = networks.translator(joint_embedding, self.is_training)
        final_output = im * mask + crude_output * (1 - mask)

        current_landmarks_map = model_utils.get_gaussian_maps(current_gauss_pt, [128, 128])
        future_landmarks_map = model_utils.get_gaussian_maps(future_gauss_pt, [128, 128])

        self.final_output = final_output
        self.crude_output = crude_output
        self.mask = mask
        self.current_landmarks_map = current_landmarks_map
        self.future_landmarks_map = future_landmarks_map
        pass

    def _compute_loss(self):
        self._compute_loss_D(self.final_output, self.input_future_im)
        self._compute_loss_G(self.final_output, self.input_future_im)

        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'img_discr' not in var.name]
        D_vars = [var for var in t_vars if 'img_discr' in var.name]
        lr = tf.train.exponential_decay(
            self.lr['start_val'], self.global_step, self.lr['step'], self.lr['decay']
        )

        self.current_lr = lr
        self.train_op_D = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(self.loss_D, var_list=D_vars)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op_G = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(self.loss_G, var_list=G_vars,
                                                                                          global_step=self.global_step)
        pass

    def _define_summary(self):
        # output summaries
        current_points = model_utils.colorize_landmark_maps(self.current_landmarks_map, self.colors)
        future_points = model_utils.colorize_landmark_maps(self.future_landmarks_map, self.colors)
        summary_current_points = tf.summary.image('current_points', current_points, max_outputs=2)
        summary_future_points = tf.summary.image('future_points', future_points, max_outputs=2)
        summary_crude_image = tf.summary.image('future_im_crude',
                                               tf.clip_by_value((self.crude_output + 1) / 2.0 * 255.0, 0, 255),
                                               max_outputs=2)
        summary_pred_im = tf.summary.image('future_im_pred',
                                           tf.clip_by_value((self.final_output + 1) / 2.0 * 255.0, 0, 255),
                                           max_outputs=2)
        summary_mask = tf.summary.image('mask', self.mask * 255.0, max_outputs=2)
        future_im_sum = tf.summary.image('future_im',
                                         (self.input_future_im + 1) / 2.0 * 255.0,
                                         max_outputs=2)
        im_sum = tf.summary.image('im',
                                  (self.input_im + 1) / 2.0 * 255.0,
                                  max_outputs=2)

        # D summaries
        summary_d_real = tf.summary.scalar('D_real', self.loss_D_real)
        summary_d_fake = tf.summary.scalar('D_fake', self.loss_D_fake)

        # G summaries
        summary_g_recon = tf.summary.scalar('reconstruction_metric', self.loss_G_recon)
        summary_g_adv = tf.summary.scalar('G_adv_loss', self.loss_G_adv)

        # setting final summary results
        self.summary_lr = tf.summary.scalar('lr', self.current_lr)
        self.summary_image = tf.summary.merge([future_im_sum,
                                               im_sum,
                                               summary_current_points,
                                               summary_future_points,
                                               summary_crude_image,
                                               summary_pred_im,
                                               summary_mask])
        self.summary_d_loss = tf.summary.merge([summary_d_real, summary_d_fake])
        self.summary_g_loss = tf.summary.merge([summary_g_recon, summary_g_adv])
        pass

    def _compute_loss_D(self, future_im_pred, future_im):
        real_ = networks.img_discr(future_im)
        fake_ = networks.img_discr(future_im_pred)
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_), logits=real_)
        )
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_), logits=fake_)
        )

        self.loss_D_real = real_loss
        self.loss_D_fake = fake_loss
        self.loss_D = real_loss + fake_loss
        pass

    def _compute_loss_G(self, future_im_pred, future_im):
        reconstruction_loss = self._compute_perceptual_loss((future_im + 1) / 2.0 * 255.0,
                                                            (future_im_pred + 1) / 2.0 * 255.0)
        fake_ = networks.img_discr(future_im_pred)
        adv_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_), logits=fake_)
        )

        self.loss_G_recon = reconstruction_loss
        self.loss_G_adv = adv_loss
        self.loss_G = reconstruction_loss + adv_loss
        pass

    def _compute_perceptual_loss(self, gt_image, pred_image):
        vgg = networks.Vgg19(self.vgg19_path)

        with tf.variable_scope("content_vgg"):
            ims = tf.concat([gt_image, pred_image], axis=0)
            feats = vgg.build(ims)
            feat_gt, feat_pred = zip(*[tf.split(f, 2, axis=0) for f in feats])
            losses = []
            for k in range(len(feats)):
                l = tf.abs(feat_gt[k] - feat_pred[k])
                l = tf.reduce_mean(l)
                losses.append(l)
                pass
            loss = tf.reduce_mean(losses)

        return loss
