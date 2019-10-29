from os import path as osp
import os
import random

import tensorflow as tf
import numpy as np
from PIL import Image

from .base_dataloader import BaseDataLoader
from utils import data as data_utils

# TODO: make image size as configurable parameter
IMAGE_SIZE = 128
N_SEQUENCE_LEN = 33


class SequenceDataLoader(BaseDataLoader):

    def __init__(self, data_dir, subset,
                 n_points, n_action,
                 with_image_seq=False,
                 random_order=True,
                 randomness=False):
        super(SequenceDataLoader, self).__init__()

        self._data_dir = data_dir
        self._random_order = random_order
        self._randomness = randomness
        self._with_image_seq = with_image_seq

        # model dependent configurations
        self.n_points = n_points
        self.n_action = n_action

        with open(osp.join(data_dir, subset + '_set.txt'), 'r') as f:
            self._images = f.read().splitlines()

        self._total = len(self._images)
        print(subset + 'set : ', self._total)
        pass

    def length(self):
        return self._total

    def get_sample_shape(self):
        shape_dict = {
            'image': [IMAGE_SIZE, IMAGE_SIZE, 3],
            'keypoints': [self.n_points, 2],
            'real_seq': [N_SEQUENCE_LEN - 1, self.n_points, 2],
            'action_code': [self.n_action]
        }
        if self._with_image_seq:
            shape_dict['real_im_seq'] = [N_SEQUENCE_LEN - 1, IMAGE_SIZE, IMAGE_SIZE, 3]

        return shape_dict

    def get_sample_dtype(self):
        dtype_dict = {
            'image': tf.float32,
            'keypoints': tf.float32,
            'real_seq': tf.float32,
            'action_code': tf.float32
        }
        if self._with_image_seq:
            dtype_dict['real_im_seq'] = tf.float32

        return dtype_dict

    def sample_generator(self):
        if self._random_order:
            current_idx = 0
            while (self._total is None) or (current_idx < self._total):
                idx = np.random.randint(len(self._images))
                yield self._get_image_at(idx)

                if self._total is not None:
                    current_idx += 1
                pass

        else:
            for idx in range(self._total):
                yield self._get_image_at(idx)

    def map_fn(self, inputs):
        with tf.name_scope('proc_im_pair'):
            inputs_dict = {
                'image': inputs['image'] * 2.0 - 1.0,
                'keypoints': inputs['keypoints'],
                'real_seq': inputs['real_seq'],
                'action_code': inputs['action_code']
            }
            if self._with_image_seq:
                inputs_dict['real_im_seq'] = inputs['real_im_seq'] * 2.0 - 1.0

            return inputs_dict

    def _get_image_at(self, idx):
        img_path, action_idx = self._images[idx].split()
        file_len = len(os.listdir(osp.join(self._data_dir, img_path)))

        keypoints = np.load(osp.join(self._data_dir, img_path.replace('frames', 'pseudo_labels') + '.npy'))
        gap = int(file_len / N_SEQUENCE_LEN)

        # start_image idx calculation
        if self._randomness:
            if gap >= 1:
                im_idx = random.randint(0, file_len - N_SEQUENCE_LEN * gap)
            else:
                n_seq = (N_SEQUENCE_LEN - 1) // 2 + 1
                im_idx = random.randint(0, file_len - n_seq)
        else:
            im_idx = 0

        # load image
        image = self._load_image(img_path, im_idx)

        # load real keypoints sequence
        if gap >= 1:
            fr_idx = [im_idx + gap * i for i in range(N_SEQUENCE_LEN)]
            real_seq = keypoints[fr_idx, :, :]

        else:
            # not enough sequence, so use same image twice
            n_seq = (N_SEQUENCE_LEN - 1) // 2 + 1
            real_seq = np.zeros([N_SEQUENCE_LEN, self.n_points, 2])
            half_real_seq = keypoints[im_idx:im_idx + n_seq, :, :]

            for i in range(n_seq - 1):
                real_seq[i * 2] = half_real_seq[i]
                real_seq[i * 2 + 1] = (half_real_seq[i] + half_real_seq[i + 1]) / 2.0
                pass
            real_seq[-1] = half_real_seq[-1]

        # random rotation
        if self._randomness:
            rotation_rand_val = random.randrange(-15, 16)
            image = image.rotate(rotation_rand_val)
            real_seq = data_utils.rotate_keypoints(real_seq, rotation_rand_val)
            pass

        # center crop
        w,h = image.size
        crop_size, ratio = data_utils.center_crop(image, IMAGE_SIZE)
        image = image.resize([int(w / ratio), int(h / ratio)]).crop(crop_size)

        # load images only if with_image_seq = True
        if self._with_image_seq:
            image_seq = []
            should_append_twice = False
            n_future_frames = (N_SEQUENCE_LEN - 1)

            if gap < 1:
                gap = 1
                should_append_twice = True
                n_future_frames //= 2

            for i in range(1, n_future_frames + 1):
                current_im = self._load_image(img_path, i * gap)
                current_im = current_im.resize([int(w / ratio), int(h / ratio)]).crop(crop_size)
                current_im = np.asarray(current_im)

                image_seq.append(np.expand_dims(current_im, 0))
                if should_append_twice:
                    image_seq.append(np.expand_dims(current_im, 0))
                pass

            image_seq = np.concatenate(image_seq, axis=0)
            image_seq = image_seq / 255.0
        else:
            image_seq = None

        # random flip
        if self._randomness:
            should_flip = random.randint(0, 1)
            if should_flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                real_seq[:, :, 0] *= -1

        image = np.asarray(image)
        action_label = data_utils.create_one_hot_label(self.n_action, action_idx)

        if self._randomness:
            rand_val = random.randint(70, 120) / 100.0
            real_seq *= rand_val

        inputs = {
            'image': image / 255.0,
            'keypoints': real_seq[0, ::],
            'real_seq': real_seq[1:, ::],
            'action_code': action_label
        }
        if self._with_image_seq:
            inputs['real_im_seq'] = image_seq

        return inputs

    def _load_image(self, img_path, idx):
        return Image.open(osp.join(self._data_dir, img_path, '{:06d}'.format(idx + 1) + '.jpg'))
