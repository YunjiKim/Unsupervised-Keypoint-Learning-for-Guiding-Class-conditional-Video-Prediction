from os import path as osp
import os
import random

import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.io import loadmat

from .base_dataloader import BaseDataLoader
from utils import data as data_utils

# TODO: make image size as configurable parameter
IMAGE_SIZE = 128


class ImagePairDataLoader(BaseDataLoader):

    def __init__(self, data_dir, subset,
                 random_order=True,
                 randomness=False):
        super(ImagePairDataLoader, self).__init__()

        self._data_dir = data_dir
        self._random_order = random_order
        self._randomness = randomness
        # TODO: split random crop, random filter, random rotate as a separate options

        with open(osp.join(data_dir, subset + '_set.txt'), 'r') as f:
            self._images = f.read().splitlines()

        self._total = len(self._images)
        print(subset + 'set : ', self._total)
        pass

    def length(self):
        return self._total

    def get_sample_shape(self):
        return {
            'image': [128, 128, 3],
            'future_image': [128, 128, 3]
        }

    def get_sample_dtype(self):
        return {
            'image': tf.float32,
            'future_image': tf.float32
        }

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
            return {
                'image': inputs['image'] * 2.0 - 1.0,
                'future_image': inputs['future_image'] * 2.0 - 1.0
            }

    def _get_image_at(self, idx):
        img_path, n_act = self._images[idx].split()
        file_len = len(os.listdir(osp.join(self._data_dir, img_path)))

        im_idx = 0
        fu_im_idx = 10

        if self._random_order:
            rand_interval = random.randint(8, 11)
            im_idx = random.randint(0, file_len - 1)
            fu_im_idx = (im_idx + rand_interval) % file_len

        # load images and add randomness
        image = Image.open(osp.join(self._data_dir, img_path, '{:06d}'.format(im_idx + 1) + '.jpg'))
        future_image = Image.open(osp.join(self._data_dir, img_path, '{:06d}'.format(fu_im_idx + 1) + '.jpg'))

        # add randomness
        w, h = image.size

        # random rotate image and the points
        if self._randomness:
            rand_val = random.randrange(-10, 11)
            image = image.rotate(rand_val)
            future_image = future_image.rotate(rand_val)

            ox, oy = image.size
            ox /= 2.0
            oy /= 2.0

        if w > h:
            ratio = h / float(IMAGE_SIZE)

            image = image.resize([int(w / ratio), int(h / ratio)])
            future_image = future_image.resize([int(w / ratio), int(h / ratio)])

            if self._randomness:
                
                crop_rand_val = random.randint(0, int(w/ratio-IMAGE_SIZE))
                should_flip = random.randint(0, 1)

                image = image.crop((crop_rand_val, 0, crop_rand_val + IMAGE_SIZE, IMAGE_SIZE))
                future_image = future_image.crop((crop_rand_val, 0, crop_rand_val + IMAGE_SIZE, IMAGE_SIZE))

                if should_flip:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    future_image = future_image.transpose(Image.FLIP_LEFT_RIGHT)

                image, future_image = data_utils.apply_random_filter([image, future_image])

            else:
                # just do the center crop
                ox, oy = image.size
                ox /= 2.0
                half_target_size = IMAGE_SIZE // 2

                image = image.crop((ox - half_target_size, 0, ox + half_target_size, IMAGE_SIZE))
                future_image = future_image.crop((ox - half_target_size, 0, ox + half_target_size, IMAGE_SIZE))
        else:

            ratio = w / float(IMAGE_SIZE)

            image = image.resize([int(w / ratio), int(h / ratio)])
            future_image = future_image.resize([int(w / ratio), int(h / ratio)])

            if self._randomness:

                crop_rand_val = random.randint(0, int(h/ratio-IMAGE_SIZE))
                should_flip = random.randint(0, 1)

                image = image.crop((0, crop_rand_val, IMAGE_SIZE, crop_rand_val + IMAGE_SIZE))
                future_image = future_image.crop((0, crop_rand_val, IMAGE_SIZE, crop_rand_val + IMAGE_SIZE))

                if should_flip:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    future_image = future_image.transpose(Image.FLIP_LEFT_RIGHT)

                image, future_image = data_utils.apply_random_filter([image, future_image])

            else:
                # just do the center crop
                ox, oy = image.size
                ox /= 2.0
                half_target_size = IMAGE_SIZE // 2

                image = image.crop((ox - half_target_size, 0, ox + half_target_size, IMAGE_SIZE))
                future_image = future_image.crop((ox - half_target_size, 0, ox + half_target_size, IMAGE_SIZE))

        image = np.asarray(image)
        future_image = np.asarray(future_image)

        return {
            'image': image / 255.0,
            'future_image': future_image / 255.0
        }
