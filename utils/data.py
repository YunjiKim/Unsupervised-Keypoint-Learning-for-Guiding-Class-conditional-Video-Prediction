import random
import math

from PIL import ImageFilter, ImageEnhance
import numpy as np


def apply_random_filter(images):
    r_id = random.randint(0, 9)

    if r_id == 0:
        images = [im.filter(ImageFilter.DETAIL) for im in images]
    elif r_id == 1:
        images = [im.filter(ImageFilter.EDGE_ENHANCE) for im in images]
    elif r_id == 2:
        images = [im.filter(ImageFilter.SMOOTH) for im in images]
    elif r_id == 3:
        images = [im.filter(ImageFilter.SMOOTH_MORE) for im in images]
    elif r_id == 4:
        images = [im.filter(ImageFilter.EDGE_ENHANCE_MORE) for im in images]
    elif r_id == 5:
        images = [im.filter(ImageFilter.BLUR) for im in images]
    elif r_id == 6:
        r_val = random.randint(0, 50)
        images = [ImageEnhance.Sharpness(im).enhance(r_val * 0.1) for im in images]
    elif r_id == 7:
        r_val = random.randint(7, 20)
        images = [ImageEnhance.Brightness(im).enhance(r_val * 0.1) for im in images]
    elif r_id == 8:
        r_val = random.randint(0, 50)
        images = [ImageEnhance.Color(im).enhance(r_val * 0.1) for im in images]
    else:
        r_val = random.randint(7, 30)
        images = [ImageEnhance.Contrast(im).enhance(r_val * 0.1) for im in images]

    return images


def center_crop(image, target_size):
    w, h = image.size
    half_target_size = target_size // 2

    if w > h:
        # calculating crop size
        ratio = h / float(target_size)
        ox, oy = int(w / ratio), int(h / ratio)
        ox /= 2.0

        crop_size = (ox - half_target_size, 0, ox + half_target_size, target_size)

    else:
        # calculating crop size
        ratio = w / float(target_size)
        ox, oy = int(w / ratio), int(h / ratio)
        oy /= 2.0

        crop_size = (0, oy - half_target_size, target_size, oy + half_target_size)
        
    return crop_size, ratio


def rotate_keypoints(keypoints, rand_val, ox=0, oy=0):
    qx = ox \
         + math.cos(math.radians(-rand_val)) * (keypoints[..., 0] - ox) \
         - math.sin(math.radians(-rand_val)) * (keypoints[..., 1] - oy)
    qy = oy \
         + math.sin(math.radians(-rand_val)) * (keypoints[..., 0] - ox) \
         + math.cos(math.radians(-rand_val)) * (keypoints[..., 1] - oy)

    return np.concatenate([np.expand_dims(qx, -1), np.expand_dims(qy, -1)], axis=-1)


def create_one_hot_label(n_classes, idx):
    label = np.zeros(n_classes)
    label[int(idx)] = 1

    return label
