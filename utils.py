import os
import scipy
import numpy as np
import tensorflow as tf

from config import cfg
import rb_data


def load_mnist(path, is_training):
    if is_training:
        data = rb_data.read_data_sets("",one_hot=False)
        trX = data.train.images
        trY = data.train.labels
        return trX, trY
    else:
        TEST_IMAGES = 'test_data/rb.HOT.15m(1).csv'
        data = rb_data.read_test_sets(TEST_IMAGES,one_hot=False)
        teX = data.images
        teY = data.labels
        return teX, teY


def get_batch_data():
    trX, trY = load_mnist(cfg.dataset, cfg.is_training)

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=cfg.num_threads,
                                  batch_size=cfg.batch_size,
                                  capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


if __name__ == '__main__':
    X, Y = load_mnist(cfg.dataset, cfg.is_training)
    print(X.get_shape())
    print(X.dtype)
