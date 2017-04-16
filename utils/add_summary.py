import tensorflow as tf
import numpy as np
from os import listdir
import os
from os.path import isfile, join
import matplotlib.pyplot as plt


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor  (for TensorBoard visualization)."""
    
    with tf.name_scope('summaries_' + name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def image_summaries(img, name):
    with tf.name_scope('summaries_image_' + name):
        tf.summary.image(name, img, 10)


def vox2img(vox, steps):
    start_z = np.sum(steps[0:len(steps) - 1])
    size = steps[-1]
    img = tf.reduce_sum(vox, [0, 0, 0, start_z], [1, 64, 64, size], 2)


def add_all_joints(prec, steps, FLAG, name='precision/'):
    summary_ = tf.Summary()
    step_c = 0
    for s, step in enumerate(steps):
        with tf.name_scope('Step_%d' % (step)):
            prec_step = tf.slice(prec, [s, 0], [1, FLAG.num_joints])
            step_c += step
            prec_step = tf.squeeze(prec_step)
            for j in xrange(FLAG.num_joints):
                tf.summary.scalar('joint_%d' % j, prec_step[j])


def get_summary_writer(FLAG, sess):
    onlyFolders = [f for f in listdir(FLAG.eval_dir + '/summaries/') if
                   isfile(join(FLAG.eval_dir + '/summaries/', f)) != 1]
    onlyFolders.sort()
    
    if (len(onlyFolders) == 0):
        return tf.summary.FileWriter(FLAG.eval_dir + '/summaries/' + '00' +
                                     '/train/', sess.graph), \
               tf.summary.FileWriter(FLAG.eval_dir + '/summaries/' + '00' +
                                     '/test/', sess.graph)
    else:
        model_name = '/%02d' % (int(onlyFolders[-1]) + 1)
        return tf.summary.FileWriter(
            FLAG.eval_dir + '/summaries/' + model_name +
            '/train/', sess.graph), \
               tf.summary.FileWriter(
                   FLAG.eval_dir + '/summaries/' + model_name +
                   '/test/', sess.graph)


def add_all(x, input, output, loss):
    tf.summary.scalar('loss', loss)
    #variable_summaries(y, 'label')
    variable_summaries(output, 'output')
    variable_summaries(input, 'input')
    # img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    img = tf.image.resize_images(x[0:1], [64, 64])
    image_summaries(img, 'input_')
    
    # get heatmap on output
    img = tf.reshape(output, (1, 64, 64, 1))
    x_min = tf.reduce_min(img)
    x_max = tf.reduce_max(img)
    img = (img - x_min) / (x_max - x_min)
    img = tf.image.grayscale_to_rgb(img, name=None)
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    image_summaries(img, 'output_heatmap')

    # get heatmap on input
    print (input.get_shape().as_list())
    img = tf.reshape(input, (1, 64, 64, 1))
    x_min = tf.reduce_min(img)
    x_max = tf.reduce_max(img)
    img = (img - x_min) / (x_max - x_min)
    img = tf.image.grayscale_to_rgb(img, name=None)
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    image_summaries(img, 'input_heatmap')
