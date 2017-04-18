import tensorflow as tf
import tensorflow.contrib.slim as slim


def generator(zt, batch_size, name='generator'):
    """function for building generator"""
    # First a fully connected layer to upsample to 512x4x4x4 = 32768
    # also get Batch size
    with tf.variable_scope(name) as scope:
        zP = _fc(zt, 4 * 4 * 4 * 256, 'fc1')
        zCon = tf.reshape(zP, [-1, 4, 4, 4, 256])
        zConv2 = _deconv3_bn_relu(zCon, 128, [batch_size, 8, 8, 8, 128],
                                  name='deconv2')
        zConv3 = _deconv3_bn_relu(zConv2, 64, [batch_size, 16, 16, 16, 64],
                                  name='deconv3')
        zConv4 = _deconv3_bn_relu(zConv3, 32, [batch_size, 32, 32, 32, 32],
                                  name='deconv4')
        zConv5 = _deconv3_sigmoid(zConv4, 14, [batch_size, 64, 64, 64, 14],
                                  name='deconv5')
    
    return zConv5


def discriminator(y, name='discriminator'):
    """Function for building discriminator
    Mirrors the generator but with Leaky Relu"""
    with tf.variable_scope(name) as scope:
        zConv1 = _conv3_bn_lrelu(y, 32, name='deconv1')
        zConv2 = _conv3_bn_lrelu(zConv1, 64, name='deconv2')
        zConv3 = _conv3_bn_lrelu(zConv2, 128, name='deconv3')
        zConv4 = _conv3_bn_lrelu(zConv3, 256, name='deconv4')
        
        conv4_flat = slim.flatten(zConv4)
        print (conv4_flat.get_shape().as_list())
    return _fc_sigmoid(conv4_flat, 1, 'fc_final')


def encoder(x, name='encoder'):
    """function for building encoder"""
    out_dict = {}
    with tf.variable_scope(name) as scope:
        xConv1 = _conv2_bn_relu(x, 32, kernel_size=11, strides=4, name='conv1')

        xConv2 = _conv2_bn_relu(xConv1, 64, kernel_size=5, strides=2,
                                name='conv2')

        xConv3 = _conv2_bn_relu(xConv2, 128, kernel_size=5, strides=2,
                                name='conv3')

        xConv4 = _conv2_bn_relu(xConv3, 256, kernel_size=5, strides=2,
                                name='conv4')

        xConv5 = _conv2(xConv4, 8, kernel_size=8, strides=1,
                                name='conv5')

        out = slim.flatten(xConv5)
    out_dict['mu'] = out[:, :256]
    out_dict['log_sigma'] = out[:, 256:]
    
    return out_dict

def gaussianKLD(mu1, lv1, mu2, lv2):
    """ Kullback-Leibler divergence of two Gaussians
        *Assuming that each dimension is independent
        mu: mean
        lv: log variance
        Equation: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """
    with tf.name_scope('GaussianKLD'):
        v1 = tf.exp(lv1)
        v2 = tf.exp(lv2)
        mu_diff_sq = tf.square(mu1 - mu2)
        dimwise_kld = .5 * (
            (lv2 - lv1) + tf.div(v1, v2) + tf.div(mu_diff_sq, v2) - 1.)
    
    return tf.reduce_sum(dimwise_kld, -1)


def gaussianSampler(z_mu, z_lv, name='GaussianSampleLayer'):
    """This function, given mu and sigma of a gaussian distribution
    generates a sample"""
    with tf.name_scope(name):
        eps = tf.random_normal(tf.shape(z_mu))
        std = tf.sqrt(tf.exp(z_lv))
    
    return tf.add(z_mu, tf.multiply(eps, std))


def lrelu(x, leak=0.2, name="lrelu"):
    """ Leaky ReLU """
    return tf.maximum(x, leak * x)


def _fc_sigmoid(input, nb_filter, name='fc_sigmoid'):
    with tf.variable_scope(name) as scope:
        shape = input.get_shape().as_list()[-1]
        with tf.device('/cpu:0'):
            var = tf.get_variable('Matrix1', [shape, nb_filter],
                                  tf.float32,
                                  tf.contrib.layers.xavier_initializer(
                                      uniform=False))
            bias = tf.get_variable('bias1', [nb_filter],
                                   initializer=tf.constant_initializer(0.0))
        print(shape)

        out = tf.nn.bias_add(tf.matmul(input, var), bias)
    
    return tf.sigmoid(out)


def _fc(input, nb_filter, name='fc'):
    with tf.variable_scope(name) as scope:
        shape = input.get_shape().as_list()[-1]
        with tf.device('/cpu:0'):
            var = tf.get_variable('Matrix1', [shape, nb_filter],
                                  tf.float32,
                                  tf.contrib.layers.xavier_initializer(
                                      uniform=False))
            bias = tf.get_variable('bias1', [nb_filter],
                                   initializer=tf.constant_initializer(0.0))
        print(shape)

        out = tf.nn.bias_add(tf.matmul(input, var), bias)
        
        out = tf.contrib.layers.batch_norm(out, 0.9, epsilon=1e-5,
                                           activation_fn=tf.nn.relu,
                                           scope=scope)
    return out


def _deconv3_bn_relu(inputs, nb_filter, out_shape, kernel_size=4, strides=2,
                     name='deconv3'):
    """ Implementation of 3D conv net layer """
    with tf.variable_scope(name) as scope:
        with tf.device('/cpu:0'):
            shape = [kernel_size, kernel_size, kernel_size,
                     nb_filter ,inputs.get_shape().as_list()[-1]]
            kernel = tf.get_variable(name, shape, initializer= \
                tf.contrib.layers.xavier_initializer(uniform=False))
            
        deconv = tf.nn.conv3d_transpose(inputs, kernel, out_shape,
                                        [1, strides, strides, strides, 1],
                                        padding='SAME')
        print(shape)

        norm = tf.contrib.layers.batch_norm(deconv, 0.9, epsilon=1e-5,
                                            activation_fn=tf.nn.relu,
                                            scope=scope)
        return norm


def _deconv3_sigmoid(inputs, nb_filter, out_shape, kernel_size=4, strides=2,
                     name='deconv3'):
    """ Implementation of 3D conv net layer """
    with tf.variable_scope(name) as scope:
        with tf.device('/cpu:0'):
            shape = [kernel_size, kernel_size, kernel_size,
                     nb_filter, inputs.get_shape().as_list()[-1]]
            kernel = tf.get_variable(name, shape, initializer= \
                tf.contrib.layers.xavier_initializer(uniform=False))
        print(shape)

        deconv = tf.nn.conv3d_transpose(inputs, kernel, out_shape,
                                        [1, strides, strides, strides, 1],
                                        padding='SAME')
        
        norm = tf.sigmoid(deconv)
        
        return norm


def _conv3_bn_lrelu(inputs, nb_filter, kernel_size=4, strides=2,
                    name='conv3'):
    """ Implementation of 3D conv net layer """
    with tf.variable_scope(name) as scope:
        with tf.device('/cpu:0'):
            shape = [kernel_size, kernel_size, kernel_size,
                     inputs.get_shape().as_list()[-1],
                     nb_filter]
            kernel = tf.get_variable(name, shape, initializer= \
                tf.contrib.layers.xavier_initializer(uniform=False))
        print(shape)
        deconv = tf.nn.conv3d(inputs, kernel,
                              [1, strides, strides, strides, 1],
                              padding='SAME')
        
        norm = tf.contrib.layers.batch_norm(deconv, 0.9, epsilon=1e-5,
                                            activation_fn=None,
                                            scope=scope)
        
        return lrelu(norm)


def _conv2_bn_relu(inputs, nb_filter, kernel_size=4, strides=2,
                   name='conv3'):
    """ Implementation of 3D conv net layer """
    with tf.variable_scope(name) as scope:
        with tf.device('/cpu:0'):
            shape = [kernel_size, kernel_size,
                     inputs.get_shape().as_list()[-1],
                     nb_filter]
            kernel = tf.get_variable(name, shape, initializer= \
                tf.contrib.layers.xavier_initializer(uniform=False))
        print(shape)

        deconv = tf.nn.conv2d(inputs, kernel,
                              [1, strides, strides, 1],
                              padding='SAME', data_format='NHWC')
        
        norm = tf.contrib.layers.batch_norm(deconv, 0.9, epsilon=1e-5,
                                            activation_fn=tf.nn.relu,
                                            scope=scope)
        
        return norm


def _conv2(inputs, nb_filter, kernel_size=4, strides=2,
           name='conv3'):
    """ Implementation of 3D conv net layer """
    with tf.variable_scope(name) as scope:
        with tf.device('/cpu:0'):
            shape = [kernel_size, kernel_size,
                     inputs.get_shape().as_list()[-1],
                     nb_filter]
            kernel = tf.get_variable(name, shape, initializer= \
                tf.contrib.layers.xavier_initializer(uniform=False))
        print(shape)

        conv = tf.nn.conv2d(inputs, kernel,
                            [1, strides, strides, 1],
                            padding='SAME', data_format='NHWC')
    
    return conv
