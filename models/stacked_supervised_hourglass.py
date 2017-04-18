import tensorflow as tf


class stacked_hourglass():
    def __init__(self, steps, name='stacked_hourglass'):
        self.nb_stack = len(steps)
        self.steps = steps
        self.name = name
        self.module_supervisions = [[], [], [], [], []]
    
    def __call__(self, x):
        with tf.name_scope(self.name) as scope:
            padding = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]],
                             name='padding')
            with tf.variable_scope("preprocessing") as sc:
                conv1 = self._conv(padding, 64, 7, 2, 'VALID', 'conv1')
                norm1 = tf.contrib.layers.batch_norm(conv1, 0.9, epsilon=1e-5,
                                                     activation_fn=tf.nn.relu,
                                                     scope=sc)
                r1 = self._residual_block(norm1, 128, 'r1')
                pool = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], 'VALID',
                                                    scope=scope)
                r2 = self._residual_block(pool, 128, 'r2')
                r3_h = self._residual_block(r2, 256, 'r3_h')
                r3 = self._residual_block(r3_h, 256, 'r3')
            hg = [None] * self.nb_stack
            ll = [None] * self.nb_stack
            ll_ = [None] * self.nb_stack
            out = [None] * self.nb_stack
            out_ = [None] * self.nb_stack
            sum_ = [None] * self.nb_stack
            with tf.variable_scope('_hourglass_0_with_supervision') as sc:
                hg[0] = self._hourglass(r3, 4, 64 * 14, '_hourglass', 0)
                ll[0] = self._conv_bn_relu(hg[0], 256, name='conv_1')
                ll_[0] = self._conv(ll[0], 256, 1, 1, 'VALID', 'll')
                out[0] = self._conv(ll[0], 14 * 64, 1, 1, 'VALID', 'out')
                out_[0] = self._conv(out[0], 256, 1, 1, 'VALID', 'out_')
                sum_[0] = tf.add_n([ll_[0], out_[0], r3])
            for i in range(1, self.nb_stack - 1):
                with tf.variable_scope(
                            '_hourglass_' + str(i) + '_with_supervision') as sc:
                    hg[i] = self._hourglass(sum_[i - 1], 4, 64 * 14,
                                            '_hourglass', i)
                    ll[i] = self._conv_bn_relu(hg[i], 256, name='conv_1')
                    ll_[i] = self._conv(ll[i], 256, 1, 1, 'VALID', 'll')
                    out[i] = self._conv(ll[i], 14 * 64, 1, 1, 'VALID', 'out')
                    out_[i] = self._conv(out[i], 256, 1, 1, 'VALID', 'out_')
                    sum_[i] = tf.add_n([ll_[i], out_[i], sum_[i - 1]])
            with tf.variable_scope(
                        '_hourglass_' + str(
                            self.nb_stack - 1) + '_with_supervision') as sc:
                hg[self.nb_stack - 1] = self._hourglass(sum_[self.nb_stack - 2],
                                                        4,
                                                        64 * 14,
                                                        '_hourglass',
                                                        self.nb_stack - 1)
                ll[self.nb_stack - 1] = self._conv_bn_relu(
                    hg[self.nb_stack - 1], 256,
                    name='conv_1')
                self.module_supervisions[-1].append(
                    self._conv(ll[self.nb_stack - 1],
                               14 * self.steps[self.nb_stack - 1],
                               1, 1,
                               'VALID', 'out'))
            
            # return tf.concat(out, axis=3)
            # return self.module_supervisions
            return self.module_supervisions
    
    def _conv(self, inputs, nb_filter, kernel_size=1, strides=1, pad='VALID',
              name='conv'):
        with tf.variable_scope(name) as scope:
            with tf.device('/cpu:0'):
                shape = [kernel_size, kernel_size,
                         inputs.get_shape().as_list()[3],
                         nb_filter]
                kernel = tf.get_variable(name, shape, initializer= \
                    tf.contrib.layers.xavier_initializer(
                        uniform=False))
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1],
                                padding=pad,
                                data_format='NHWC')
            return conv
    
    def _conv_bn_relu(self, inputs, nb_filter, kernel_size=1, strides=1,
                      name='weights'):
        with tf.variable_scope(name) as scope:
            with tf.device('/cpu:0'):
                shape = [kernel_size, kernel_size,
                         inputs.get_shape().as_list()[3],
                         nb_filter]
                kernel = tf.get_variable(name, shape, initializer= \
                    tf.contrib.layers.xavier_initializer(uniform=False))
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1],
                                padding='SAME', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5,
                                                activation_fn=tf.nn.relu,
                                                scope=scope)
            return norm
    
    def _conv_block(self, inputs, nb_filter_out, name='_conv_block'):
        with tf.variable_scope(name) as scope:
            with tf.variable_scope('norm_conv1') as sc:
                norm1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5,
                                                     activation_fn=tf.nn.relu,
                                                     scope=sc,
                                                     fused=True)
                conv1 = self._conv(norm1, nb_filter_out / 2, 1, 1, 'SAME',
                                   name='conv1')
            with tf.variable_scope('norm_conv2') as sc:
                norm2 = tf.contrib.layers.batch_norm(conv1, 0.9, epsilon=1e-5,
                                                     activation_fn=tf.nn.relu,
                                                     scope=sc,
                                                     fused=True)
                conv2 = self._conv(norm2, nb_filter_out / 2, 3, 1, 'SAME',
                                   name='conv2')
            with tf.variable_scope('norm_conv3') as sc:
                norm3 = tf.contrib.layers.batch_norm(conv2, 0.9, epsilon=1e-5,
                                                     activation_fn=tf.nn.relu,
                                                     scope=sc,
                                                     fused=True)
                conv3 = self._conv(norm3, nb_filter_out, 1, 1, 'SAME',
                                   name='conv3')
            return conv3
    
    def _skip_layer(self, inputs, nb_filter_out, name='_skip_layer'):
        if inputs.get_shape()[3].__eq__(tf.Dimension(nb_filter_out)):
            return inputs
        else:
            with tf.name_scope(name) as scope:
                conv = self._conv(inputs, nb_filter_out, 1, 1, 'SAME',
                                  name='conv')
                return conv
    
    def _residual_block(self, inputs, nb_filter_out, name='_residual_block'):
        with tf.variable_scope(name) as scope:
            _conv_block = self._conv_block(inputs, nb_filter_out)
            _skip_layer = self._skip_layer(inputs, nb_filter_out)
            return tf.add(_skip_layer, _conv_block)
    
    def _hourglass(self, inputs, n, nb_filter_res, name='_hourglass', rank_=0):
        with tf.variable_scope(name) as scope:
            
            if n > 1:
                # Upper branch
                up1 = self._residual_block(inputs, (128*64*14) /
                                           nb_filter_res, \
                      'up1')
                # Lower branch
                pool = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2],
                                                    'VALID')
                low1 = self._residual_block(pool, (128*64*2*14) / nb_filter_res,
                                            'low1')
                
                low2 = self._hourglass(low1, n - 1, nb_filter_res / 2, 'low2',
                                       rank_)
                low2 = tf.concat([low2, up1], axis=-1, name = 'merged1')
            else:
                low1 = self._residual_block(inputs, (128*64*14) / nb_filter_res, 'low1_1')
                low2 = self._residual_block(low1, (128*64*14) / nb_filter_res, 'low2_1')
            
            low3 = self._residual_block(low2, (128*64*14) / nb_filter_res, 'low3')
            low3 = self._conv_bn_relu(low3, (128*64*14) / nb_filter_res,
                                      name='low3_1')
            
            lower1 = self._conv(low3, nb_filter_res, 1, 1, 'VALID', 'lower1')
            
            self.module_supervisions[n - 1].append(lower1)
            
            if n < 4:
                lower2 = tf.image.resize_nearest_neighbor(lower1,
                                                          tf.shape(lower1)[
                                                          1:3] * 2,
                                                          name='upsampling_2')
                lower2 = self._conv_bn_relu(lower2, ((128*64*14) / (nb_filter_res*2)),
                                             name = 'lower2')
                low4 = tf.image.resize_nearest_neighbor(low3,
                                                        tf.shape(low3)[1:3] *2,
                                                        name='upsampling_1')
                low4 = self._conv_block(low4, ((128*64*14) / (nb_filter_res*2))
                                            , name = 'low4_1')
            else:
                lower2 = self._conv_bn_relu(lower1, (128*64*14) / (nb_filter_res*2),
                                            name='lower2')
                low4 = self._residual_block(low3, (128*64*14) / (nb_filter_res*2), 'low4')
                low4 = self._conv_block(low4, (128*64*14) / (nb_filter_res*2),
                                          name='low4_1')
            
            if n < 5:
                return tf.add(lower2, low4, name='merge')
