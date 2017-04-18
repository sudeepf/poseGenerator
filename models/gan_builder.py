import models.vae_gan_3d as gan
import tensorflow as tf
import numpy as np
import utils.add_summary
import utils.eval_utils
import utils.data_prep


class HGgraphBuilder():
    def __init__(self, FLAG):
        with tf.variable_scope('model_graph'):
            # first Build Model and define all the place holders
            # This can go to another file under name model
            # These parameters can be read from a external config file
            print ("start build model...")
            
            # from string model struct to list model struct
            steps = map(int, FLAG.structure_string.split('-'))
            total_dim = np.sum(np.array(steps))
            
            self._x = tf.placeholder(tf.float32,
                                     [None, FLAG.image_res, FLAG.image_res,
                                      FLAG.image_c])
            self.y = tf.placeholder(tf.float32, [None, FLAG.volume_res,
                                                 FLAG.volume_res,
                                                 FLAG.num_joints *
                                                 total_dim])
            
            # If I ever write a handle for accuracy computation in TF
            self.gt = tf.placeholder(tf.float32, [None, FLAG.num_joints,
                                                  3])
            
            self.output = hg.stacked_hourglass(steps, 'stacked_hourglass')(
                self._x)
            
            # Defining Loss with root mean square error
            self.loss = tf.reduce_mean(tf.square(self.output - self.y))
            
            self.optimizer = tf.train.RMSPropOptimizer(FLAG.learning_rate)
            
            self.train_step = tf.Variable(0, name='global_step',
                                          trainable=False)
            
            self.train_rmsprop = self.optimizer.minimize(self.loss,
                                                         self.train_step)
            
            utils.add_summary.add_all(self._x, self.y, self.output, self.loss)


class HGgraphBuilder_MultiGPU():
    def __init__(self, FLAG):
        # This Class defines an object to train model with multi gpu's
        print ("Start building Multi GPU model")
        
        with tf.device('/cpu:0'):
            self.optimizer_generator = tf.train.RMSPropOptimizer(
                FLAG.learning_rate)
            self.optimizer_discriminator = tf.train.RMSPropOptimizer(
                FLAG.learning_rate)
            self.optimizer_encoder = tf.train.RMSPropOptimizer(
                FLAG.learning_rate)
            
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
            
            tower_grads_generator = []
            tower_grads_discriminator = []
            tower_grads_encoder = []

            # Define the array of the place holders
            self._x = []
            self.y = []
            self.zt = []
            self.loss = 0
            self.output = []
            self.label = []
            self.tensor_64 = []
            self.tensor_64_2d = []
            self.const_1 = tf.constant(FLAG.const_1)
            self.const_2 = tf.constant(FLAG.const_2)
            self.loss_discriminator = []
            self.loss_generator = []
            self.loss_encoder = []
            
            steps = map(int, FLAG.structure_string.split('-'))
            total_dim = np.sum(np.array(steps))
            
            with tf.variable_scope('model_graph'):
                for i, ii in enumerate(map(int, FLAG.gpu_string.split('-'))):
                    with tf.device('/gpu:%d' % ii):
                        label = []
                        with tf.name_scope('GPU_%d' % (i)) as scope:
                            # Calculate the loss for one tower of the CIFAR model. This function
                            # constructs the entire CIFAR model but shares the variables across
                            # all towers.
                            
                            mult_fac = tf.constant(FLAG.joint_prob_max)
                            
                            _x = tf.placeholder(tf.float32,
                                                [None, FLAG.image_res,
                                                 FLAG.image_res,
                                                 FLAG.image_c])
                            
                            
                            if FLAG.train_2d == True:
                                
                                print ('2D is ON')
                                tensor_64 = tf.placeholder(tf.float32,
                                                           [FLAG.batch_size,
                                                            2, FLAG.num_joints,
                                                            64])
                                
                                label.append(tf.scalar_mul(mult_fac,
                                                           utils.data_prep.heatmap_vec_gpu(
                                                               tensor_64[:, 0],
                                                               tensor_64[:, 1],
                                                               1, FLAG)))
                                
                                
                            else:
                                
                                tensor_64 = tf.placeholder(tf.float32,
                                                           [FLAG.batch_size,
                                                            3, FLAG.num_joints,
                                                            64])
                                
                                label.append(tf.scalar_mul(mult_fac,
                                                           utils.data_prep.volumize_vec_gpu(
                                                               tensor_64[:, 0],
                                                               tensor_64[:, 1],
                                                               tensor_64[:,
                                                               2], 1, FLAG)))
                                
                                
                            # label = utils.data_prep.prepare_output_gpu(y, steps, FLAG)
                            self.label.append(label)
                            
                            # If I ever write a handle for accuracy computation in TF
                            zt = tf.placeholder(tf.float32,
                                                [None, 256])
                            
                            output = self.tower_loss(_x, label[0], zt, FLAG,
                                                 'GPU_%d' % (i))
                            
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
                            self.output.append(output)
                            # Add Device spec summaries

                            self.t_vars = tf.trainable_variables()

                            e_vars = [var for var in self.t_vars if
                                           ('encoder') in var.name]
                            g_vars = [var for var in self.t_vars if
                                           ('generator') in var.name]
                            d_vars = [var for var in self.t_vars if
                                           ('discriminator') in var.name]
                            
                            print(len(e_vars))
                            print(".......")
                            print(len(g_vars))
                            print(".......")
                            print(len(d_vars))
                            # Calculate the gradients for the batch of data on this CIFAR tower.
                            grads_discriminator = \
                                self.optimizer_discriminator.compute_gradients(
                                    self.loss_discriminator[i], d_vars)

                            grads_generator = \
                                self.optimizer_discriminator.compute_gradients(
                                    self.loss_generator[i], g_vars)

                            grads_encoder = \
                                self.optimizer_discriminator.compute_gradients(
                                    self.loss_encoder[i], e_vars)
                            
                            # Keep track of the gradients across all towers.
                            tower_grads_discriminator.append(grads_discriminator)
                            tower_grads_generator.append(grads_generator)
                            tower_grads_encoder.append(grads_encoder)
                            
                            # Append all the place holder to CPU mem
                            self._x.append(_x)
                            self.y.append(label[0])
                            self.zt.append(zt)
                            self.tensor_64.append(tensor_64)
                            
                    
                        
                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            
            
            loss = tf.reduce_mean(tf.stack(self.loss_generator))
            loss /= len(map(int, FLAG.gpu_string.split('-')))
            tf.summary.scalar('generator_loss', loss)

            loss = tf.reduce_mean(tf.stack(self.loss_encoder))
            loss /= len(map(int, FLAG.gpu_string.split('-')))
            tf.summary.scalar('encoder_loss', loss)

            loss = tf.reduce_mean(tf.stack(self.loss_discriminator))
            loss /= len(map(int, FLAG.gpu_string.split('-')))
            tf.summary.scalar('discriminator_loss', loss)
            
            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads_g = self.average_gradients(tower_grads_generator)
            grads_d = self.average_gradients(tower_grads_discriminator)
            grads_e = self.average_gradients(tower_grads_encoder)
            
            
            # Apply the gradients to adjust the shared variables.
            self.train_rmsprop_g = self.optimizer_generator.apply_gradients(
                grads_g, global_step=global_step)
            self.train_rmsprop_e = self.optimizer_encoder.apply_gradients(
                grads_e, global_step=global_step)
            self.train_rmsprop_d = self.optimizer_discriminator.apply_gradients(
                grads_d, global_step=global_step)
            
            # Add histograms for trainable variables.
            # for var in tf.trainable_variables():
            #	summaries.append(tf.summary.histogram(var.op.name, var))
    
    def train_GAN(self, feed_x, feed_vec, feed_zt, sess):
        """This Function basically takes feed data and train over that"""
        #Creat dicts
        print (np.shape(feed_zt[0]))
        feed_dict_x = {i: d for i, d in zip(self._x, feed_x)}
        feed_dict_vec_64 = {i: d for i, d in
                            zip(self.tensor_64, feed_vec)}
        feed_dict_zt = {i: d for i, d in zip(self.zt, feed_zt)}
        
        feed_dict_zt.update(feed_dict_vec_64)
        #First train Discriminator
        loss_d, _ = sess.run([self.loss_discriminator, self.train_rmsprop_d],
                             feed_dict_zt)
        #train Encoder
        feed_dict_x.update(feed_dict_zt)
        loss_e, _ = sess.run([self.loss_encoder, self.train_rmsprop_e],
                             feed_dict_x)
        #train Generator
        loss_g, _ = sess.run([self.loss_generator, self.train_rmsprop_g],
                             feed_dict_x)
        
        loss_d = np.mean(np.array(loss_d))
        loss_e = np.mean(np.array(loss_e))
        loss_g = np.mean(np.array(loss_g))
        
        print (loss_d, loss_e, loss_g)
        
        return loss_d, loss_e, loss_g
    
    def tower_loss(self, _x, y, zt, FLAG, name):
        """Calculate the total loss on a single tower running the CIFAR model.
        Args:
            scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        Returns:
             Tensor of shape [] containing the total loss for a batch of data
        """
        # Bit tricky as we need three step learning and all should happen
        # one after the another, while first step doesnot require label
        # memory why would i send all of them then
        
        # Get Encoder
        Ex = gan.encoder(_x)
        # Build Generator
        Gz = gan.generator(zt, FLAG.batch_size)
        # Build Detector
        Dx = gan.discriminator(y)
        # Get for unreal | reuse the weights
        tf.get_variable_scope().reuse_variables()
        Dg = gan.discriminator(Gz)
        
        tf.get_variable_scope().reuse_variables()
        # reconstruct from encoded input
        ze = gan.gaussianSampler(Ex['mu'], Ex['log_sigma']) # Sampled Z
        Ge = gan.generator(ze, FLAG.batch_size)
        
        
        # Self note that D is a probability of something being fake not other
        #  way round
        
        # These functions together define the optimization objective of the GAN.
        d_loss = -tf.reduce_mean(tf.log(Dx + 1e-7) + tf.log(1. - Dg + 1e-7))
        # This
        # optimizes the discriminator.

        shape_ = Ge.get_shape().as_list()
        yy = tf.reshape(y, [FLAG.batch_size] + [64,64,64,14])
        
        g_loss = -tf.reduce_mean(tf.log(1. - Dg + 1e-7)) + \
                 tf.reduce_mean(tf.square(Ge - yy))
        
        # This optimizes the generator.
        #  Calculate the total loss for the current tower.
        e_loss = 5*gan.gaussianKLD(Ex['mu'], Ex['log_sigma'],
                                 tf.zeros_like(Ex['mu']),
                                 tf.zeros_like(Ex['log_sigma'])) + \
                 tf.reduce_mean(tf.square(Ge - yy))
        
        
        self.loss_generator.append(g_loss)
        self.loss_discriminator.append(d_loss)
        self.loss_encoder.append(e_loss)
        
        # Get trainanble weights and get part them accorgint to network
       
        return Ge
    
    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
                is over individual gradients. The inner list is over the gradient
                calculation for each tower.
        Returns:
             List of pairs of (gradient, variable) where the gradient has been averaged
             across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        
        return average_grads
