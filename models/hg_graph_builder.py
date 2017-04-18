import models.stacked_supervised_hourglass as hg
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
            self.optimizer = tf.train.RMSPropOptimizer(FLAG.learning_rate)
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
            
            tower_grads = []
            
            # Define the array of the place holders
            self._x = []
            self.y = []
            self.gt = []
            self.loss = 0
            self.output = []
            self.label = []
            self.tensor_8 = []
            self.tensor_16 = []
            self.tensor_32 = []
            self.tensor_64 = []
            self.tensor_8_2d = []
            self.tensor_16_2d = []
            self.tensor_32_2d = []
            self.tensor_64_2d = []
            self.const_1 = tf.constant(FLAG.const_1)
            self.const_2 = tf.constant(FLAG.const_2)
            self.loss_KL = []
            self.loss_recon = []
            self.loss_gan = []

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
                            
                            FLAG.train_2d = False
                            if i % 2 == 0:
                                FLAG.train_2d = True
                            
                            if FLAG.train_2d == True:
                                
                                print ('2D is ON')
                                tensor_8 = tf.placeholder(tf.float32,
                                                          [FLAG.batch_size,
                                                           2, FLAG.num_joints,
                                                           8])
                                tensor_16 = tf.placeholder(tf.float32,
                                                           [FLAG.batch_size, 2,
                                                            FLAG.num_joints,
                                                            16])
                                tensor_32 = tf.placeholder(tf.float32,
                                                           [FLAG.batch_size,
                                                            2, FLAG.num_joints,
                                                            32])
                                tensor_64 = tf.placeholder(tf.float32,
                                                           [FLAG.batch_size,
                                                            2, FLAG.num_joints,
                                                            64])

                                label.append(tf.scalar_mul(mult_fac,
                                    utils.data_prep.heatmap_vec_gpu(
                                        tensor_64[:, 0],
                                        tensor_64[:, 1], 1, FLAG)))

                                label.append(tf.scalar_mul(mult_fac,
                                    utils.data_prep.heatmap_vec_gpu(
                                        tensor_32[:, 0],
                                        tensor_32[:, 1], 2, FLAG)))

                                label.append(tf.scalar_mul(mult_fac,
                                    utils.data_prep.heatmap_vec_gpu(
                                        tensor_16[:, 0],
                                        tensor_16[:, 1], 4, FLAG)))

                                label.append(tf.scalar_mul(mult_fac,
                                    utils.data_prep.heatmap_vec_gpu(
                                        tensor_8[:, 0],
                                        tensor_8[:, 1], 8, FLAG)))
                            
                            else:
                                
                                tensor_8 = tf.placeholder(tf.float32,
                                                          [FLAG.batch_size,
                                                           3, FLAG.num_joints,
                                                           8])
                                tensor_16 = tf.placeholder(tf.float32,
                                                           [FLAG.batch_size, 3,
                                                            FLAG.num_joints,
                                                            16])
                                tensor_32 = tf.placeholder(tf.float32,
                                                           [FLAG.batch_size,
                                                            3, FLAG.num_joints,
                                                            32])
                                tensor_64 = tf.placeholder(tf.float32,
                                                           [FLAG.batch_size,
                                                            3, FLAG.num_joints,
                                                            64])
                            
                                label.append(tf.scalar_mul(mult_fac,
                                    utils.data_prep.volumize_vec_gpu(
                                         tensor_64[:,0],
                                         tensor_64[:,1],
                                         tensor_64[:,
                                         2], 1, FLAG)))
                            
                                label.append(tf.scalar_mul(mult_fac,
                                    utils.data_prep.volumize_vec_gpu(
                                        tensor_32[:, 0],
                                        tensor_32[:, 1],
                                        tensor_32[:, 2], 2, FLAG)))
    
                                label.append(tf.scalar_mul(mult_fac,
                                    utils.data_prep.volumize_vec_gpu(
                                        tensor_16[:, 0],
                                        tensor_16[:, 1],
                                        tensor_16[:,
                                        2], 4, FLAG)))
    
                                label.append(tf.scalar_mul(mult_fac,
                                    utils.data_prep.volumize_vec_gpu(
                                        tensor_8[:, 0],
                                        tensor_8[:, 1],
                                        tensor_8[:,
                                        2], 8, FLAG)))
                            
                            # label = utils.data_prep.prepare_output_gpu(y, steps, FLAG)
                            self.label.append(label)
                            
                            # If I ever write a handle for accuracy computation in TF
                            gt = tf.placeholder(tf.float32, [None, FLAG.num_joints,
                                                             3])
                            
                            loss, output = self.tower_loss(_x, label, gt, steps,
                                                           scope, FLAG, 'GPU_%d' % (i))
                            
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
                            self.output.append(output)
                            # Add Device spec summaries
                            
                            
                            # Calculate the gradients for the batch of data on this CIFAR tower.
                            grads = self.optimizer.compute_gradients(loss)
                            
                            # Keep track of the gradients across all towers.
                            tower_grads.append(grads)
                            
                            # Append all the place holder to CPU mem
                            self._x.append(_x)
                            self.y.append(label[0])
                            self.gt.append(gt)
                            self.tensor_64.append(tensor_64)
                            self.tensor_32.append(tensor_32)
                            self.tensor_16.append(tensor_16)
                            self.tensor_8.append(tensor_8)
                            self.loss += loss

                    if FLAG.train_2d == True:
                        with tf.variable_scope('summaries_2D'):
                            label_sum = tf.reduce_sum(label[0][0],
                                                      axis=-1)
                            utils.add_summary.add_all(_x[0:1], label_sum,
                                                      tf.reduce_sum(output[
                                                                    -1:][0][
                                                                        0][0],
                                                                    axis=-1),
                                                      loss)

                            
                    else:
                        with tf.variable_scope('summaries_3D'):
                            label_sum = tf.reduce_sum(
                                tf.reduce_sum(label[0][0],
                                              axis=-1), axis=-1)
                            utils.add_summary.add_all(_x[0:1], label_sum,
                                                      tf.reduce_sum(output[
                                                                    -1:][0][
                                                                        0][0],
                                                                    axis=-1),
                                                      loss)

                            utils.add_summary.add_all_joints(
                                utils.eval_utils.get_precision_MultiGPU(
                                    output[-1:][0][0],
                                    label[0], [64], FLAG), [64], FLAG)
                    
                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            
            self.loss /= len(map(int, FLAG.gpu_string.split('-')))
            tf.summary.scalar('Overall_loss', self.loss)

            
            
            
    
            
            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = self.average_gradients(tower_grads)
            
            # Add histograms for gradients.
            # for grad, var in grads:
            #	if grad is not None:
            #		summaries.append(
            #			tf.summary.histogram(var.op.name + '/gradients', grad))
            
            # Apply the gradients to adjust the shared variables.
            self.train_rmsprop = self.optimizer.apply_gradients(grads,
                                                                global_step=global_step)
        
        # Add histograms for trainable variables.
        # for var in tf.trainable_variables():
        #	summaries.append(tf.summary.histogram(var.op.name, var))
    
    def tower_loss(self, _x, y, gt, steps, scope, FLAG, name):
        """Calculate the total loss on a single tower running the CIFAR model.
        Args:
            scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        Returns:
             Tensor of shape [] containing the total loss for a batch of data
        """
        # Get images and labels for CIFAR-10.
        # Build inference Graph.
        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        
        
        
        # with tf.variable_scope(scope):
        
        output = hg.stacked_hourglass(steps, 'stacked_hourglass')(_x)
        
        # Defining Loss with root mean square error
        # loss = tf.reduce_mean(tf.square(output - y))
        for i in xrange(len(y)):
            for j in xrange(len(steps)):
                
                shape_ = output[i][j].get_shape().as_list()
                out = tf.reshape(output[i][j], [FLAG.batch_size] + shape_[1:-1]
                                 +[shape_[-1] / FLAG.num_joints,
                                  FLAG.num_joints])
                if FLAG.train_2d == True:
                    out = tf.reduce_sum(out, axis=-2)
                    
                    yy = tf.reshape(y[3-i], out.get_shape().as_list())
    
                    loss = tf.reduce_mean(tf.square(out - yy))
                    # Calculate the total loss for the current tower.
                    tf.add_to_collection('losses', loss)
                else:
                    yy = tf.reshape(y[3 - i], out.get_shape().as_list())
    
                    loss = tf.reduce_mean(tf.square(tf.reduce_sum(out - yy,
                                                                  axis=-2)))
                    # Calculate the total loss for the current tower.
                    tf.add_to_collection('losses', loss)
        
        # Add final loss
        shape_ = output[-1][0].get_shape().as_list()
        out = tf.reshape(output[-1][0], [FLAG.batch_size] + shape_[1:-1]
                         + [shape_[-1] / FLAG.num_joints,
                            FLAG.num_joints])


        if FLAG.train_2d == True:
            out = tf.reduce_sum(out, axis=-2)

        yy = tf.reshape(y[0], out.get_shape().as_list())

        loss = tf.reduce_mean(tf.square(out - yy))
        # Calculate the total loss for the current tower.
        tf.add_to_collection('losses', loss)
        
        
        # Calculate the total loss for the current tower.
        # tf.add_to_collection('losses', loss)
        # Calculate the total loss for the current tower.
        total_loss = tf.add_n(tf.get_collection('losses', scope), name='total_loss')
        
        return total_loss, output
    
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
