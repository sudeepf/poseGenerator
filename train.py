from __future__ import print_function

# import torch
import tensorflow as tf
from os import listdir
import os
from os.path import isfile, join
import numpy as np
from scipy import misc

import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# Get all the custom helper util headers
import utils.data_prep
import utils.add_summary
import utils.train_utils
import utils.eval_utils
import utils.get_flags
import models.hg_graph_builder

# Read up and set up all the flag variables
FLAG = utils.get_flags.get_flags()


def main(_):
    if not FLAG.dataset_dir:
        raise ValueError(
            'You must supply the dataset directory with --dataset_dir')
    
    DataHolder = utils.train_utils.DataHolder(FLAG)
    
    print('data loaded... phhhh')
    
    with tf.Graph().as_default():
        
        builder = models.hg_graph_builder.HGgraphBuilder_MultiGPU(FLAG)
        
        # builder = models.hg_graph_builder.HGgraphBuilder(FLAG)
        print("build finished, There it stands, tall and strong...")
        
        graph_def = tf.get_default_graph().as_graph_def()
        graphpb_txt = str(graph_def)
        with open('graphpb.txt', 'w') as f:
            f.write(graphpb_txt)
        
        # lol = lol2
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        saver = tf.train.Saver()
        
        with tf.Session(config=config) as sess:
            # with tf.Session() as sess:
            merged = tf.summary.merge_all()
            
            # All the variable initialiezed in MoFoking RunTime
            # Confusing the world gets when yoda asks initializer operator before
            
            
            train_writer, test_writer = utils.add_summary.get_summary_writer(
                FLAG, sess)
            print(FLAG.load_ckpt_path)
            ckpt = tf.train.get_checkpoint_state(FLAG.load_ckpt_path)
            if ckpt and ckpt.model_checkpoint_path:
                if os.path.isabs(ckpt.model_checkpoint_path):
                    # Restores from checkpoint with absolute path.
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    # Restores from checkpoint with relative path.
                    saver.restore(sess, os.path.join(FLAG.load_ckpt_path,
                                                     ckpt.model_checkpoint_path))
                
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/imagenet_train/model.ckpt-0,
                # extract global_step from it.
                global_step = \
                ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Succesfully loaded model from %s at step=%s.' %
                      (ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found')
                # tf.global_variables_initializer().run()
                saver.restore(sess, "./tensor_record//tmp/model64-64.ckpt")
                print('model Initialized...')
            
            print("Let the Training Begin...")
            
            # Train the model, and also write summaries.
            # Every 10th step, measure test-set accuracy, and write test summaries
            # All other steps, run train_step on training data, & add training summaries
            structure = map(int, FLAG.structure_string.split('-'))
            
            for step in range(DataHolder.train_data_size):
                
                if step % 1000 == 99:  # Record execution stats
                    save_path = saver.save(sess,
                                           FLAG.eval_dir + "/tmp/model" +
                                           FLAG.structure_string + ".ckpt")
                    print('Adding Model data for ', step, 'at ', save_path)
                
                if step % 10000 == 9999:  # Record execution stats
                    save_path = saver.save(sess, FLAG.checkpoint_dir +
                                           '/model_%05d_' %
                                           step + FLAG.structure_string + '.ckpt')
                    print('Adding Model data for ', step, 'at ', save_path)
                
                _x = []
                vec_64 = []
                vec_32 = []
                vec_16 = []
                vec_8 = []
                gt = []
                time_ = time.clock()
                for i in map(int, FLAG.gpu_string.split('-')):
                    fd = DataHolder.get_next_train_batch()
                    _x.append(fd[0])
                    vec_64.append(fd[1])
                    vec_32.append(fd[2])
                    vec_16.append(fd[3])
                    vec_8.append(fd[4])
                    #gt.append(fd[5])
                
                feed_dict_x = {i: d for i, d in zip(builder._x, _x)}
                feed_dict_vec_64 = {i: d for i, d in
                                   zip(builder.tensor_64, vec_64)}
                feed_dict_vec_32 = {i: d for i, d in
                                   zip(builder.tensor_32, vec_32)}
                feed_dict_vec_16 = {i: d for i, d in
                                   zip(builder.tensor_16, vec_16)}
                feed_dict_vec_8 = {i: d for i, d in
                                   zip(builder.tensor_8, vec_8)}
                feed_dict_gt = {i: d for i, d in zip(builder.gt, gt)}
                feed_dict_x.update(feed_dict_vec_64)
                feed_dict_x.update(feed_dict_vec_32)
                feed_dict_x.update(feed_dict_vec_16)
                feed_dict_x.update(feed_dict_vec_8)
                #feed_dict_x.update(feed_dict_gt)
                
                # print ("PreProcessing Time - incd reading", time.clock()-time_)
                time_ = time.clock()
                
                if step % 10 == 1:
                    summary, loss_, _ = sess.run([merged,
                                                            builder.loss,
                                                  builder.train_rmsprop],
                                                 feed_dict_x)
                    
                    train_writer.add_summary(summary, step)
                    
                    continue
                
                run_metadata = tf.RunMetadata()
                loss_, _, data_out = sess.run([builder.loss,
                                            builder.train_rmsprop,builder.label],
                                            feed_dict_x)
                
                # print("Time to feed and run the network", time.clock()-time_)
                print("Grinding... Loss = " + str(loss_))
                

            train_writer.close()
            test_writer.close()


if __name__ == '__main__':
    tf.app.run()
