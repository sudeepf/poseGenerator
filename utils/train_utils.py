from __future__ import print_function

# import torch
import tensorflow as tf
from os import listdir
import os
from os.path import isfile, join
from scipy import misc

import sys
import numpy as np
import time
import matplotlib.pyplot as plt

# Get all the custom helper util headers
import utils.data_prep
import utils.add_summary
import data_prep
import utils.get_flags


class DataHolder():
    def __init__(self, FLAG):
        self.FLAG = FLAG
        # get all the subjects
        if FLAG.train_2d == False:
            train_subjects = FLAG.data_split_string_train.split('-')
            test_subjects = FLAG.data_split_string_test.split('-')
        else:
            train_subjects = FLAG.data_split_string_train_2d.split('-')
            test_subjects = FLAG.data_split_string_test.split('-')
        
        # Train Folder and Test Folder
        print('Train Dataset: -> ', train_subjects)
        print('Test Dataset: -> ', test_subjects)
        
        # list all the matfiles
        # make one for test and one for train
        self.mFiles_train = []
        for ind, folder in enumerate(train_subjects):
            mat_folder_path = join(join(FLAG.dataset_dir, folder), 'mats')
            mFiles_ = [join(join(join(FLAG.dataset_dir, folder), 'mats'), f) for
                       f in
                       listdir(mat_folder_path) if f.split('.')[-1] == 'mat']
            self.mFiles_train += mFiles_
        print('Total Train Actions x Subjects x Acts -> ',
              len(self.mFiles_train))
        
        self.mFiles_test = []
        for ind, folder in enumerate(test_subjects):
            mat_folder_path = join(join(FLAG.dataset_dir, folder), 'mats')
            mFiles_ = [join(join(join(FLAG.dataset_dir, folder), 'mats'), f) for
                       f in
                       listdir(mat_folder_path) if f.split('.')[-1] == 'mat']
            self.mFiles_test += mFiles_
        print('Total Test Actions x Subjects x Acts -> ', len(self.mFiles_test))
        
        self.read_mat_files()
        
        self.train_data_size = np.shape(self.imgFiles)[0]
        self.test_data_size = np.shape(self.imgFiles_test)[0]
        
        print('Total Training Data Frames are ', self.train_data_size,
              self.FLAG.train_2d)
        print('Total Testing Data Frames are ', self.test_data_size,
              self.FLAG.train_2d)
        
        # initializing training and testing iterations
        self.train_iter = 0
        self.test_iter = 0
        
        # Getting Suffled Mask
        self.mask_train = np.random.permutation(self.train_data_size)
        self.mask_test = np.random.permutation(self.test_data_size)
    
    def read_mat_files(self):
        self.imgFiles, self.pose2, self.pose3 = \
            data_prep.get_list_all_training_frames(self.mFiles_train,
                                                   self.FLAG.train_2d)
        
        self.imgFiles_test, self.pose2_test, self.pose3_test = \
            data_prep.get_list_all_training_frames(self.mFiles_test,
                                                   self.FLAG.train_2d)
    
    def get_dict(self, train, imgFiles, pose2, pose3 = None):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        steps = map(int, self.FLAG.structure_string.split('-'))
        total_dim = np.sum(np.array(steps))
        
        
        if self.FLAG.train_2d == True:
            
            image_b, pose2_b, _ = utils.data_prep.get_batch(imgFiles,
                                                              pose2)
            
            image_b, pose2_b, _ = utils.data_prep.crop_data_top_down(
                image_b,
                pose2_b, None, self.FLAG)
           
            image, pose2, vec_64, vec_32, vec_16, vec_8 = \
                utils.data_prep.get_vector_gt_2d(image_b, pose2_b,
                                              self.FLAG)
            
        
        else:
            image_b, pose2_b, pose3_b = utils.data_prep.get_batch(imgFiles,
                                                                  pose2,
                                                                  pose3)
            
            image_b, pose2_b, pose3_b = utils.data_prep.crop_data_top_down(
                image_b,
                pose2_b,
                pose3_b, self.FLAG)
	
            image, pose2, pose3, vec_64, vec_32, vec_16, vec_8 = \
                utils.data_prep.get_vector_gt(image_b, pose2_b, pose3_b,
                                              self.FLAG)
    
        
        return image, vec_64, vec_32, vec_16, vec_8, pose3
    
    def get_next_train_batch(self):
        """
		:return: This Function basically gives out next batch data everytime its
		been called, as it has all the information it needs, I totally needed
		this function now my life becomes much simpler
		"""
        offset = min((self.train_iter * self.FLAG.batch_size), \
                     (self.train_data_size - self.FLAG.batch_size))
        mask_ = self.mask_train[offset:(offset + self.FLAG.batch_size)]
        if self.FLAG.train_2d == False:
            fd = self.get_dict(True, self.imgFiles[mask_], self.pose2[mask_],
                               self.pose3[mask_])
        else:
            fd = self.get_dict(True, self.imgFiles[mask_], self.pose2[mask_])
        
        self.train_iter += 1
        
        if self.train_iter * self.FLAG.batch_size > self.train_data_size:
            print('<<<<<<<<<<<<<<<<<:Epoch:>>>>>>>>>>>>>>>>> ')
            self.train_iter = 0
            self.mask_train = np.random.permutation(self.train_data_size)
        
        return fd[0], fd[1], fd[2], fd[3], fd[4]
