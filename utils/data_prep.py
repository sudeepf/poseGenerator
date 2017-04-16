# for reading dataset

import scipy.io
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tensorflow as tf
import math

def get_list_all_training_frames(list_of_mat, flag2d):
    pose3_ = []
    pose2_ = []
    files_ = []
    
    for ind, mFile in enumerate(list_of_mat):
        mat = scipy.io.loadmat(mFile)
        pose2_.append(mat['poses2'])
        if flag2d == False:
            pose3_.append(mat['poses3'])
        files_.append(mat['imgs'])
        ratio = 100 * (float(ind) / float(len(list_of_mat)))
        if ratio % 10 == 0:
            print('Successfully loaded -> ', ratio, '%')
    
    pose3 = None
    if flag2d==False:
        pose3 = np.concatenate(pose3_, axis=0)
    
    pose2 = np.concatenate(pose2_, axis=0)
    files = np.concatenate(files_, axis=0)
    
    return files, pose2, pose3


def get_list_all_testing_frames(list_of_mat):
    pose3_ = []
    pose2_ = []
    files_ = []
    gt_ = []
    for ind, mFile in enumerate(list_of_mat):
        mat = scipy.io.loadmat(mFile)
        pose2_.append(mat['poses2'])
        pose3_.append(mat['poses3'])
        files_.append(mat['imgs'])
        gt_.append(mat['p3GT'])
        ratio = 100 * (float(ind) / float(len(list_of_mat)))
        if ratio % 10 == 0:
            print('Successfully loaded -> ', ratio, '%')
    
    pose3 = np.concatenate(pose3_, axis=0)
    pose2 = np.concatenate(pose2_, axis=0)
    files = np.concatenate(files_, axis=0)
    gt = np.concatenate(gt_, axis=0)
    return files, pose2, pose3, gt


def get_batch(imgFiles, pose2, pose3=None):
    data = []
    ii = 0
    for name in imgFiles:
        ii += 1
        im = misc.imread(name[:])
        data.append(im)
    return data, pose2, pose3

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox = origin[0]
    oy = origin[1]
    px = point[0]
    py = point[1]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array([qx, qy])

def crop_data_top_down(images, pose2, pose3=None, FLAG=None):
    num_data_points = len(images)
    images_ = []
    pose2_ = []
    pose3_ = []
    for ii in xrange(num_data_points):
        im = images[ii]
        #print(np.max(im))
        imSize = min(np.shape(im)[1], np.shape(im)[0])
        p2 = pose2[ii]  # + Cam_C
        
        #plt.imshow(im)
        #plt.scatter(p2_v[:, 0], p2_v[:, 1])
        #plt.show()
        #Rotate Dataset
        

        p2_v = p2[np.min(p2, -1) > 0]
        #plt.imshow(im)
        #plt.scatter(p2_v[:,0], p2_v[:,1])
        #plt.show()
        min_ = np.min(p2_v, axis=0)
        max_ = np.max(p2_v, axis=0)
        hW = np.max(max_ - min_)
        midP = np.mean(p2_v, axis=0)
        
        verSkw = np.random.uniform(0.2, 0.8)
        horizSkw = np.random.uniform(0.3, 0.5)
        incSiz = np.random.uniform(hW * -0.0, hW * 0.8)

        if FLAG.train_2d == True:
            verSkw = np.random.uniform(0.5, 0.5)
            horizSkw = np.random.uniform(0.5, 0.5)
            incSiz = np.random.uniform(hW * 0.5, hW * 0.5)

        # hW /= 2
        hW += incSiz
        skw = [verSkw, horizSkw]
        min_ = midP - skw * np.array(hW)
        
        hW = hW.astype(np.int)
        min_[1] = max(min_[1], 0)
        min_[0] = max(min_[0], 0)
        max_[1] = min((midP[1] + hW*0.5), np.shape(im)[0])
        max_[0] = min((midP[0] + hW*0.5), np.shape(im)[1])
        
        # Debugging Stuff
        # implot = plt.imshow(im)
        # plt.scatter(x=p2[:, 0], y=p2[:, 1], c='r')
        # plt.scatter(midP[0], midP[1], c='b')
        # plt.scatter(min_[0], min_[1], c='b')
        # plt.scatter(max_[0], max_[1], c='g')
        
        # plt.show()
        min_ = min_.astype(np.int)
        max_ = max_.astype(np.int)
        im_ = im[min_[1]:max_[1], min_[0]:max_[0]]
        p2 -= min_
        if pose3 is not None:
            pose3[ii, :, :2] -= min_

        rotate_ = np.random.uniform(-40., 40.)
        im_ = misc.imrotate(im_, -1 * rotate_).astype(np.float32)
        midd = np.array([np.shape(im_)[1], np.shape(im_)[0]]) / 2
        rotate_rad = (rotate_ * 3.14) / 180.
        if FLAG.train_2d == True:
            for indd, p_tmp in enumerate(p2):
                p2[indd] = rotate(midd, tuple(p_tmp), rotate_rad)
        else:
            for indd, p_tmp in enumerate(pose3[ii, :]):
                pose3[ii, indd, :2] = rotate(midd, tuple(p_tmp[:2]), rotate_rad)
                p2[indd] = pose3[ii, indd, :2]
                
        # Change color channel contrast randomly for 3D humanpose dataset
        #if FLAG.train_2d == False:
            #im_ = im_.astype(np.float32)
            #rand_r = np.random.uniform(0.5, 1)
            #rand_g = np.random.uniform(0.5, 1)
            #rand_b = np.random.uniform(0.5, 1)
            #print(np.max(im_[:, :, 0]))
            #im_[:, :, 0] = im_[:, :, 0]* rand_r
            #im_[:, :, 0] = im_[:, :, 0]* rand_g
            #im_[:, :, 0] = im_[:, :, 0]* rand_b
        
        images_.append(im_)
        pose2_.append(p2)
        #plt.imshow(im_)
        #plt.scatter(p2[:,0], p2[:,1])
        #plt.show()
        
        if pose3 is not None:
            pose3_.append(pose3[ii, :, :])
            
    if FLAG.train_2d == True:
        return images_, pose2_, None
        
    
    return images_, pose2_, pose3_


def data_vis(image, pose2, pose3, Cam_C, ind):
    im = image[ind]
    p2 = pose2[ind]
    p3 = pose3[ind]
    implot = plt.imshow(im)
    plt.scatter(x=p2[:, 0], y=p2[:, 1], c='r')
    plt.scatter(x=p3[:, 0], y=p3[:, 1], c='b')
    plt.show()


def gaussian(x, mu, sig, max_prob=1):
    const_ = 1. / (sig * 2.50599)
    return const_ * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def plot_3d(image, threshold=0.5):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    p = p[:, :, ::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    plt.show()


def volumize_gt(image_b, pose2_b, pose3_b, resize_factor, im_resize_factor,
                sigma, mul_factor, max_prob, FLAG):
    num_of_data = FLAG.batch_size
    batch_data = np.empty((0, 14, resize_factor, resize_factor, resize_factor))
    pose2 = []
    pose3 = []
    image = []
    for ii in xrange(num_of_data):
        # print (ii, im_resize_factor, np.shape(image_b[ii]))
        im_ = misc.imresize(image_b[ii], (im_resize_factor, im_resize_factor))
        size_scale_ = np.array(np.shape(image_b[ii])[:2], dtype=np.float) / \
                      np.array(resize_factor, dtype=np.float)
        p2_ = pose2_b[ii] / size_scale_
        p3_ = pose3_b[ii]
        p3_[:, 0:2] = p3_[:, 0:2] / size_scale_
        p3_[:, 2] = p3_[:, 2] / np.mean(size_scale_)
        p3_[:, 2] *= mul_factor
        p3_[:, 2] += resize_factor / 2
        
        vec_x = np.empty((1, resize_factor))
        vec_y = np.empty((1, resize_factor))
        vec_z = np.empty((1, resize_factor))
        volume_b = np.empty((0, resize_factor, resize_factor, resize_factor))
        # vol_joint_ = np.zeros((64,64,64))
        for jj in xrange(14):
            for kk in xrange(resize_factor):
                vec_x[0, kk] = gaussian(kk, p3_[jj, 0], sigma, max_prob)
                vec_y[0, kk] = gaussian(kk, p3_[jj, 1], sigma, max_prob)
                vec_z[0, kk] = gaussian(kk, p3_[jj, 2], sigma, max_prob)
            bub = np.expand_dims(vec_y.transpose().dot(vec_x), axis=0)
            vol_joint = np.tensordot(bub, vec_z.transpose(), axes=([0], [1]))
            vol_joint = np.expand_dims(vol_joint, axis=0)
            volume_b = np.concatenate((volume_b, vol_joint), axis=0)
        # vol_joint_ += vol_joint
        # plot_3d(vol_joint_)
        volume_b = np.expand_dims(volume_b, axis=0)
        batch_data = np.concatenate((batch_data, volume_b), axis=0)
        pose2.append(p2_)
        pose3.append(p3_)
        image.append(im_)
    
    return batch_data, image, pose2, pose3


def get_vector_gt(image_b, pose2_b, pose3_b, FLAG):
    num_of_data = FLAG.batch_size
    vec_64 = np.empty((FLAG.batch_size, 3, FLAG.num_joints, FLAG.volume_res))
    vec_32 = np.empty(
        (FLAG.batch_size, 3, FLAG.num_joints, FLAG.volume_res // 2))
    vec_16 = np.empty(
        (FLAG.batch_size, 3, FLAG.num_joints, FLAG.volume_res // 4))
    vec_8 = np.empty(
        (FLAG.batch_size, 3, FLAG.num_joints, FLAG.volume_res // 8))
    pose2 = []
    pose3 = []
    image = np.empty((FLAG.batch_size, FLAG.image_res, FLAG.image_res, 3))
    
    for ii in xrange(num_of_data):
        # print (ii, im_resize_factor, np.shape(image_b[ii]))
        im_ = misc.imresize(image_b[ii], (FLAG.image_res, FLAG.image_res))
        size_scale_ = np.array([np.shape(image_b[ii])[1],np.shape(image_b[ii])[0]], dtype=np.float) / \
                      np.array([FLAG.volume_res, FLAG.volume_res], dtype=np.float)
        
        p2_ = pose2_b[ii] / size_scale_
        p3_ = pose3_b[ii]
        p3_[:, 0:2] = p3_[:, 0:2] / size_scale_
        p3_[:, 2] = p3_[:, 2] / np.mean(size_scale_)
        p3_[:, 2] *= FLAG.mul_factor
        p3_[:, 2] += FLAG.volume_res / 2
        
        for jj in xrange(14):
            for kk in xrange(FLAG.volume_res):
                vec_64[ii, 0, jj, kk] = gaussian(kk, p3_[jj, 0], FLAG.sigma,
                                                 FLAG.joint_prob_max)
                vec_64[ii, 1, jj, kk] = gaussian(kk, p3_[jj, 1], FLAG.sigma,
                                                 FLAG.joint_prob_max)
                vec_64[ii, 2, jj, kk] = gaussian(kk, p3_[jj, 2], FLAG.sigma,
                                                 FLAG.joint_prob_max)
        
        for jj in xrange(14):
            for kk in xrange(FLAG.volume_res / 2):
                vec_32[ii, 0, jj, kk] = gaussian(kk, p3_[jj, 0] // 2,
                                                 FLAG.sigma,
                                                 FLAG.joint_prob_max)
                vec_32[ii, 1, jj, kk] = gaussian(kk, p3_[jj, 1] // 2,
                                                 FLAG.sigma,
                                                 FLAG.joint_prob_max)
                vec_32[ii, 2, jj, kk] = gaussian(kk, p3_[jj, 2] // 2,
                                                 FLAG.sigma,
                                                 FLAG.joint_prob_max)
        
        for jj in xrange(14):
            for kk in xrange(FLAG.volume_res / 4):
                vec_16[ii, 0, jj, kk] = gaussian(kk, p3_[jj, 0] // 4,
                                                 FLAG.sigma,
                                                 FLAG.joint_prob_max)
                vec_16[ii, 1, jj, kk] = gaussian(kk, p3_[jj, 1] // 4,
                                                 FLAG.sigma,
                                                 FLAG.joint_prob_max)
                vec_16[ii, 2, jj, kk] = gaussian(kk, p3_[jj, 2] // 4,
                                                 FLAG.sigma,
                                                 FLAG.joint_prob_max)
        
        for jj in xrange(14):
            for kk in xrange(FLAG.volume_res / 8):
                vec_8[ii, 0, jj, kk] = gaussian(kk, p3_[jj, 0] // 8, FLAG.sigma,
                                                FLAG.joint_prob_max)
                vec_8[ii, 1, jj, kk] = gaussian(kk, p3_[jj, 1] // 8, FLAG.sigma,
                                                FLAG.joint_prob_max)
                vec_8[ii, 2, jj, kk] = gaussian(kk, p3_[jj, 2] // 8, FLAG.sigma,
                                                FLAG.joint_prob_max)
        
        pose2.append(p2_)
        pose3.append(p3_)
        image[ii, :, :, :] = im_
    
    return image, pose2, pose3, vec_64, vec_32, vec_16, vec_8


def get_vector_gt_2d(image_b, pose2_b, FLAG):
    num_of_data = FLAG.batch_size
    vec_64 = np.empty((FLAG.batch_size, 2, FLAG.num_joints, FLAG.volume_res))
    vec_32 = np.empty(
        (FLAG.batch_size, 2, FLAG.num_joints, FLAG.volume_res // 2))
    vec_16 = np.empty(
        (FLAG.batch_size, 2, FLAG.num_joints, FLAG.volume_res // 4))
    vec_8 = np.empty(
        (FLAG.batch_size, 2, FLAG.num_joints, FLAG.volume_res // 8))
    pose2 = []
    
    image = np.empty((FLAG.batch_size, FLAG.image_res, FLAG.image_res, 3))
    
    for ii in xrange(num_of_data):
        
        im_ = misc.imresize(image_b[ii], (FLAG.image_res, FLAG.image_res))
        size_scale_ = np.array([np.shape(image_b[ii])[1],np.shape(image_b[ii])[0]], dtype=np.float) / \
                      np.array([FLAG.volume_res, FLAG.volume_res], dtype=np.float)
        p2_ = pose2_b[ii] / size_scale_
        #plt.imshow(im_)
        #plt.show()
        for jj in xrange(14):
            for kk in xrange(FLAG.volume_res):
                vec_64[ii, 0, jj, kk] = gaussian(kk, p2_[jj, 0], FLAG.sigma,
                                                 FLAG.joint_prob_max)
                vec_64[ii, 1, jj, kk] = gaussian(kk, p2_[jj, 1], FLAG.sigma,
                                                 FLAG.joint_prob_max)
        
        for jj in xrange(14):
            for kk in xrange(FLAG.volume_res / 2):
                vec_32[ii, 0, jj, kk] = gaussian(kk, p2_[jj, 0] // 2,
                                                 FLAG.sigma,
                                                 FLAG.joint_prob_max)
                vec_32[ii, 1, jj, kk] = gaussian(kk, p2_[jj, 1] // 2,
                                                 FLAG.sigma,
                                                 FLAG.joint_prob_max)
        
        for jj in xrange(14):
            for kk in xrange(FLAG.volume_res / 4):
                vec_16[ii, 0, jj, kk] = gaussian(kk, p2_[jj, 0] // 4,
                                                 FLAG.sigma,
                                                 FLAG.joint_prob_max)
                vec_16[ii, 1, jj, kk] = gaussian(kk, p2_[jj, 1] // 4,
                                                 FLAG.sigma,
                                                 FLAG.joint_prob_max)
        
        for jj in xrange(14):
            for kk in xrange(FLAG.volume_res / 8):
                vec_8[ii, 0, jj, kk] = gaussian(kk, p2_[jj, 0] // 8, FLAG.sigma,
                                                FLAG.joint_prob_max)
                vec_8[ii, 1, jj, kk] = gaussian(kk, p2_[jj, 1] // 8, FLAG.sigma,
                                                FLAG.joint_prob_max)
        
        pose2.append(p2_)
        image[ii] = im_

    return image, pose2, vec_64, vec_32, vec_16, vec_8


def volumize_vec_gpu(tensor_x, tensor_y, tensor_z, scale, FLAG):
    """
	
	:param tensor_x: Probability distribution of GroundTruth along x axis
	:param tensor_y: Probability distribution of GroundTruth along y axis
	:param tensor_z: Probability distribution of GroundTruth along z axis
	:param FLAG: Parameters
	:return: Volumized representation for all joints in form of
	Batch - X - Y - Z - Joints
	
	"""
    list_b = []
    for ii in xrange(FLAG.batch_size):
        list_j = []
        for jj in xrange(FLAG.num_joints):
            vol = tf.matmul(tf.transpose(tensor_y[ii, jj:jj + 1]),
                            tensor_x[ii, jj:jj + 1])
            vol = tf.reshape(vol, [1, FLAG.volume_res // scale,
                                   FLAG.volume_res // scale])
            vol = tf.tensordot(vol, tf.transpose(tensor_z[ii, jj:jj + 1]),
                               axes=[[
                                   0], [1]])
            vol = tf.expand_dims(vol, 3)
            list_j.append(vol)
        list_b.append(tf.concat(list_j, 3))
    
    return tf.stack(list_b, 0)


def heatmap_vec_gpu(tensor_x, tensor_y, scale, FLAG):
    """

	:param tensor_x: Probability distribution of GroundTruth along x axis
	:param tensor_y: Probability distribution of GroundTruth along y axis
	:param tensor_z: Probability distribution of GroundTruth along z axis
	:param FLAG: Parameters
	:return: Volumized representation for all joints in form of
	Batch - X - Y - Z - Joints

	"""
    list_b = []
    for ii in xrange(FLAG.batch_size):
        list_j = []
        for jj in xrange(FLAG.num_joints):
            vol = tf.matmul(tf.transpose(tensor_y[ii, jj:jj + 1]),
                            tensor_x[ii, jj:jj + 1])
            vol = tf.reshape(vol, [FLAG.volume_res // scale,
                                   FLAG.volume_res // scale, 1])
            list_j.append(vol)
        list_b.append(tf.concat(list_j, 2))
    
    return tf.stack(list_b, 0)


def prepare_output(batch_data, steps=[1, 2, 4, 64]):
    out_res = np.shape(batch_data)[0]
    batch_size = np.shape(batch_data)[1]
    output = np.empty((0, batch_size, 14, out_res, out_res))
    for ii in steps:
        slice_ind = out_res / ii
        slice_start = 0
        for slice_end in range(slice_ind - 1, out_res, slice_ind):
            out_i = np.empty((0, 14, out_res, out_res))
            for data in xrange(batch_size):
                out_ = np.empty((0, out_res, out_res))
                vol_joint_ = np.zeros((64, 64, 64))
                for j in xrange(14):
                    data_j = batch_data[:, data, :, :, j]
                    slice_ = np.sum(data_j[slice_start:slice_end + 1, :, :],
                                    axis=0)
                    
                    slice_ = np.expand_dims(slice_, axis=0)
                    out_ = np.concatenate((out_, slice_), axis=0)
                
                out_ = np.expand_dims(out_, axis=0)
                out_i = np.concatenate((out_i, out_), axis=0)
            out_i = np.expand_dims(out_i, axis=0)
            output = np.concatenate((output, out_i), axis=0)
            slice_start = slice_end + 1
    
    return np.array(output)



def prepare_output_gpu(batch_data, steps, FLAG):
    """input dims are
		# Batch - X - Y - Z - Joints
		We Want #Batch - X- Y- Z_*Joints"""
    list_b = []
    for b in xrange(FLAG.batch_size):
        list_fna = []
        for ss in steps:
            slice_ind = FLAG.volume_res / ss
            slice_start = 0
            for slice_end in xrange(slice_ind - 1, FLAG.volume_res, slice_ind):
                list_j = []
                for jj in xrange(FLAG.num_joints):
                    app = tf.expand_dims(tf.reduce_sum(
                        batch_data[b, :, :, slice_start:slice_end + 1, jj], 2),
                        2)
                    list_j.append(tf.expand_dims(app, 3))
                list_fna.append(tf.concat(list_j, 3))
                slice_start = slice_end + 1
        list_b.append(tf.concat(list_fna, 2))
    out_ = tf.stack(list_b, 0)
    return tf.reshape(out_, [FLAG.batch_size, FLAG.volume_res, FLAG.volume_res,
                             FLAG.num_joints * sum(steps)])
