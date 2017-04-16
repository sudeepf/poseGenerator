import tensorflow as tf
import numpy as np
import utils.add_summary as summ
from numpy import unravel_index
import matplotlib.pyplot as plt


def compute_precision(prediction, gt, steps, mul_factor, num_joints):
    """" Given prediction stack, GT coordinates and scale between two """
    predictions_cord = get_coordinate(prediction, steps, num_joints)
    joint_wise_error = np.zeros((num_joints))
    data_point = 0
    for ii, pred_cord in enumerate(predictions_cord):
        gt_ = gt[ii]
        
        pred_cord = pred_cord.astype(float)
        # Get Root Joints ie Hips
        RJ1_pred = pred_cord[8, :]
        RJ2_pred = pred_cord[11, :]
        RJ1_gt = gt_[8, :]
        RJ2_gt = gt_[11, :]
        
        # Get mean of pose
        MR_gt = (RJ1_gt + RJ2_gt) / 2
        MR_pred = (RJ1_pred + RJ2_pred) / 2
        
        # Allign mean of pred to mean of GT
        
        pred_cord -= MR_pred
        gt_ -= MR_gt
        
        # Get Root limb length
        RL_pred = np.linalg.norm(RJ1_pred[0:2] - RJ2_pred[0:2])
        RL_gt = np.linalg.norm(RJ1_gt[0:2] - RJ2_gt[0:2])
        if RL_gt < 1e-4 or RL_pred == 0:
            continue
        # Get scale from limb length
        scale_ = RL_gt / RL_pred
        data_point += 1
        pred_cord[:, 2] /= mul_factor
        pred_cord[:, :] *= scale_
        
        # plt.scatter(x=pred_cord[:, 0], y=pred_cord[:, 1], c='r')
        # plt.scatter(x=gt_[:, 0], y=gt_[:, 1], c='b')
        # plt.show()
        
        for jj in xrange(num_joints):
            joint_wise_error[jj] += (np.linalg.norm(pred_cord[jj] - gt_[
                jj]))
    
    return (joint_wise_error / data_point)


def get_coordinate(prediction, steps, num_joints):
    out_shape = np.shape(prediction)
    pred_ = np.reshape(prediction, (out_shape[0], out_shape[1], out_shape[2],
                                    out_shape[2], num_joints))
    # print(np.shape(pred_))
    # plt.imshow(np.sum(pred_[0, :, :, 0, :], axis=2))
    # plt.show()
    # Pred_ size is now Batch - X - Y - Z - Joints
    # we need in Batch - Joint - 3(X,Y,Z)
    pred_ = np.rollaxis(pred_, 4, 1)
    pred_ = np.swapaxes(pred_, 2, 3)
    out_shape = np.shape(pred_)
    
    cords = np.zeros((out_shape[0], out_shape[1], 3))
    # Would need to itterate over the batch size and joints unfortunately
    for fi, frame in enumerate(pred_[:]):
        for ji, joint in enumerate(frame[:]):
            cords[fi, ji, :] = np.array(
                unravel_index(joint.argmax(), joint.shape))
    
    # plt.scatter(x=cords[0,:, 0], y=cords[0,:, 1], c='b')
    # plt.show()
    return cords


def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(tf.cast(argmax / (shape[1] * shape[2]), tf.float32))
    output_list.append(tf.cast(tf.mod(argmax, (shape[1] * shape[2])) / shape[2],
                               tf.float32))
    output_list.append(tf.cast(tf.mod((tf.mod(argmax, (shape[1] * shape[2])) /
                                       shape[2]), shape[2]), tf.float32))
    return tf.stack(output_list)


def get_coordinate_tensor(vol, steps, batch_size, vol_res, joints):
    """This Function Basically convert volumetric rep to coordinate based rep
	returns a list of coordinates for each batch and each step and each joint
	"""
    
    cords = []
    total_dim = np.sum(np.array(steps))
    shape_ = vol.get_shape().as_list()
    vol = tf.reshape(vol, [batch_size] + shape_[1:-1]
                                 + [shape_[-1] / joints,
                                  joints], name= 'some')


    for bi in xrange(batch_size):
        v = tf.slice(vol, [bi, 0, 0, 0, 0],
                     [1, shape_[-2], shape_[-2], total_dim, joints])
        v = tf.squeeze(v)
        step_c = 0
        for si in xrange(len(steps)):
            v_ = tf.slice(v, [0, 0, step_c, 0],
                          [vol_res, vol_res, steps[si], joints])
            step_c += steps[si]
            for j in xrange(joints):
                v_j = tf.slice(v_, [0, 0, 0, j],
                               [vol_res, vol_res, steps[si], 1])
                # v_j = tf.squeeze(v_j)
                v_jf = tf.argmax(tf.reshape(v_j, [-1]), axis=0)
                cord_ = unravel_argmax(v_jf, tf.cast(tf.shape(v_j), tf.int64))
                cords.append(cord_)
    return cords


def get_precision_MultiGPU(output, y, steps, FLAG):
    batch_size = FLAG.batch_size
    joints = FLAG.num_joints
    vol_res = FLAG.volume_res
    output_cords = []
    y_cords = []
    out_shape = output.get_shape().as_list()
    y_ = tf.reshape(y, [FLAG.batch_size] + out_shape[1:])
    output_cords.append(
        get_coordinate_tensor(output, steps, batch_size, vol_res,
                              joints))
    y_cords.append(get_coordinate_tensor(y_, steps, batch_size, vol_res,
                                         joints))

    # Convert list of arrays to tensor
    out_cord_t = tf.stack(output_cords)
    y_cord_t = tf.stack(y_cords)
    # Reshape to make it iterable
    out_cord_t = tf.reshape(out_cord_t, [len(output_cords) * batch_size,
                                         len(steps), joints, 3])
    y_cord_t = tf.reshape(y_cord_t, [len(output_cords) * batch_size, len(steps),
                                     joints, 3])
    
    # Iterate over tensors
    joint_prec = tf.reduce_mean(tf.norm(out_cord_t - y_cord_t, axis=3), axis=0)
    
    return joint_prec
