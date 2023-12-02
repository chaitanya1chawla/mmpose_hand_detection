import os
import sys
import numpy as np
import math

from scipy.spatial.transform import Rotation as R
#from IPython import embed

def invert(homogenous_matrix):
        
    inverse = np.zeros((4,4))
    rotation = R.from_matrix(homogenous_matrix[:3,:3])
    inverse[:3, :3] = rotation.inv().as_matrix()
    inverse[:3, -1] = -rotation.inv().apply(homogenous_matrix[:3,-1])
    inverse[-1, -1] = 1.

    return inverse

def invert_batch_matrix(homogenous_matrix):
    
    inverse = np.zeros((homogenous_matrix.shape[0], 4,4))
    rotation = R.from_matrix(homogenous_matrix[:,:3,:3])
    inverse[:, :3, :3] = rotation.inv().as_matrix()
    inverse[:, :3, -1] = -rotation.inv().apply(homogenous_matrix[:, :3,-1])
    inverse[:, -1, -1] = 1.

    return inverse

def create_matrix(position, orientation):

    homogenous_matrix = np.zeros((4,4))
    homogenous_matrix[:3,-1] = position
    homogenous_matrix[:3,:3] = R.from_quat(orientation).as_matrix()
    homogenous_matrix[-1,-1] = 1.

    return homogenous_matrix

def create_batch_matrix(position, orientation):

    traj_length = position.shape[0]
    homogenous_matrix = np.zeros((traj_length, 4,4))
    homogenous_matrix[:, :3,-1] = position
    homogenous_matrix[:, :3,:3] = R.from_quat(orientation).as_matrix()
    homogenous_matrix[:,-1,-1] = 1.

    return homogenous_matrix

def define_ground_world_transformation():

    rotation = R.from_euler('Z', 180, degrees=True)

    ground_in_world_homogenous_matrix = np.zeros((4,4))
    ground_in_world_homogenous_matrix[:3, :3] = rotation.as_matrix()
    ground_in_world_homogenous_matrix[:3, -1] = np.array([0.595, 0.0145, -0.15])
    ground_in_world_homogenous_matrix[-1, -1] = 1.

    return ground_in_world_homogenous_matrix

def transform_pose_from_ground_to_world(pose_array):

    ground_in_world_homogenous_matrix = define_ground_world_transformation()
    
    # Set object pose in homogenous matrix
    object_in_ground_hmat = np.zeros((pose_array.shape[0], 4,4))
    object_in_ground_hmat[:,:3, :3] = R.from_quat(pose_array[:,3:]).as_matrix()
    object_in_ground_hmat[:,:3, -1] = pose_array[:,:3]
    object_in_ground_hmat[:, -1, -1] = 1.

    # Transform
    # world_in_ground_hmat = invert(ground_in_world_homogenous_matrix)
    object_in_world_hmat = np.matmul(ground_in_world_homogenous_matrix, object_in_ground_hmat)

    # Retrieve pose. 
    object_in_world_quaternion = R.from_matrix(object_in_world_hmat[:, :3, :3]).as_quat()
    object_in_world_position = object_in_world_hmat[:, :3, -1]

    return object_in_world_position, object_in_world_quaternion

def transform_point_3d_from_cam_to_ground(points, gnd_R, gnd_t):

    idx = np.arange(3)
    points_in_cam_hmat = np.zeros((len(points), 4, 4))
    # Saving the rotation matrix as identity
    points_in_cam_hmat[:, idx, idx] = 1.
    points_in_cam_hmat[:, :3, -1] = np.array(points)
    points_in_cam_hmat[:, -1, -1] = 1.

    gnd_in_cam_hmat = np.zeros((4,4))
    gnd_in_cam_hmat[:3, :3] = gnd_R
    gnd_in_cam_hmat[:3, -1] = np.reshape(gnd_t,(3,))
    gnd_in_cam_hmat[-1, -1] = 1.

    cam_in_gnd_hmat = invert(gnd_in_cam_hmat)

    points_in_gnd_hmat = np.matmul(cam_in_gnd_hmat, points_in_cam_hmat)

    # Retrieve new points
    points_in_gnd_position = points_in_gnd_hmat[:, :3, -1]

    return points_in_gnd_position

def transform_pose_from_cam_to_ground(pose_R, pose_t, gnd_R, gnd_t):
    
    object_in_cam_hmat = np.zeros((len(pose_R), 4, 4))
    object_in_cam_hmat[:, :3, :3] = pose_R
    object_in_cam_hmat[:, :3, -1] = pose_t
    object_in_cam_hmat[:, -1, -1] = 1.

    gnd_in_cam_hmat = np.zeros((4,4))
    gnd_in_cam_hmat[:3, :3] = gnd_R
    gnd_in_cam_hmat[:3, -1] = np.reshape(gnd_t,(3,))
    gnd_in_cam_hmat[-1, -1] = 1.

    cam_in_gnd_hmat = invert(gnd_in_cam_hmat)

    # Transform
    # world_in_ground_hmat = invert(ground_in_world_homogenous_matrix)
    object_in_gnd_hmat = np.matmul(cam_in_gnd_hmat, object_in_cam_hmat)

    # Retrieve pose. 
    object_in_gnd_R = object_in_gnd_hmat[:, :3, :3]
    object_in_gnd_t = object_in_gnd_hmat[:, :3, -1]

    return object_in_gnd_t, object_in_gnd_R