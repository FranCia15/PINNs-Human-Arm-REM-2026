# Copyright 2023 HYBRID Team INCIA, UMR5287, CNRS, Université de Bordeaux

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% IMPORTS
# - Built-in
import copy
import math
# - Third-party
from typing import Any
import numpy as np
from numpy.linalg import norm
import functools
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R


# %% METHODS
def rot_mat_alpha(phi):
    """
    Elementary rotation matrix along X axis, with a possibly weird rotation
    direction convention (due to the fact that we work in left hand frame, 
    arbitrary rotation direction)
    """
    c = np.cos(phi)
    s = np.sin(phi)
    sq = np.array([[c, s], [-s, c]]).transpose((2, 0, 1))
    res = np.tile(np.eye(3), (len(phi), 1, 1))
    res[:, 1:, 1:] = sq
    return res


def rot_mat_beta(phi):
    """
    Elementary rotation matrix along X axis, with a possibly weird rotation
    direction convention (due to the fact that we work in left hand frame, 
    arbitrary rotation direction)
    """
    c = np.cos(phi)
    s = np.sin(phi)
    sq = np.array([[c, s], [-s, c]]).transpose((2, 0, 1))
    res = np.tile(np.eye(3), (len(phi), 1, 1))
    res[:, :2, :2] = sq

    return res

def rot_mat_x(phi):
    """
    Get elemental rotation matrix about first coordinate (x) by the given angle

    Parameters
    ----------
    phi : float
        Rotation angle, in rad

    Returns
    -------
    A 3-by-3 rotation matrix, as a ``np.matrix`` object
    [[1, 0, 0],
     [0, c, -s],
     [0, s, c]]
    """
    phi=np.atleast_1d(phi)
    c = np.cos(phi)
    s = np.sin(phi)
    full_mats = np.full((c.shape[0], 3, 3), np.identity(3))
    full_mats[:, 1, 1] = c
    full_mats[:, 1, 2] = -s
    full_mats[:, 2, 1] = s
    full_mats[:, 2, 2] = c
    return full_mats

def rot_mat_y(phi):
    """
    Get elemental rotation matrix about second coordinate (y) by the given
    angle

    Parameters
    ----------
    phi : float
        Rotation angle, in rad

    Returns
    -------
    A 3-by-3 rotation matrix, as a ``np.matrix`` object
    [[c, 0, s],
     [0, 1, 0],
     [-s, 0, c]]
    """
    phi=np.atleast_1d(phi)
    c = np.cos(phi)
    s = np.sin(phi)
    full_mats = np.full((c.shape[0], 3, 3), np.identity(3))
    full_mats[:, 2, 2] = c
    full_mats[:, 2, 0] = -s
    full_mats[:, 0, 2] = s
    full_mats[:, 0, 0] = c
    return full_mats

def rot_mat_z(phi):
    """
    Get elemental rotation matrix about third coordinate (z) by the given angle

    Parameters
    ----------
    phi : float
        Rotation angle, in rad

    Returns
    -------
    A 3-by-3 rotation matrix, as a ``np.matrix`` object
    [[c, -s, 0],
     [s, c, 0],
     [0, 0, 1]]
    """
    phi=np.atleast_1d(phi)
    c = np.cos(phi)
    s = np.sin(phi)
    full_mats = np.full((c.shape[0], 3, 3), np.identity(3))
    full_mats[:, 0, 0] = c
    full_mats[:, 0, 1] = -s
    full_mats[:, 1, 0] = s
    full_mats[:, 1, 1] = c
    return full_mats

def prod_conj(q1, q2):
    """
    Products of conjugates of first quaternions with second quaternions
    (useful for switching to another reference frame)
    """
    return (R.from_quat(q1).inv() * R.from_quat(q2)).as_quat()

def prod_quats(q1, q2):
    """
    Product of two quaternions
    """
    return (R.from_quat(q1) * R.from_quat(q2)).as_quat()


# def quat2rot_novec(quat):  # DEPRECATED quat2rot_mat is better
#     """
#     3x3 matrix expressing same rotation as given unitary quaternion
#     """
#     qx, qy, qz, qw = quat
#     return np.array([[1 - 2 * qy**2 - 2 * qz**2,
#                       2 * qx * qy - 2 * qz * qw,
#                       2 * qx * qz + 2 * qy * qw],
#                      [2 * qx * qy + 2 * qz * qw,
#                       1 - 2 * qx**2 - 2 * qz**2,
#                       2 * qy * qz - 2 * qx * qw],
#                      [2 * qx * qz - 2 * qy * qw,
#                       2 * qy * qz + 2 * qx * qw,
#                       1 - 2 * qx**2 - 2 * qy**2]])


def quat2rot_mat(quats):
    """
    3x3 matrices expressing same rotations as given quaternions
    """
    return R.from_quat(quats).as_matrix()


def rot_mat2quat(mats):
    """
    Quaternions expressing same rotations as given 3x3 matrices
    29/10/2021 https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/ (named "Alternative method" in the page)
    """
    return R.from_matrix(mats).as_quat()


def rot_mat2angles(mats, order_of_rots):
    """
    Extract euler angles from rotation matrices based on order of rotation
    """
    r: R = R.from_matrix(mats)
    return r.as_euler(order_of_rots)


def quat2angles(quats, order_of_rots, rot_prev: R = R.identity(),
                degrees=False):
    """
    Joint angle(s) extracted from given segment's and its parent's quaternions
    """
    return (rot_prev.inv() * R.from_quat(quats)).as_euler(order_of_rots, degrees=degrees)


def quats2config(quats_upper: np.ndarray, quats_fore: np.ndarray, quats_hand: np.ndarray, rot_ref: R = R.identity(),
                 degrees: bool = False):
    """
    Joint angles of whole arm extracted from quaternions of all three segments
    and of egocentric reference frame, if applicable

    Parameters:
        quats_upper (np array (X,4)): Quats expressing the rotation of the shoulder (in a world referential)
        quats_fore (np array (X,4)): Quats expressing the rotation of the elbow (in a world referential)
        quats_hand (np array (X,4)): Quats expressing the rotation of the wrist (in a world referential)
        rot_ref (array of scipy rotations): Expressing the rotation of a reference frame for the shoulder (in a world referential)
        degrees (bool): Specify the units of the output, False for radians, True for degrees

    Returns:
        config_angles (np array (X,7)): The joint angles computed as
            shoulder_pitch shoulder_roll shoulder_yaw(arm_yaw) elb_pitch wri_yaw(forearm_yaw) wri_pitch(hand_pitch) wri_roll(hand_roll)
    """
    rot_upper = R.from_quat(quats_upper)
    rot_fore = R.from_quat(quats_fore)
    rot_hand = R.from_quat(quats_hand)

    shou_eul_angles = (rot_ref.inv() * rot_upper).as_euler("XZY", degrees=degrees)
    elb_eul_angles = (rot_upper.inv() * rot_fore).as_euler("XZY", degrees=degrees)

    #Correction to get the full yaw of the forearm inside the wrist angles
    rot_fore_only_pitch = rot_upper * R.from_euler("X", elb_eul_angles[:,:1], degrees=degrees)
    
    wri_eul_angles = (rot_fore_only_pitch.inv() * rot_hand).as_euler("YXZ", degrees=degrees)

    #Concatenate then return
    config_angles = np.hstack((shou_eul_angles,
                                elb_eul_angles[:,0:1],
                                wri_eul_angles))

    return config_angles

def quats2configFromWristToShou(quats_upper, quats_fore, quats_hand, quats_ref=None,
                 unit="rad"):
    """
    OBSOLETE DO NOT USE THIS FUNCTION
    Joint angles of whole arm extracted from quaternions of all three segments
    and of egocentric reference frame, if applicable

    Parameters:
        quats_upper (np array (X,4)): Quats expressing the rotation of the shoulder (in a world referential)
        quats_fore (np array (X,4)): Quats expressing the rotation of the elbow (in a world referential)
        quats_hand (np array (X,4)): Quats expressing the rotation of the wrist (in a world referential)
        quats_ref (np array (X,4)): Quats expressing the rotation of a reference frame for the shoulder (in a world referential)
        unit (string "rad" or "deg"): Specify the units of the output

    Returns:
        config_angles (np array (X,7)): The joint angles computed as
            shoulder_pitch shoulder_roll shoulder_yaw(arm_yaw) elb_pitch wri_yaw(forearm_yaw) wri_pitch(hand_pitch) wri_roll(hand_roll)
    """
    # If quats_ref given, make sure it is 2-D before calling quat2angles
    if quats_ref is not None and len(quats_ref.shape) == 1:
        quats_ref = np.tile(quats_ref, (quats_upper.shape[0], 1))
    wri_eul_angles = quat2angles(quats_hand,
                                order_of_rots="ZXY",
                                quats_prev=quats_ref,
                                unit=unit)
    elb_eul_angles = quat2angles(quats_fore,
                                order_of_rots="YZX",
                                quats_prev=quats_hand,
                                unit=unit)
    shou_eul_angles = quat2angles(quats_upper,
                                order_of_rots="YZX",
                                quats_prev=quats_fore,
                                unit=unit)
    

    #Correction to get the full yaw of the forearm inside the wrist angles
    wri_eul_angles[:,2] +=  elb_eul_angles[:,0]

    #Concatenate then return
    config_angles = np.hstack((wri_eul_angles,
                                elb_eul_angles[:,2:3],
                                shou_eul_angles))

    return config_angles

def quat2vecY(quats):
    """
    3D coordinates of Y vectors (second column) from rotation matrices
    expressing same rotations as given quaternions
    """
    qx, qy, qz, qw = quats.T
    vecs = np.array([2 * (qx * qy - qz * qw),
                     1 - 2 * (qx ** 2 + qz ** 2),
                     2 * (qy * qz + qx * qw)])

    return vecs.T

def quat2vecZ(quats):
    """
    3D coordinates of Y vectors (second column) from rotation matrices
    expressing same rotations as given quaternions
    """
    qx, qy, qz, qw = quats.T
    vecs = np.array([2 * (qw * qy + qx * qz),
                     2 * (qy * qz - qw * qx),
                     1 - 2 * (qx ** 2 + qy ** 2)])

    return vecs.T

def quat2vecX(quats):
    """
    3D coordinates of Y vectors (second column) from rotation matrices
    expressing same rotations as given quaternions
    """
    qx, qy, qz, qw = quats.T
    vecs = np.array([1 - 2 * (qy ** 2 + qz ** 2),
                     2 * (qx * qy + qw * qz),
                     2 * (qx * qz + qw * qy)])

    return vecs.T

def config2quats(rads: list or np.ndarray, prev_rot: R = R.identity(), list_axis_order: list=["XZY", "X", "YXZ"]):
    """
    Get three quaternions describing orientations of arm segments corresponding
    to given articular angles

    Parameters:
        rads (list or np array of size 7): The 7 angles in radians from which to compute the quaternions, they must be in the correct order
        prev_mat (np array (:,3,3)): Rotation matrices that will be apply before the first computation of angles rotation
        list_axis_order (list of string): Each string represents the order in which to apply the rotation to generate a quaternion

    Returns:
        tuple_of_quats: (tuple(np array (X,4), np array (X,4), np array (X,4))): Quaterions corresponding to the angles (shoulder, elbow and wrist usually)
    """
    assert(len(rads) == 7)
    #rads = np.deg2rad(angles)
    # Get global rotation matrices

    tuple_of_quats = ()
    nb_elems_treated = 0
    for axis_order in list_axis_order:
        nb_elems_to_treat = len(axis_order)

        prev_rot = prev_rot * R.from_euler(axis_order, rads[nb_elems_treated:nb_elems_treated + nb_elems_to_treat])
        tuple_of_quats = (*tuple_of_quats, prev_rot.as_quat())

        nb_elems_treated += nb_elems_to_treat

    return tuple_of_quats

def config2quats_multi(rads: list or np.ndarray, prev_rot: R = R.identity(), list_axis_order: list=["XZY", "X", "YXZ"]):
    """
    Get three quaternions describing orientations of arm segments corresponding
    to given articular angles

    Parameters:
        rads (list or np array of size 7): The 7 angles in radians from which to compute the quaternions, they must be in the correct order
        prev_mat (np array (:,3,3)): Rotation matrices that will be apply before the first computation of angles rotation
        list_axis_order (list of string): Each string represents the order in which to apply the rotation to generate a quaternion

    Returns:
        tuple_of_quats: (tuple(np array (X,4), np array (X,4), np array (X,4))): Quaterions corresponding to the angles (shoulder, elbow and wrist usually)
    """
    #rads = np.deg2rad(angles)
    # Get global rotation matrices

    tuple_of_quats = ()
    nb_elems_treated = 0
    for axis_order in list_axis_order:
        nb_elems_to_treat = len(axis_order)

        prev_rot = prev_rot * R.from_euler(axis_order, rads[:, nb_elems_treated:nb_elems_treated + nb_elems_to_treat])
        tuple_of_quats = (*tuple_of_quats, prev_rot.as_quat())

        nb_elems_treated += nb_elems_to_treat

    return tuple_of_quats

def angles2rot_mat(angs: np.ndarray, prev_mat: np.ndarray = None, axis_order: str = "XZY"):
    """
    Return global rotation matrix corresponding to given axis order and articular
    angles
    The angles will always be applied from the first to the last alongside the second
        dimension of the numpy array

    Parameters:
        angs (np array (X,Y)): The angles used to create the rotation matrices
        prev_mat (np array (:,3,3)): Rotation matrices that will be apply before the angles rotations
        seg (string ["XZY", "X", "YXZ"]): Specify on which axis should be applied the angles MUST BE IN UPPERCASE

    Returns:
        mat (np array (X,3,3)): Rotation matrices corresponding to prev_mat then the angles rotations
    """
    r: R = R.from_euler(axis_order, angs)
    if prev_mat is not None:
        r = R.from_matrix(prev_mat) * r
        return r.as_matrix()
    else:
        result = r.as_matrix()
        if len(result.shape) != 3:
            return np.expand_dims(result, axis=0)
        return result

def create_transfo_mat(rot_mat, ori_pos):
    """Return transformation matrix from rotation matrix N x 3 x 3 and
    position N x 3
    """
    t_matrix = np.zeros((rot_mat.shape[0], 4 , 4))
    t_matrix[:, 3, 3] = np.ones((t_matrix.shape[0]))
    t_matrix[:, :3, :3] = rot_mat
    t_matrix[:, :3, 3] = ori_pos
    return t_matrix

def inverse_transfo_mat(transfo_mat):
    """Return the inverse transformation matrix
    Transpose rotation part and then apply transformation of origine
    """
    inv_mat = np.copy(transfo_mat)
    inv_mat[:, :3, :3] = np.swapaxes(inv_mat[:, :3, :3], 1 ,2)
    inv_mat[:, :3, 3:4] = -np.matmul(inv_mat[:, :3, :3], inv_mat[:, :3, 3:4])
    return inv_mat

def inverse_quat(quat):
    """Return the inverse transformation matrix
    Transpose rotation part and then apply transformation of origine
    """
    inv_quat = np.copy(quat)
    inv_quat[:, 3] = - inv_quat[:, 3]
    return inv_quat

def change_quat_ref_handeness(quats):
    """Change handeness of quaternion
    """
    quats[:, 0] = -quats[:, 0]
    quats[:, 3] = -quats[:, 3]
    return quats

def pos_3d_to_4d(pos_3d):
    """Convert position of Nx3 to Nx4 by adding 1 at the end of each position
    """
    pos_4d = np.hstack((pos_3d, np.ones((pos_3d.shape[0], 1))))
    return pos_4d

def pos_4d_to_3d(pos_4d):
    """Convert position of Nx4 to Nx3 by removing last index of each position
    """
    pos_3d = np.delete(pos_4d, 3, axis=1)
    return pos_3d

def get_obj_pos_from_ref1_to_ref2(
    obj_pos_ref1,
    ref2_pos_ref1,
    ref2_quat_ref1):
    """
    Take an Nx3 position and return the position in a new referential

    Parameters:
        obj_pos_ref1 (np array (X,3)): Position expressed in ref 1 to change to ref 2
        ref2_pos_ref1 (np array (X,3)): Position of ref 2 expressed in ref 1
        ref2_quat_ref1 (np array (X,4)): Quaternion of ref 2 expressed in ref 1

    Returns:
        pos_in_ref (np array (X,3)): Position now expressed in ref N
    """
    return R.from_quat(ref2_quat_ref1).apply(obj_pos_ref1 - ref2_pos_ref1, inverse=True)

def get_parent_pos_in_child_ref(
    child_pos_ref_parent,
    child_quat_ref_parent):
    """
    Returns the position of the parent expressed in the child

    Parameters:
        child_pos_ref_parent (np array (X,3)): Position of child expressed in parent
        child_quat_ref_parent (np array (X,4)): Quaternion of child expressed in parent

    Returns:
        parent_pos_in_child_ref (np array (X,3)): Position of parent expressed in child
    """
    parent_pos_in_child_ref = get_obj_pos_from_ref1_to_ref2(np.zeros(np.shape(child_pos_ref_parent)), child_pos_ref_parent, child_quat_ref_parent)

    return parent_pos_in_child_ref

def get_spherical_coordinates(
    pos_in_cartesian_coords,
    phi_particular_case_option="zero"):
    """
    Take an Nx3 position in cartesian coordinates and return the position in spherical coordinates

    Parameters:
        pos_in_cartesian_coords (np array (X,3)): Position in cartesian coordinates
        phi_particular_case_option (string): Specify the data put for the particular case of phi

    Returns:
        pos_in_spherical_coords (np array (X,3)): Position now expressed in spherical coordinates
    """
    # Based on: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    #Compute of radius
    radius = np.linalg.norm(pos_in_cartesian_coords, ord=2, axis=1)
    
    #Compute of theta
    theta = np.arccos(pos_in_cartesian_coords[:,2]/radius)

    #Compute of phi
    x = pos_in_cartesian_coords[:,0]
    y = pos_in_cartesian_coords[:,1]
    phi = np.zeros((pos_in_cartesian_coords.shape[0]))
    for i in range(pos_in_cartesian_coords.shape[0]):
        if x[i] > 0:
            phi[i] = np.arctan(y[i]/x[i])
        elif x[i] < 0:
            if y[i] >=0:
                phi[i] = np.arctan(y[i]/x[i]) + np.pi
            else:
                phi[i] = np.arctan(y[i]/x[i]) - np.pi
        else:
            if y[i] >0:
                phi[i] = np.pi/2
            elif y[i]<0:
                phi[i] = -np.pi/2
            else:
                if phi_particular_case_option=="previous":
                    if i != 0:
                        phi[i] = phi[i-1]
                elif phi_particular_case_option=="nan":
                    phi[i] = np.nan
    return np.concatenate((radius[:,None], theta[:,None], phi[:,None]), axis=1)

def get_angular_magnitude_of_quats(
    quats: 'TODO'):#TODO:Wrong computation
    """
    WRONG COMPUTATION
    Get the amount of rotation of a quaternion
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Recovering_the_axis-angle_representation

    Parameters:
        quats (np array (X,4)):The array of quats from which to compute angular magnitude.

    Returns:
        angular_magnitude (np array (X,1)):The angular magnitudes computed.
    """
    return 2*np.arctan2(np.sqrt(np.square(quats[:,0])+np.square(quats[:,1])+np.square(quats[:,2])),quats[:,3])


def quat2rot_vec(quat):
    '''''
    Function to compute rotation vector and angle from a quaternion
    q = cos(theta/2) + sin(theta/2)*[xi + yj + zk]
    theta = 2*arccos(w)
    without the angle magnitude 
    q_vec = (1/sin(theta/2))*[x,y,z]
    with the angle magnitude 
    q_vec = (theta/sin(theta/2))*[x,y,z]

    https://www.mathworks.com/help/fusion/ref/quaternion.rotvec.html

    Parameters:
        quats (np array (X,4)):The array of quats from which to compute angular magnitude.

    Returns:
        rot_vec (np array (X,3)):The rotation axis computed. 
        theta  (np array (X,1)):The angle computed. 

    '''
    w = quat[:, -1:] # w element of quaternion
    if w[0] > 1:
        w[0] = 0.99999999 # Can't be 1 because of the arccos that will create a zero for the division coming afterwards
    v = quat[:, :-1] # x,y,z element of quaternion
    theta = 2*np.arccos(w)
    rot_vec = (1/np.sin(theta/2))*v  # Vector is normalized
    #v = (theta/np.sin(theta/2))*v   # Distribuiting theta on the axis, vector not normalized
    return rot_vec, theta

def rot_vec2quat(rot_vec, theta):
    '''''
    Function to compute quaternion from rotation axis and angle

    Parameters:
        rot_vec (np array (X,3)):The rotation axis computed. 
        theta  (np array (X,1)):The angle computed.
    Returns:
        quats (np array (X,4)):The array of quats from which to compute angular magnitude.
    '''
    quat = np.concatenate((np.sin(theta/2)*rot_vec, np.cos(theta/2)), axis=1)
    return quat

def rot_mat2ortho6d(rot_mat):
    """
    Converts a rotation matrix to a 6d representation
    This function considers that the rotation matrix is normalized

    Parameters:
        rot_mat (np array (X,3,3)): Numpy array of rotation matrices

    Returns:
        ortho6d (np array (X,3,2)): Numpy array of rotation expressed in 6d
    """
    return rot_mat[:, :, :2]

def ortho6d2rot_mat(ortho6d):
    """
    Converts a 6d rotation representation to a rotation matrix
    According to :
    On the Continuity of Rotation Representations in Neural Networks
    Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li
    https://arxiv.org/abs/1812.07035
    Supplemental document section B

    Parameters:
        ortho6d (np array (X,3,2)): Numpy array of rotation expressed in 6d

    Returns:
        rot_mat (np array (X,3,3)): Numpy array of rotation matrices
    """
    rot_mat_v1 = ortho6d[:, :, 0]/np.linalg.norm(ortho6d[:, :, 0], axis=1, keepdims=True)
    rot_mat_v2 = ortho6d[:, :, 1] - np.sum(rot_mat_v1 * ortho6d[:, :, 1], axis=1, keepdims=True)*rot_mat_v1
    rot_mat_v2 = rot_mat_v2/np.linalg.norm(rot_mat_v2, axis=1, keepdims=True)
    rot_mat_v3 = np.cross(rot_mat_v1, rot_mat_v2)
    return np.concatenate(
        (
            np.expand_dims(rot_mat_v1, axis=2),
            np.expand_dims(rot_mat_v2, axis=2),
            np.expand_dims(rot_mat_v3, axis=2)
        ),
        axis=2)

def config2ortho6d_config(config):
    """
    Converts an euler angles config to an ortho 6d config

    Parameters:
        config (np array (X,7)): Numpy array of config rotations in euler angles

    Returns:
        ortho6d_config (np array (X,3,8)): Numpy array of config rotations in 6d
    """
    #sp sr
    ortho6d_shoulder = rot_mat2ortho6d(angles2rot_mat(config[:, 0:2], axis_order = "XZ"))

    #ay
    ortho6d_shouYaw =  rot_mat2ortho6d(angles2rot_mat(config[:, 2:3], axis_order = "Y"))

    #ep
    ortho6d_elbow =  rot_mat2ortho6d(angles2rot_mat(config[:, 3:4], axis_order = "X"))

    #fy hp hr
    ortho6d_wrist =  rot_mat2ortho6d(angles2rot_mat(config[:, 4:7], axis_order = "YXZ"))

    return np.concatenate((ortho6d_shoulder, ortho6d_shouYaw, ortho6d_elbow, ortho6d_wrist), axis=2)

def ortho6d_config2config(ortho6d_config):
    """
    Converts an an ortho 6d config to an euler angles config

    Parameters:
        ortho6d_config (np array (X,3,8)): Numpy array of config rotations in 6d

    Returns:
        config (np array (X,7)): Numpy array of config rotations in euler angles
        
    """
    #sp sr
    sp_sr = rot_mat2angles(ortho6d2rot_mat(ortho6d_config[:, : , 0:2]), order_of_rots="XZY")[:, 0:2]

    #ay
    ay = rot_mat2angles(ortho6d2rot_mat(ortho6d_config[:, : , 2:4]), order_of_rots="YXZ")[:, :1]

    #ep
    ep = rot_mat2angles(ortho6d2rot_mat(ortho6d_config[:, : , 4:6]), order_of_rots="X")[:, :1]

    #fy hp hr
    fy_hp_hr = rot_mat2angles(ortho6d2rot_mat(ortho6d_config[:, : , 6:8]), order_of_rots="YXZ")[:, :3]

    return np.concatenate((sp_sr, ay, ep, fy_hp_hr), axis=1)

def angle_between_two_vectors(v1, v2):
    """ Modified from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python """
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793 """
    """ v1, v2 are 2 np.array of dims (n, 3) """
    v1_u = normalize_vectors(v1)
    v2_u = normalize_vectors(v2)
    return np.arccos(np.clip(np.sum(np.multiply(v1_u, v2_u), axis=1), -1.0, 1.0))

def apply_euler_angles_to_vect_Y(angles, order:str="XZY", degrees:bool=False):
    """ Apply vector (0,1,0) from alpha beta angles to get a point on a sphere """
    """ Angles must be a np array of dims (n,3) """
    vector = np.array([0,1,0])
    r = R.from_euler(order,angles, degrees=degrees)
    vectors = r.apply(vector)
    return vectors

def angle_distance(a, b):
    """Compute the lowest distance between two angles in radian"""
    """a and b are in shape (n,1)"""
    return abs(((a-b) + math.pi) % (2*math.pi) - math.pi)

def angles2rot_mat_with_offset(angles, offset, axis_order = "XZY"):
    """Compute angles to rot_mat and reverse the offset"""
    offset_mat = quat2rot_mat(offset)
    rot_mat_with_offset = angles2rot_mat(angles, axis_order=axis_order)

    #As the offset matrix is a rotation matrix, it is always inversible
    rot_mat = np.matmul(rot_mat_with_offset, np.linalg.inv(offset_mat))
    return rot_mat


def angular_velocities_betweenTwoQuats(q1, q2, dt):
# Adapted from: https://www.euclideanspace.com/physics/kinematics/angularvelocity/
    q_transformation = prod_quats(q2, inverse_quat(q1))
    axis_angle_transformation = np.linalg.norm(R.from_quat(q_transformation).as_rotvec(), axis=1)
    return axis_angle_transformation / dt


def angular_velocities_betweenTwoQuats_approx(q1, q2, dt):
# Adapted from: https://mariogc.com/post/angular-velocity-quaternions/
# Warning: this approximation is approximative
    x = 0
    y = 1
    z = 2
    w = 3
    return (2 / dt) * np.array([
        q1[:, w]*q2[:, x] - q1[:, x]*q2[:, w] - q1[:, y]*q2[:, z] + q1[:, z]*q2[:, y],
        q1[:, w]*q2[:, y] + q1[:, x]*q2[:, z] - q1[:, y]*q2[:, w] - q1[:, z]*q2[:, x],
        q1[:, w]*q2[:, z] - q1[:, x]*q2[:, y] + q1[:, y]*q2[:, x] - q1[:, z]*q2[:, w]]).T

def get_quat_look_at_target(position, target, up):
    # https://www.programcreek.com/python/?CodeExample=look+at
    # position is the origin of the new referentiel np array (X,3)
    # target is the direction to look at in the old referentiel np array (X,3)
    # up is the up direction for the new referentiel np array (X,3)
        
    forward = np.subtract(target, position)
    forward = np.divide(forward, np.expand_dims(np.linalg.norm(forward, axis=1), axis=1))

    right = np.cross(forward, up)
        
    # if forward and up vectors are parallel, right vector is zero; 
    #   fix by perturbing up vector a bit
    epsilon = np.repeat(np.expand_dims(np.array([0.001, 0, 0]), axis=0), right.shape[0], axis=0)
    forward_up_parallel = np.where(np.linalg.norm(right, axis=1) < 0.001)
    right[forward_up_parallel] = np.cross(forward[forward_up_parallel], up[forward_up_parallel] + epsilon[forward_up_parallel])

    right = np.divide(right, np.expand_dims(np.linalg.norm(right, axis=1), axis=1))
        
    up = np.cross(right, forward)
    up = np.divide(up, np.expand_dims(np.linalg.norm(up, axis=1), axis=1))

    return rot_mat2quat(np.moveaxis(np.array([[right[:, 0], up[:, 0], -forward[:, 0]], 
                                              [right[:, 1], up[:, 1], -forward[:, 1]], 
                                              [right[:, 2], up[:, 2], -forward[:, 2]]]),
                                    source=-1,
                                    destination=0))

def align_to_direction_vector(direction_vector, original_rot):
    """
    Apply a rotation to another rotation in order that the z axis of the rotation is lined up
    with a direction vector. The rotation is limited to the Yaw so the target is not tilted.

    Original_rot must be a quaternion (or array of quaternions)
    """

    #Work in 2 dims
    if direction_vector.ndim == 1: direction_vector = np.expand_dims(direction_vector, axis=0)
    if original_rot.ndim == 1: original_rot = np.expand_dims(original_rot, axis=0)

    direction_vector_ref_original_rot = R.from_quat(original_rot).apply(direction_vector, inverse=True)

    #Retrieve the angle to apply on the yaw
    #(angle between the positive z-axis and projection of direction vector onto the x-z plane)
    yaw = np.arctan2(direction_vector_ref_original_rot[:,0], direction_vector_ref_original_rot[:,2])

    #Transform to rotmat and then quat to apply to original rot
    return (R.from_quat(original_rot)*R.from_euler("Y", yaw)).as_quat()


def normalize_vectors(vectors):
    return vectors / np.expand_dims(np.linalg.norm(vectors, ord=2, axis=-1), axis=1)

def compute_direction_vector_in_tgt_ref(basePos, tgtPos, tgtQuat):
    # DEPRECATED -> use normalize vector and do the ref change yourself if you need it
    #Retrieve the direction vector between the base and tgt and normalize it
    direction_vector = basePos - tgtPos
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    #Put the direction vector in the referential of the tgt centered in 0,0,0
    direction_vector = get_obj_pos_from_ref1_to_ref2(direction_vector, np.array([0,0,0]), tgtQuat)
    return direction_vector

def preserve_sign_quat(base_quat, quat):
    # Change sign of a quaternion based on other one 
    # A quaternion q and -q represents the same rotation
    if base_quat.ndim == 1: base_quat = np.expand_dims(base_quat, axis=0)
    if quat.ndim == 1: quat = np.expand_dims(quat, axis=0)
 
    if (base_quat[:, -1] > 0 and quat[:, -1] < 0) or (base_quat[:, -1] < 0 and quat[:, -1] > 0):
        quat = -quat
    return quat

def get_obj_from_ref1_to_ref2(obj_pos_ref1, obj_quat_ref1, ref1_pos_ref2, ref1_quat_ref2):
    obj_pos_ref2 = get_obj_pos_from_ref1_to_ref2(
            obj_pos_ref1 = obj_pos_ref1,
            ref2_pos_ref1 = get_parent_pos_in_child_ref(
                ref1_pos_ref2,
                ref1_quat_ref2
            ),
            ref2_quat_ref1 = inverse_quat(ref1_quat_ref2)
        )
    obj_quat_ref2 = prod_quats(
            q1 = ref1_quat_ref2,
            q2 = obj_quat_ref1
        )
    return obj_pos_ref2, obj_quat_ref2


def angle_between_two_quat_vector_wise(v1, v2, vector_idx):
    """ vector_idx: 0, 1, 2 for x, y, z respectively"""
    return angle_between_two_vectors(
        v1= R.from_quat(v1).as_matrix()[:, vector_idx],
        v2= R.from_quat(v2).as_matrix()[:, vector_idx])
