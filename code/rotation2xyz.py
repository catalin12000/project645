
import numpy as np, math, re
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import transforms3d.euler as euler, read_bvh_hierarchy, cv2 as cv
import rotation2xyz as rc
import torch
import functools

def get_child_dict(skel):
    child_dict = {}
    for t in skel.keys():
        parent = skel[t]['parent']
        if parent in child_dict.keys():
            child_dict[parent].append(t)
        else:
            child_dict[parent] = [
             t]

    return child_dict


def get_hip_transform(motion, skel):
    offsets_t = motion[0:3]
    Zrotation = motion[3]
    Yrotation = motion[4]
    Xrotation = motion[5]
    theta = [
     Xrotation, Yrotation, Zrotation]
    Rotation = eulerAnglesToRotationMatrix_hip(theta)
    Transformation = np.zeros((4, 4))
    Transformation[0:3, 0:3] = Rotation
    Transformation[3][3] = 1
    Transformation[0][3] = offsets_t[0]
    Transformation[1][3] = offsets_t[1]
    Transformation[2][3] = offsets_t[2]
    return Transformation


def get_skeleton_position(motion, non_end_bones, skel):
    pos_dict = OrderedDict()
    for bone in skel.keys():
        pos = get_pos(bone, motion, non_end_bones, skel)
        pos_dict[bone] = pos[0:3]

    return pos_dict


def get_bone_start_end(positions, skeleton):
    bone_list = []
    for bone in positions.keys():
        if bone != 'hip':
            bone_end = positions[bone]
            bone_start = positions[skeleton[bone]['parent']]
            bone_tuple = (bone_start, bone_end)
            bone_list.append(bone_tuple)

    return bone_list


def rotation_dic_to_vec(rotation_dictionary, non_end_bones, position):
    motion_vec = np.zeros(6 + len(non_end_bones) * 3)
    motion_vec[0:3] = position['hip']
    motion_vec[3] = rotation_dictionary['hip'][2]
    motion_vec[4] = rotation_dictionary['hip'][1]
    motion_vec[5] = rotation_dictionary['hip'][0]
    for i in range(0, len(non_end_bones)):
        motion_vec[3 * (i + 2)] = rotation_dictionary[non_end_bones[i]][2]
        motion_vec[3 * (i + 2) + 1] = rotation_dictionary[non_end_bones[i]][0]
        motion_vec[3 * (i + 2) + 2] = rotation_dictionary[non_end_bones[i]][1]

    return motion_vec


def get_pos(bone, motion, non_end_bones, skel):
    global_transform = np.dot(get_hip_transform(motion, skel), get_global_transform(bone, skel, motion, non_end_bones))
    position = np.dot(global_transform, np.array([0, 0, 0, 1])[:, np.newaxis])
    return position


def get_global_transform(bone, skel, motion, non_end_bones):
    parent = skel[bone]['parent']
    Transformation = get_relative_transformation(bone, non_end_bones, motion, skel)
    while parent != None:
        parent_transformation = get_relative_transformation(parent, non_end_bones, motion, skel)
        Transformation = np.dot(parent_transformation, Transformation)
        parent = skel[parent]['parent']

    return Transformation


def get_relative_transformation(bone, non_end_bones, motion, skel):
    end_bone = 0
    try:
        bone_index = non_end_bones.index(bone)
    except:
        end_bone = 1

    if end_bone == 0:
        Zrotation = motion[6 + 3 * bone_index]
        Xrotation = motion[6 + 3 * bone_index + 1]
        Yrotation = motion[6 + 3 * bone_index + 2]
        theta = [
         Xrotation, Yrotation, Zrotation]
        Rotation = eulerAnglesToRotationMatrix(theta)
    else:
        Rotation = np.identity(3)
    Transformation = np.zeros((4, 4))
    Transformation[0:3, 0:3] = Rotation
    Transformation[3][3] = 1
    offsets_t = np.array(skel[bone]['offsets'])
    Transformation[0][3] = offsets_t[0]
    Transformation[1][3] = offsets_t[1]
    Transformation[2][3] = offsets_t[2]
    return Transformation

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)

def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_quaternion_np(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def quaternion_to_matrix_np(q):
    r, i, j, k = q
    two_s = 2.0 / np.dot(q, q)
    return np.array([
        [1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r)],
        [two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r)],
        [two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)]
    ])

def eulerAnglesToRotationMatrix(theta1):
    theta = np.array(theta1) * (math.pi / 180)
    R_x = np.array([[1, 0, 0],
     [
      0, math.cos(theta[0]), -math.sin(theta[0])],
     [
      0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
     [
      0, 1, 0],
     [
      -math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
     [
      math.sin(theta[2]), math.cos(theta[2]), 0],
     [
      0, 0, 1]])
    R = np.dot(R_z, np.dot(R_x, R_y))
    return R


def eulerAnglesToRotationMatrix_hip(theta1):
    theta = np.array(theta1) * (math.pi / 180)
    R_x = np.array([[1, 0, 0],
     [
      0, math.cos(theta[0]), -math.sin(theta[0])],
     [
      0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
     [
      0, 1, 0],
     [
      -math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
     [
      math.sin(theta[2]), math.cos(theta[2]), 0],
     [
      0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def inside_image(x, y):
    return x >= 0 and x < 512 and y >= 0 and y < 424


def visualize_joints(bone_list, focus):
    m = np.zeros((424, 600, 3))
    m.astype(np.uint8)
    for bone in bone_list:
        p1x = bone[0][0]
        p1y = bone[0][1]
        p1z = bone[0][2] + 400
        p2x = bone[1][0]
        p2y = bone[1][1]
        p2z = bone[1][2] + 400
        p1 = (
         int(p1x * focus / p1z + 300.0), int(-p1y * focus / p1z + 204.0))
        p2 = (int(p2x * focus / p2z + 300.0), int(-p2y * focus / p2z + 204.0))
        if inside_image(p1[0], p1[1]) and inside_image(p2[0], p2[1]):
            cv.line(m, p1, p2, (255, 0, 0), 2)
            cv.circle(m, p1, 2, (0, 255, 255), -1)
            cv.circle(m, p2, 2, (0, 255, 255), -1)

    return m


def visualize_joints2(bone_list, focus):
    m = np.zeros((424, 600, 3))
    m.astype(np.uint8)
    for bone in bone_list:
        p1x = bone[0][0]
        p1y = bone[0][1]
        p1z = bone[0][2] + 400
        p2x = bone[1][0]
        p2y = bone[1][1]
        p2z = bone[1][2] + 400
        p1 = (
         int(p1x * focus / p1z + 300.0), int(-p1y * focus / p1z + 204.0))
        p2 = (int(p2x * focus / p2z + 300.0), int(-p2y * focus / p2z + 204.0))
        if inside_image(p1[0], p1[1]) and inside_image(p2[0], p2[1]):
            cv.line(m, p1, p2, (255, 0, 0), 2)
            cv.circle(m, p1, 2, (0, 255, 255), -1)
            cv.circle(m, p2, 2, (0, 255, 255), -1)

    return m


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-06


def rotationMatrixToEulerAngles(R):
    assert isRotationMatrix(R)
    sy = math.sqrt(R[(0, 0)] * R[(0, 0)] + R[(1, 0)] * R[(1, 0)])
    singular = sy < 1e-06
    if not singular:
        x = math.atan2(R[(2, 1)], R[(2, 2)])
        y = math.atan2(-R[(2, 0)], sy)
        z = math.atan2(R[(1, 0)], R[(0, 0)])
    else:
        x = math.atan2(-R[(1, 2)], R[(1, 1)])
        y = math.atan2(-R[(2, 0)], sy)
        z = 0
    return np.array([x, y, z])


def xyz_to_rotations(skel, position):
    all_rotations = {}
    for bone in skel.keys():
        if bone != 'hip':
            parent = skel[bone]['parent']
            parent_xyz = position[parent]
            bone_xyz = position[bone]
            displacement = bone_xyz - parent_xyz
            displacement_normalized = displacement / np.linalg.norm(displacement)
            orig_offset = np.array(skel[bone]['offsets'])
            orig_offset_normalized = orig_offset / np.linalg.norm(orig_offset)
            rotation = rel_rotation(orig_offset_normalized, np.transpose(displacement_normalized))
            all_rotations[parent] = rotationMatrixToEulerAngles(rotation) * (180 / math.pi)

    return all_rotations


def rel_rotation(a, b):
    v = np.cross(a, b)
    c = np.dot(a, b)
    ssc = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Rotation = np.identity(3) + ssc + np.dot(ssc, ssc) * (1 / (1 + c))
    return Rotation


def xyz_to_rotations_debug(skel, position):
    all_rotations = {}
    all_rotation_matrices = {}
    children_dict = get_child_dict(skel)
    while len(children_dict.keys()) - 1 > len(all_rotation_matrices.keys()):
        for bone in children_dict.keys():
            if bone == None:
                continue
            parent = skel[bone]['parent']
            if bone in all_rotation_matrices.keys():
                continue
            if parent not in all_rotation_matrices.keys() and parent != None:
                continue
            upper = parent
            parent_rot = np.identity(3)
            while upper != None:
                upper_rot = all_rotation_matrices[upper]
                parent_rot = np.dot(upper_rot, parent_rot)
                upper = skel[upper]['parent']

            children = children_dict[bone]
            children_xyz = np.zeros([len(children), 3])
            children_orig = np.zeros([len(children), 3])
            for i in range(len(children)):
                children_xyz[i, :] = np.array(position[children[i]]) - np.array(position[bone])
                children_orig[i, :] = np.array(skel[children[i]]['offsets'])
                children_xyz[i, :] = children_xyz[i, :] * np.linalg.norm(children_orig[i, :]) / np.linalg.norm(children_xyz[i, :])
                assert np.allclose(np.linalg.norm(children_xyz[i, :]), np.linalg.norm(children_orig[i, :]))

            parent_space_children_xyz = np.dot(children_xyz, parent_rot)
            rotation = kabsch(parent_space_children_xyz, children_orig)
            if bone == 'hip':
                all_rotations[bone] = np.array(euler.mat2euler(rotation, 'sxyz')) * (180.0 / math.pi)
            else:
                angles = np.array(euler.mat2euler(rotation, 'syxz')) * (180.0 / math.pi)
                all_rotations[bone] = [
                 angles[1], angles[0], angles[2]]
            all_rotation_matrices[bone] = rotation

    return (all_rotation_matrices, all_rotations)


def kabsch(p, q):
    A = np.dot(np.transpose(p), q)
    V, s, W = np.linalg.svd(A)
    A_2 = np.dot(np.dot(V, np.diag(s)), W)
    assert np.allclose(A, A_2)
    d = np.sign(np.linalg.det(np.dot(np.transpose(W), np.transpose(V))))
    s_2 = np.ones(len(s))
    s_2[len(s) - 1] = d
    rotation = np.dot(np.transpose(W), np.dot(np.diag(s_2), np.transpose(V)))
    assert isRotationMatrix(rotation)
    return np.transpose(rotation)
