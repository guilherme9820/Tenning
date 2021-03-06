import tensorflow as tf
import numpy as np


def rad2deg(rad):
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180


def angle_from_dcm(dcm, unit='rad'):
    """ Extract the rotation angles from a given set of Direction Cosine Matrices (DCMs).

    Args:
        dcms: An array of shape (S, 3, 3), where S is the number of DCMs.
        unit: Defines the unit of the returning angles. Can be either 'deg' or 'rad'.

    Returns:
        A set of rotation angles of shape (S).
    """

    cond = (unit == 'deg') or (unit == 'rad')
    assert cond, f"The 'unit' argument must be either 'deg' or 'rad', but got {unit}"

    eps = 1e-7

    cos = (tf.linalg.trace(dcm) - 1.) * 0.5

    cos = tf.minimum(tf.maximum(cos, -1 + eps), 1 - eps)

    angles = tf.acos(cos)

    if unit == 'deg':
        return rad2deg(angles)

    return angles


def angle_from_quaternion(quaternions, unit='rad'):
    """ Extract the rotation angles from a given set of rotation quaternions.

    Args:
        quaternions: An array of shape (S, 4), where S is the number of quaternions.
                     It considers the quaternions are in the (x,y,z,w) order.
        unit: Defines the unit of the returning angles. Can be either 'deg' or 'rad'.

    Returns:
        A set of rotation angles of shape (S).
    """

    cond = (unit == 'deg') or (unit == 'rad')
    assert cond, f"The 'unit' argument must be either 'deg' or 'rad', but got {unit}"

    im = quaternions[:, :-1]

    sin = tf.linalg.norm(im, axis=-1)
    cos = quaternions[:, -1]

    angles = 2 * tf.atan2(sin, cos)

    if unit == 'deg':
        return rad2deg(angles)

    return angles


def axisangle_from_dcm(dcms):
    """ Extract the rotation axes and the rotation angles from a given set of \
        Direction Cosine Matrices (DCMs).

    Args:
        dcms: An array of shape (S, 3, 3), where S is the number of DCMs.

    Returns:
        rot_axes: A set of rotation axes of shape (S, 3).
        angles: A set of rotation angles of shape (S).
    """

    angle = angle_from_dcm(dcms)

    axes = tf.convert_to_tensor([dcms[:, 1, 2] - dcms[:, 2, 1],
                                 dcms[:, 2, 0] - dcms[:, 0, 2],
                                 dcms[:, 0, 1] - dcms[:, 1, 0]])
    axes = tf.transpose(axes)

    constant = tf.math.reciprocal(2*tf.sin(angle))[:, tf.newaxis]

    rot_axes = constant * axes

    return rot_axes, angle


def dcm_from_axisangle(axes, angles):
    """ Builds a Direction Cosine Matrix from the given rotation axes and angles.

    Args:
        axes: An array of shape (S, 3), where S is the number of rotation axes.
        angles: An array of shape (S), where S is the number of rotation angles.

    Returns:
        A Direction Cosine Matrix of shape (S, 3, 3).
    """

    axes = tf.cast(axes, 'float32')
    angles = tf.cast(angles, 'float32')

    if len(tf.shape(axes)) < 2:
        axes = axes[tf.newaxis, :]

    cos = tf.math.cos(angles)
    sin = tf.math.sin(angles)
    e_1, e_2, e_3 = axes[:, 0], axes[:, 1], axes[:, 2]

    dcm = tf.convert_to_tensor([[cos + (1-cos)*e_1**2, e_1*e_2*(1-cos) + e_3*sin, e_1*e_3*(1-cos) - e_2*sin],
                                [e_1*e_2*(1-cos) - e_3*sin, cos + (1-cos)*e_2**2, e_2*e_3*(1-cos) + e_1*sin],
                                [e_1*e_3*(1-cos) + e_2*sin, e_2*e_3*(1-cos) - e_1*sin, cos + (1-cos)*e_3**2]])

    return tf.transpose(dcm, [2, 0, 1])


def gen_random_dcm(num_samples, min_angle=-np.pi, max_angle=np.pi, unit='rad'):
    """ Generates random rotation matrices where the axes and the angles \
        are drawn from a uniform distribution. The angles can be specified \
        in radians or degrees.

    Args:
        num_samples: Number of matrices to be generated.
        min_angle: The lowest possible angle that will be drawn (default: -pi rad).
        max_angle: The highest possible angle that will be drawn (default: pi rad).
        unit: The angular unit. It could be either 'deg' or 'rad'.

    Returns:
        A set of Direction Cosine Matrices of shape (S, 3, 3).
    """

    cond = (unit == 'deg') or (unit == 'rad')
    assert cond, f"The 'unit' argument must be either 'deg' or 'rad', but got {unit}"

    if unit == 'deg':
        min_angle, max_angle = np.radians(min_angle), np.radians(max_angle)

    axes = tf.random.uniform([num_samples, 3])
    axes = tf.linalg.normalize(axes, axis=1)[0]

    angles = tf.random.uniform([num_samples], min_angle, max_angle)

    return dcm_from_axisangle(axes, angles)


def gen_rot_quaternion(num_samples, min_angle=-np.pi, max_angle=np.pi, unit='rad'):
    """ Generates random rotation quaternions where the axes and the angles \
        are drawn from a uniform distribution. The angles can be specified \
        in radians or degrees.

    Args:
        num_samples: Number of quaternions to be generated.
        min_angle: The lowest possible angle that will be drawn (default: -pi rad).
        max_angle: The highest possible angle that will be drawn (default: pi rad).
        unit: The angular unit. It could be either 'deg' or 'rad'.

    Returns:
        A set of rotation quaternions of shape (S, 4).
    """

    cond = (unit == 'deg') or (unit == 'rad')
    assert cond, f"The 'unit' argument must be either 'deg' or 'rad', but got {unit}"

    if unit == 'deg':
        min_angle, max_angle = np.radians(min_angle), np.radians(max_angle)

    theta = tf.random.uniform([num_samples, 1], min_angle, max_angle)/2
    sin = tf.math.sin(theta)
    axis = tf.random.uniform([num_samples, 3])
    axis = tf.linalg.normalize(axis, axis=1)[0]  # batch*3
    scalar = tf.math.cos(theta)
    vector = axis * sin

    return tf.concat([vector, scalar], axis=1)


def rotation_matrix_from_ortho6d(ortho6d):
    """ Maps an array r ??? R^6 from a representation space to the Special Orthogonal \
        Group SO(3) [Zhou2019]. This implementation is an adaptation from the original \
        source code found at https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py

    Args:
        ortho6d: An array of shape (S, 6), where S is the number of
                 samples.

    Returns:
        The array represented in SO(3).

    References:
        - [Zhou2019] Zhou, Yi, et al. "On the continuity of rotation representations
                     in neural networks." Proceedingsof the IEEE/CVF Conference on
                     Computer Vision and Pattern Recognition. 2019.
    """

    x_raw = ortho6d[..., :3]
    y_raw = ortho6d[..., 3:]

    x = tf.linalg.normalize(x_raw, axis=-1)[0]

    z = tf.linalg.cross(x, y_raw)  # (batch, 3)

    z = tf.linalg.normalize(z, axis=-1)[0]  # (batch, 3)
    y = tf.linalg.cross(z, x)  # (batch, 3)

    matrix = tf.stack([x, y, z], axis=-1)

    return matrix


def dcm_from_quaternion(quaternion):

    quat = tf.cast(quaternion, 'float32')

    quat /= tf.linalg.norm(quat, axis=1, keepdims=True)

    qw = quat[:, 3][:, tf.newaxis]
    qx = quat[:, 0][:, tf.newaxis]
    qy = quat[:, 1][:, tf.newaxis]
    qz = quat[:, 2][:, tf.newaxis]

    # Unit quaternion rotation matrices computatation
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw

    row0 = tf.concat([1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw], axis=1)  # batch*3
    row1 = tf.concat([2*xy + 2*zw,  1-2*xx-2*zz, 2*yz-2*xw], axis=1)  # batch*3
    row2 = tf.concat([2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy], axis=1)  # batch*3

    matrix = tf.concat([row0[:, tf.newaxis, :],
                        row1[:, tf.newaxis, :],
                        row2[:, tf.newaxis, :]], axis=1)

    return matrix


def quaternion_from_dcm(dcm):

    if len(tf.shape(dcm)) < 3:
        dcm = dcm[tf.newaxis, ...]

    A11 = dcm[:, 0, 0]
    A12 = dcm[:, 0, 1]
    A13 = dcm[:, 0, 2]
    A21 = dcm[:, 1, 0]
    A22 = dcm[:, 1, 1]
    A23 = dcm[:, 1, 2]
    A31 = dcm[:, 2, 0]
    A32 = dcm[:, 2, 1]
    A33 = dcm[:, 2, 2]

    x1 = tf.transpose([1 + 2*A11 - tf.linalg.trace(dcm),
                       A12 + A21,
                       A13 + A31,
                       A23 - A32])
    x1 /= tf.linalg.norm(x1, axis=1, keepdims=True)

    x2 = tf.transpose([A21 + A12,
                       1 + 2*A22 - tf.linalg.trace(dcm),
                       A23 + A32,
                       A31 - A13])
    x2 /= tf.linalg.norm(x2, axis=1, keepdims=True)

    x3 = tf.transpose([A31 + A13,
                       A32 + A23,
                       1 + 2*A33 - tf.linalg.trace(dcm),
                       A12 - A21])
    x3 /= tf.linalg.norm(x3, axis=1, keepdims=True)

    x4 = tf.transpose([A23 - A32,
                       A31 - A13,
                       A12 - A21,
                       1 + tf.linalg.trace(dcm)])
    x4 /= tf.linalg.norm(x4, axis=1, keepdims=True)

    # Each row vector is a rotation quaternion
    quaternions = tf.stack([x1, x2, x3, x4], axis=1)

    squared_qi = tf.linalg.diag_part(quaternions)**2
    indices = tf.argmax(squared_qi, axis=1)

    def get_quat(elements):
        quat = elements[0]
        idx = elements[1]
        return tf.slice(quat, [idx, 0], [1, 4])[0]

    quaternions = tf.map_fn(get_quat, [quaternions, indices], dtype=tf.float32)

    return quaternion_inverse(quaternions)


def gen_boresight_vector(num_samples, observations, weight=2):

    vec = np.random.uniform(size=(num_samples, observations, 3))

    sample_axis = np.arange(num_samples)[:, np.newaxis]
    sample_axis = np.tile(sample_axis, [1, observations])
    sample_axis = np.reshape(sample_axis, [-1])

    obs_axis = np.arange(observations)[np.newaxis, :]
    obs_axis = np.tile(obs_axis, [num_samples, 1])
    obs_axis = np.reshape(obs_axis, [-1])

    max_indices = np.argmax(vec, axis=-1).reshape([-1])
    max_indices = np.array([sample_axis, obs_axis, max_indices])
    max_indices = max_indices.T

    bore_axis = np.random.uniform(0, 3, [num_samples * observations]).astype('int32')
    bore_axis = np.array([sample_axis, obs_axis, bore_axis])
    bore_axis = bore_axis.T

    vec[max_indices[:, 0], max_indices[:, 1], max_indices[:, 2]] = vec[bore_axis[:, 0], bore_axis[:, 1], bore_axis[:, 2]]

    vec[bore_axis[:, 0], bore_axis[:, 1], bore_axis[:, 2]] = weight

    vec /= weight

    vec = tf.linalg.normalize(vec, axis=-1)[0]

    return tf.cast(vec, "float32")


def svdo(matrix, mode='plus'):
    """ Performs symmetric orthogonalization as detailed in equations 1 and 2 \
        from paper An Analysis of SVD for Deep Rotation Estimation \
        (https://proceedings.neurips.cc/paper/2020/file/fec3392b0dc073244d38eba1feb8e6b7-Paper.pdf)

    Args:
        matrix (Union[tf.Tensor, np.ndarray]): A matrix of shape [N, 3, 3] where N is the batch size.
        mode (str, optional): If mode == 'plus' then the special orthogonalization is performed (maps to SO(n)),
                              an orientation-preserving orthogonalization is performed otherwise. Defaults to 'plus'.

    Returns:
        Returns an orthogonal matrix of shape [N, 3, 3].
    """

    batch_size = tf.shape(matrix)[0]

    s, u, v = tf.linalg.svd(matrix)

    if mode == 'plus':

        temp1 = tf.linalg.det(tf.matmul(u, v, transpose_b=True))  # (batch_size, )
        temp1 = temp1[:, tf.newaxis]  # (batch_size, 1)

        temp = tf.concat([tf.ones([batch_size, 2]), temp1], axis=1)   # (batch_size, 3)
        temp = temp[:, tf.newaxis, :]  # (batch_size, 1, 3)

        return tf.matmul(u * temp, v, transpose_b=True)

    return tf.matmul(u, v, transpose_b=True)


def quaternion_multiplication(quat1, quat2):
    """ Performs the multiplication between two quaternions.
        The input quaternions are considered to be ordered as follows:
            quaternion = [x, y, z, w]
    Args:
        quat1, quat2: A tensor of shape [..., 4] whose inner-most dimension has 4 elements.

    Returns:
        The Hamilton product between a set of quaternions.
    """

    quat1 = tf.convert_to_tensor(quat1, dtype='float32')
    quat2 = tf.convert_to_tensor(quat2, dtype='float32')

    original_shape = tf.shape(quat1)

    quat1 = tf.reshape(quat1, [-1, 4])
    quat2 = tf.reshape(quat2, [-1, 4])

    i_comp = quat1[:, 0]*quat2[:, 3] + quat1[:, 3]*quat2[:, 0] + quat1[:, 1]*quat2[:, 2] - quat1[:, 2]*quat2[:, 1]
    j_comp = quat1[:, 3]*quat2[:, 1] - quat1[:, 0]*quat2[:, 2] + quat1[:, 1]*quat2[:, 3] + quat1[:, 2]*quat2[:, 0]
    z_comp = quat1[:, 3]*quat2[:, 2] + quat1[:, 0]*quat2[:, 1] - quat1[:, 1]*quat2[:, 0] + quat1[:, 2]*quat2[:, 3]
    real = quat1[:, 3]*quat2[:, 3] - quat1[:, 0]*quat2[:, 0] - quat1[:, 1]*quat2[:, 1] - quat1[:, 2]*quat2[:, 2]

    new_shape = tf.concat([original_shape[:-1], [1]], axis=0)

    i_comp = tf.reshape(i_comp, new_shape)
    j_comp = tf.reshape(j_comp, new_shape)
    z_comp = tf.reshape(z_comp, new_shape)
    real = tf.reshape(real, new_shape)

    result = tf.concat([i_comp, j_comp, z_comp, real], axis=-1)

    return result


def quaternion_inverse(quaternion):

    if len(tf.shape(quaternion)) < 2:
        quaternion = quaternion[tf.newaxis, :]

    samples = tf.shape(quaternion)[0]
    idx = tf.range(samples)[:, tf.newaxis]  # idx = [[0], [1], ..., [batch_size-1]]
    # idx = [[0, 3], [1, 3], ..., [batch_size-1, 3]]
    idx = tf.concat([idx, 3*tf.ones([samples, 1], dtype='int32')], axis=-1)
    return tf.tensor_scatter_nd_update(-quaternion, idx, quaternion[:, 3])


def rotate_vector(rotations, vectors, representation='dcm'):

    rotations = tf.convert_to_tensor(rotations, dtype='float32')
    vectors = tf.convert_to_tensor(vectors, dtype='float32')

    if len(tf.shape(rotations)) < 2:
        rotations = rotations[tf.newaxis, ...]

    if len(tf.shape(vectors)) < 2:
        vectors = vectors[tf.newaxis, ...]

    if representation == 'quaternion':

        original_shape = tf.shape(vectors)
        rotation_shape = tf.shape(rotations)

        vectors = tf.reshape(vectors, [-1, 3])
        # We need to transform the point clound from R^3 to R^4 by adding zeros
        zeros = tf.zeros([tf.shape(vectors)[0], 1])
        pcs = tf.concat([vectors, zeros], axis=-1)  # (..., 4)

        quat_inv = quaternion_inverse(rotations)

        diff = tf.reduce_prod(original_shape[:-1]) / tf.reduce_prod(rotation_shape[:-1])
        quat = rotations[:, tf.newaxis, :]
        quat = tf.tile(quat, [1, diff, 1])  # (batch, num_points, 4)
        quat_inv = quat_inv[:, tf.newaxis, :]
        quat_inv = tf.tile(quat_inv, [1, diff, 1])  # (batch, num_points, 4)

        rotated = quaternion_multiplication(quaternion_multiplication(quat, pcs), quat_inv)
        rotated = rotated[..., :-1]  # removes the last column since it is filled with zeros

    else:
        if len(tf.shape(vectors)) < 3:
            vectors = vectors[..., tf.newaxis]

        rotated = tf.matmul(rotations, vectors, transpose_b=True)
        rotated = tf.transpose(rotated, [0, 2, 1])

    return tf.squeeze(rotated)


def average_rotation_svd(dcms):
    """ Calculates the projected arithmetic mean of N rotation matrices [Curtis1993]. \
        The mean is calculated w.r.t. the first dimension of the input array.

    Args:
        dcms: An array of shape (N, S, 3, 3) or (S, 3, 3), where N are the number
              of rotation estimations per sample and S is the number of samples.
              Each sample of the array is a rotation matrix.

    Returns:
        The projected mean of the rotation matrices.

    References:
        - [Curtis1993] Curtis, W. Dan, Adam L. Janin, and Karel Zikan. "A note on averaging
            rotations." Proceedings of IEEE Virtual Reality Annual International Symposium. IEEE, 1993.
    """

    dcm_mean = tf.reduce_mean(dcms, axis=0)

    _, u, v = tf.linalg.svd(dcm_mean)

    return tf.matmul(u, v, transpose_b=True)


def average_rotation_dec(dcms):
    """ Calculates the projected arithmetic mean of N rotation matrices. \
        The mean is calculated w.r.t. the first dimension of the input array. This \
        implementation follows the equation (3.7) of paper [Moakher2002].

    Args:
        dcms: An array of shape (N, S, 3, 3) or (S, 3, 3), where N are the number
              of rotation estimations per sample ans S is the number of samples.
              Each sample of the array is a rotation matrix.

    Returns:
        The projected mean of the rotation matrices.

    References:
        - [Moakher2002] Moakher, Maher. "Means and averaging in the group of rotations."
            SIAM journal on matrix analysis and applications 24.1 (2002): 1-16.
    """

    r_mean = tf.reduce_mean(dcms, axis=0)

    if len(tf.shape(r_mean)) < 3:
        r_mean = r_mean[tf.newaxis, ...]

    r_dets = tf.linalg.det(r_mean)

    s_vector = tf.where(r_dets >= 0, 1., -1.)

    m_matrix = tf.matmul(r_mean, r_mean, transpose_a=True)

    vals, vecs = tf.linalg.eigh(m_matrix)

    # reciprocal of square root
    sqrts_inv = tf.math.rsqrt(vals)

    diag = tf.convert_to_tensor([sqrts_inv[:, 0],
                                 sqrts_inv[:, 1],
                                 s_vector * sqrts_inv[:, 2]])
    diag = tf.linalg.diag(tf.transpose(diag))

    temp1 = tf.matmul(r_mean, vecs)
    temp2 = tf.matmul(diag, vecs, transpose_b=True)

    projected_mean = tf.matmul(temp1, temp2)

    return projected_mean
