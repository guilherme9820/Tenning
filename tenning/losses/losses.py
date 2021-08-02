from tensorflow.keras.losses import kullback_leibler_divergence
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np
from typing import Union


class TverskyLoss(LossFunctionWrapper):

    def __init__(self, beta, reduction=tf.keras.losses.Reduction.AUTO, name='tversky'):
        super().__init__(tversky,
                         beta=beta,
                         name=name,
                         reduction=reduction)


class DiceLoss(LossFunctionWrapper):

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='dice'):
        super().__init__(tversky,
                         beta=0.5,
                         name=name,
                         reduction=reduction)


class LearnableClusteringObjective(LossFunctionWrapper):

    def __init__(self, hinge_margin=2, reduction=tf.keras.losses.Reduction.AUTO, name='lco'):
        super().__init__(mcl,
                         hinge_margin=hinge_margin,
                         name=name,
                         reduction=reduction)


class MCL(LossFunctionWrapper):

    def __init__(self, label_smoothing=0., from_logits=False, reduction=tf.keras.losses.Reduction.AUTO, name='mcl'):
        super().__init__(mcl,
                         name=name,
                         reduction=reduction,
                         from_logits=from_logits,
                         label_smoothing=label_smoothing)


class CapsuleMarginLoss(LossFunctionWrapper):

    def __init__(self, m_plus=0.9, m_minus=0.1, weight=0.5, reduction=tf.keras.losses.Reduction.AUTO, name='capsule_margin_loss'):
        super().__init__(capsule_margin_loss,
                         m_plus=m_plus,
                         m_minus=m_minus,
                         weight=weight,
                         name=name,
                         reduction=reduction)


class GeodesicLoss(LossFunctionWrapper):

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='geodesic_loss'):
        super().__init__(geodesic_distance,
                         name=name,
                         reduction=reduction)


class EuclideanLoss(LossFunctionWrapper):

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='euclidean_loss'):
        super().__init__(euclidean_distance,
                         name=name,
                         reduction=reduction)


class WahbaLoss(LossFunctionWrapper):

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='wahba_loss'):
        super().__init__(wahba_loss,
                         name=name,
                         reduction=reduction)


class CosineLoss(LossFunctionWrapper):

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='cosine_loss'):
        super().__init__(cosine_loss,
                         name=name,
                         reduction=reduction)


class QuatChordalSquaredLoss(LossFunctionWrapper):

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='quat_chordal_squared_loss'):
        super().__init__(quat_chordal_squared_loss,
                         name=name,
                         reduction=reduction)


class MSSSIML1(LossFunctionWrapper):

    def __init__(self,
                 alpha=0.84,
                 max_val=1.0,
                 power_factors=[0.5, 1., 2., 4., 8.],
                 filter_size=11,
                 filter_sigma=1.5,
                 k1=0.01,
                 k2=0.03,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='msssim_l1'):

        super().__init__(msssim_l1,
                         alpha=alpha,
                         max_val=max_val,
                         power_factors=power_factors,
                         filter_size=filter_size,
                         filter_sigma=filter_sigma,
                         k1=k1,
                         k2=k2,
                         name=name,
                         reduction=reduction)


def capsule_margin_loss(y_true,
                        y_pred,
                        m_plus,
                        m_minus,
                        weight):

    present_error_raw = tf.square(tf.maximum(0., m_plus - y_pred))

    present_error = tf.reshape(present_error_raw, shape=(-1, 10))

    absent_error_raw = tf.square(tf.maximum(0., y_pred - m_minus))

    absent_error = tf.reshape(absent_error_raw, shape=(-1, 10))

    loss = tf.add(y_true * present_error, weight * (1.0 - y_true) * absent_error)

    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))


def mcl(*,
        y_pred,
        from_logits=False,
        label_smoothing=0):

    pairwise_sim = tf.linalg.matmul(y_pred, y_pred, transpose_b=True)

    pairwise_sim = tf.cast(tf.reshape(pairwise_sim, [-1, 1]), tf.float32)

    binarized_sim = tf.where(pairwise_sim >= 0.5, 1., 0.)

    return tf.keras.losses.binary_crossentropy(binarized_sim, pairwise_sim, from_logits=from_logits, label_smoothing=label_smoothing)


def tversky(y_true,
            y_pred,
            beta):

    y_pred = tf.cast(tf.argmax(y_pred, axis=1), 'float32')
    y_true = tf.cast(tf.argmax(y_true, axis=1), 'float32')

    confusion_matrix = tf.reshape(tf.math.confusion_matrix(y_true, y_pred), [-1])

    FP = tf.cast(tf.slice(confusion_matrix, [1], [1]), 'float32')
    FN = tf.cast(tf.slice(confusion_matrix, [2], [1]), 'float32')
    TP = tf.cast(tf.slice(confusion_matrix, [3], [1]), 'float32')

    epsilon = 1e-7

    tversky = (TP + epsilon) / (TP + beta * FP + (1. - beta) * FN + epsilon)

    return 1. - tversky


def learnable_clustering_objective(y_true,
                                   y_pred,
                                   hinge_margin):

    features1, features2 = tf.split(y_pred, 2, axis=1)

    pq = kullback_leibler_divergence(features1, features2)
    qp = kullback_leibler_divergence(features2, features1)
    similarity = tf.where(y_true >= 0.5, 1., 0.)

    total_loss = hinge_embedding(similarity, pq, margin=hinge_margin) + hinge_embedding(similarity, qp, margin=hinge_margin)

    return tf.reduce_sum(total_loss)


def hinge_embedding(y_true, y_pred, margin=1.):
    """Computes the hinge loss """

    def _maybe_convert_labels(y_true):
        """Converts binary labels into -1/1."""
        are_zeros = math_ops.equal(y_true, 0)
        are_ones = math_ops.equal(y_true, 1)
        is_binary = math_ops.reduce_all(math_ops.logical_or(are_zeros, are_ones))

        def _convert_binary_labels():
            # Convert the binary labels to -1 or 1.
            return 2. * y_true - 1.

        updated_y_true = smart_cond.smart_cond(is_binary,
                                               _convert_binary_labels, lambda: y_true)
        return updated_y_true

    y_true = _maybe_convert_labels(y_true)

    loss = tf.where(y_true == 1., y_pred, math_ops.maximum(margin - y_pred, 0.))
    return K.mean(loss, axis=-1)


def get_symmetric_indices(array_size, raw_indices):

    # (size / 2) - 1
    pivot = int((array_size >> 1) - 1)

    # Converts raw indices (float) to symmetric indices (integer) within range [-pivot, pivot + 1]
    symm_indices = tf.where(tf.math.sign(raw_indices) >= 0, raw_indices + 1, raw_indices)
    symm_indices = tf.cast(symm_indices, tf.int64)

    # Shifts the symmetric indices from range [-pivot, pivot + 1] to range [0, 2*pivot + 1]
    shifted_indices = symm_indices + pivot

    return shifted_indices


def msssim_l1(y_true,
              y_pred,
              alpha,
              max_val,
              power_factors,
              filter_size,
              filter_sigma,
              k1,
              k2):

    msssim = tf.image.ssim_multiscale(y_true,
                                      y_pred,
                                      max_val,
                                      power_factors=power_factors,
                                      filter_size=filter_size,
                                      filter_sigma=filter_sigma,
                                      k1=k1,
                                      k2=k2)

    l1 = tf.math.abs(y_true - y_pred)

    return alpha * tf.reduce_sum(msssim) + (1. - alpha) * tf.reduce_sum(l1)


def geodesic_distance(y_true, y_pred):
    """ Computes the geodesic distance between two rotation matrices.
        This is a natural Riemannian metric on the compact Lie group
        SO(3).

    Args:
        y_true: Groundtruth rotation matrices. Must be an array with shape (batch, 3, 3).
        y_pred: Predicted rotation matrices. Must be an array with shape (batch, 3, 3).

    Returns:
        Cost corresponding to the difference between the true and
        predicted rotation matrices.
    """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # To avoid NaN values when arccos is evaluated for -1 and 1
    eps = 1e-7

    mul = tf.matmul(y_true, y_pred, transpose_b=True)

    cos = (tf.linalg.trace(mul) - 1) * 0.5

    cos = tf.minimum(tf.maximum(cos, -1 + eps), 1 - eps)

    theta = tf.math.acos(cos)

    return theta


def euclidean_distance(y_true, y_pred):

    return tf.reduce_mean(tf.norm(y_true - y_pred, axis=-1), axis=-1)


def wahba_loss(y_true: Union[np.ndarray, tf.Tensor],
               y_pred: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
    """ Loss function proposed by Grace Wahba in [Wahba1965].

    Args:
        y_true: True body vectors. Must be an array with shape (batch, observations, 3).
        y_pred: Predicted body vectors. Must be an array with shape (batch, observations, 3).

    Returns:
        Cost corresponding to the difference between the true and predicted body vectors.

    References:
        - [Wahba1965] Wahba, Grace. "A least squares estimate of satellite attitude." SIAM review 7.3 (1965): 409-409.
    """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    norm = tf.norm(y_true - y_pred, axis=-1)

    obs = tf.shape(norm)[-1]
    weights = tf.ones_like(norm) / obs

    error = weights * norm

    return 0.5 * tf.reduce_mean(error, axis=-1)


def cosine_loss(y_true, y_pred):
    # For normalized `p_t_given_x` and `t`, dot-product (batched)
    # outputs a cosine value, i.e. between -1 (worst) and 1 (best)
    cos_angles = tf.reduce_sum(tf.multiply(y_true, y_pred), 1, keepdims=True)

    # Rescale to a cost going from 2 (worst) to 0 (best) each, then take mean.
    return tf.reduce_mean(1 - cos_angles)


def quat_chordal_squared_loss(q_true, q_pred):

    distance = quaternion_distance(q_true, q_pred)

    losses = 2*(distance**2)*(4. - distance**2)

    return losses


def quaternion_distance(q_1, q_2):
    minus = tf.linalg.norm(q_1 - q_2, axis=-1)
    plus = tf.linalg.norm(q_1 + q_2, axis=-1)

    return tf.math.minimum(minus, plus)
