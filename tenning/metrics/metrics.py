from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras import backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.metrics import MeanMetricWrapper
import tensorflow as tf
import numpy as np
from tenning.losses import geodesic_distance
from tenning.losses import euclidean_distance
from tenning.losses import wahba_loss
from tenning.losses import quaternion_distance


class F1Score(Metric):

    def __init__(self,
                 threshold=None,
                 name='f1_score',
                 dtype=None):
        """Creates a `F1Score` instance.
        Args:
          threshold: (Optional) A float value within [0, 1] range. A threshold is compared with prediction
            values to determine the truth value of predictions (i.e., above the
            threshold is `true`, below is `false`). If threshold is None, the
            default is used (threshold=0.5).
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super().__init__(name=name, dtype=dtype)
        self.init_thresholds = threshold

        self.threshold = metrics_utils.parse_init_thresholds(
            threshold, default_threshold=0.5)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=[1],
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=[1],
            initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=[1],
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive, false positive and false negative statistics.
        Args:
          y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted values. Each element must be in the range `[0, 1]`.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        Returns:
          Update op.
        """
        shape = tf.shape(y_pred)

        if shape[-1] != 1:
            y_pred = math_ops.cast(math_ops.argmax(y_pred, axis=1), 'float32')
            y_true = math_ops.cast(math_ops.argmax(y_true, axis=1), 'float32')
        # else:
        #     y_pred = math_ops.cast(array_ops.squeeze(y_pred), 'float32')
        #     y_true = math_ops.cast(array_ops.squeeze(y_true), 'float32')

        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.threshold)

    def result(self):
        result = math_ops.div_no_nan(self.true_positives,
                                     self.true_positives + 0.5 * (self.false_positives + self.false_negatives))
        return result[0] if len(self.threshold) == 1 else result

    def reset_states(self):
        num_thresholds = len(to_list(self.threshold))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'threshold': self.init_thresholds
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdjustedRSquare(Metric):

    def __init__(self,
                 name='adjusted_r_square',
                 dtype=None):
        super().__init__(name=name, dtype=dtype)

        self.adj_r = self.add_weight(
            'adjusted_r_square',
            shape=[],
            dtype=tf.float32,
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):

        SS_res = tf.reduce_sum(tf.math.square(y_true - y_pred))
        SS_tot = tf.reduce_sum(tf.math.square(y_true - tf.reduce_mean(y_true)))
        r2 = (1 - SS_res / (SS_tot + 1e-07))

        shape = tf.shape(y_pred)

        adj_r = 1 - ((1 - r2) * (shape[0] - 1)) / (shape[0] - shape[1] - 1)

        self.adj_r.assign(adj_r)

    def result(self):
        return self.adj_r.read_value()

    def reset_states(self):
        self.adj_r.assign(0.)

    def get_config(self):
        base_config = super().get_config()
        return base_config


class MeanEuclideanDistance(MeanMetricWrapper):

    def __init__(self, name='euclidean_distance', dtype=None):
        super().__init__(euclidean_distance, name, dtype=dtype)


class GeodesicError(MeanMetricWrapper):

    def __init__(self, name='geodesic_error', dtype=None):
        super().__init__(geodesic_distance, name, dtype=dtype)


class WahbaError(MeanMetricWrapper):

    def __init__(self, name='wahba_error', dtype=None):
        super().__init__(wahba_loss, name, dtype=dtype)


class QuaternionDistance(MeanMetricWrapper):

    def __init__(self, name='quaternion_error', dtype=None):
        super().__init__(quaternion_distance, name, dtype=dtype)
