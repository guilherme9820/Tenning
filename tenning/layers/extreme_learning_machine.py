from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np


class OSELM(Layer):

    def __init__(self,
                 n_hidden_nodes,
                 n_output_nodes,
                 batch_size,
                 reg_weight=5,
                 trainable=True,
                 name='oselm',
                 **kwargs):
        super().__init__(name=name, trainable=trainable, **kwargs)

        self.n_hidden_nodes = n_hidden_nodes
        self.n_output_nodes = n_output_nodes
        self.reg_weight = reg_weight
        self.batch_size = batch_size

        self._p = self.add_weight(shape=[n_hidden_nodes, n_hidden_nodes], initializer=tf.zeros_initializer(), trainable=False, name=self.name + '/p')

        self._beta = self.add_weight(shape=[n_hidden_nodes, n_output_nodes], initializer=tf.zeros_initializer(), trainable=False, name=self.name + '/beta')

    def build(self, input_shape):

        # When Tensorflow is building the network graph _target must have a valid shape
        self._target = tf.zeros([self.batch_size, self.n_output_nodes])

        self.is_finished_init_train = self.add_weight(shape=[], initializer=tf.zeros_initializer(), trainable=False, name=self.name + '/cond', dtype=tf.bool)

    def call(self, H, training=True):

        if training:

            HT = tf.transpose(H)
            if not self.is_finished_init_train:
                # Initial training
                num_feats = tf.shape(H)[1]
                Id = tf.linalg.eye(num_feats)
                HTH = tf.matmul(HT, H)
                p = tf.linalg.inv(HTH + self.reg_weight * Id)
                p = self._p.assign(p)
                pHT = tf.matmul(p, HT)
                pHTt = tf.matmul(pHT, self._target)
                self._beta.assign(pHTt)

                self.is_finished_init_train.assign(tf.constant(True))
            else:
                # Sequential training
                Id = tf.linalg.eye(self.batch_size)
                Hp = tf.matmul(H, self._p)
                HpHT = tf.matmul(Hp, HT)
                temp = tf.numpy_function(np.linalg.pinv, [Id + HpHT], tf.float32, name="np.linalg.pinv")
                pHT = tf.matmul(self._p, HT)
                p = self._p.assign_sub(tf.matmul(tf.matmul(pHT, temp), Hp))
                pHT = tf.matmul(p, HT)
                Hbeta = tf.matmul(H, self._beta)
                self._beta.assign_add(tf.matmul(pHT, self._target - Hbeta))

        return tf.matmul(H, self._beta)

    def _set_target(self, value):
        self._target = value

    target = property(fset=_set_target)

    @property
    def beta(self):
        return self._beta.read_value()

    def get_config(self):

        config = super().get_config()

        config.update({'n_hidden_nodes': self.n_hidden_nodes,
                       'n_output_nodes': self.n_output_nodes,
                       'reg_weight': self.reg_weight,
                       'batch_size': self.batch_size,
                       'trainable': self.trainable,
                       'name': self.name})

        return config


# @custom_export('models.WOSELM')
# class WOSELM(OSELM):
#     """docstring for WOSELM"""

#     def __init__(self,
#                  n_hidden_nodes=None,
#                  n_output_nodes=None,
#                  reg_weight=5,
#                  trainable=True,
#                  name='woselm',
#                  **kwargs):
#         super().__init__(name=name,
#                          trainable=trainable,
#                          n_hidden_nodes=n_hidden_nodes,
#                          n_output_nodes=n_output_nodes,
#                          reg_weight=reg_weight,
#                          **kwargs)

#     def _class_weights(self):

#         true_positive = tf.math.count_nonzero(self._target, dtype=tf.int32)
#         total = tf.size(self._target)
#         true_negative = total - true_positive

#         w_neg = 1
#         w_pos = true_negative / true_positive

#         class_weight = tf.squeeze(tf.where(self._target == 1, w_pos, w_neg))

#         return tf.cast(class_weight, tf.float32)

#     def _train_graph(self, H, training=True):

#         if training:

#             class_weight = self._class_weights()
#             class_weight = tf.linalg.diag(class_weight)

#             HT = tf.transpose(H)
#             if not self.is_finished_init_train:
#                 # Initial training
#                 num_feats = tf.shape(H)[1]
#                 Id = tf.linalg.eye(num_feats)
#                 HTW = tf.matmul(HT, class_weight)
#                 HTH = tf.matmul(HTW, H)
#                 p = tf.linalg.inv(HTH + self.reg_weight * Id)
#                 p = self._p.assign(p)
#                 pHT = tf.matmul(p, HT)
#                 pHTt = tf.matmul(pHT, tf.matmul(class_weight, self._target))
#                 self._beta.assign(pHTt)

#                 self.is_finished_init_train = True
#             else:
#                 # Sequential training
#                 inv_class_weights = np.linalg.inv(class_weight)
#                 Hp = tf.matmul(H, self._p)
#                 HpHT = tf.matmul(Hp, HT)
#                 temp = np.linalg.pinv(inv_class_weights + HpHT)
#                 pHT = tf.matmul(self._p, HT)
#                 p = self._p.assign_sub(tf.matmul(tf.matmul(pHT, temp), Hp))
#                 pHT = tf.matmul(p, HT)
#                 pHTW = tf.matmul(pHT, class_weight)
#                 Hbeta = tf.matmul(H, self._beta)
#                 self._beta.assign_add(tf.matmul(pHTW, self._target - Hbeta))

#         return self.infer(H)
