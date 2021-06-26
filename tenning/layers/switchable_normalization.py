import tensorflow as tf
from tensorflow.keras.layers import Layer


class SwitchableNormalization(Layer):

    def __init__(self,
                 momentum=0.9,
                 trainable=True,
                 **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.momentum = momentum

    def build(self, input_shape):
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=[input_shape[-1], ], initializer=tf.initializers.ones, trainable=self.trainable)
        self.beta = self.add_weight(name='beta', shape=[input_shape[-1], ], initializer=tf.initializers.zeros, trainable=self.trainable)
        self.mean_weights = self.add_weight(name='mean_weights', shape=[3], initializer=tf.initializers.ones, trainable=self.trainable)
        self.var_weights = self.add_weight(name='var_weights', shape=[3], initializer=tf.initializers.ones, trainable=self.trainable)

        if len(input_shape) == 4:
            # Conv2D
            self.moving_mean = self.add_weight(name='moving_mean', shape=[1, 1, 1, input_shape[-1]], initializer=tf.initializers.zeros, trainable=False)
            self.moving_var = self.add_weight(name='moving_var', shape=[1, 1, 1, input_shape[-1]], initializer=tf.initializers.zeros, trainable=False)
        else:
            # Conv1D
            self.moving_mean = self.add_weight(name='moving_mean', shape=[1, 1, input_shape[-1]], initializer=tf.initializers.zeros, trainable=False)
            self.moving_var = self.add_weight(name='moving_var', shape=[1, 1, input_shape[-1]], initializer=tf.initializers.zeros, trainable=False)

    def assign_moving_average(self, variable, value):
        delta = variable * self.momentum + value * (1 - self.momentum)
        return variable.assign(delta)

    def call(self, input_tensor, training=True):

        original_shape = tf.shape(input_tensor)

        # (batch, H*W, channels)
        input_tensor = tf.reshape(input_tensor, [original_shape[0], -1, original_shape[-1]])

        instance_mean = tf.reduce_mean(input_tensor, [1], keepdims=True)
        instance_var = tf.reduce_mean(tf.math.squared_difference(input_tensor, tf.stop_gradient(instance_mean)), [1], keepdims=True)

        layer_mean = tf.reduce_mean(instance_mean, [-1], keepdims=True)
        layer_var = tf.reduce_mean(instance_var + instance_mean**2, [-1], keepdims=True) - layer_mean**2

        if training:
            batch_mean = tf.reduce_mean(instance_mean, [0], keepdims=True)
            batch_var = tf.reduce_mean(instance_var + instance_mean**2, [0], keepdims=True) - batch_mean**2

            batch_mean = self.assign_moving_average(self.moving_mean, batch_mean)
            batch_var = self.assign_moving_average(self.moving_var, batch_var)
        else:
            batch_mean, batch_var = self.moving_mean, self.moving_var

        mean_weights = tf.nn.softmax(self.mean_weights)
        var_weights = tf.nn.softmax(self.var_weights)

        mean = mean_weights[0] * instance_mean + mean_weights[1] * layer_mean + mean_weights[2] * batch_mean
        var = var_weights[0] * instance_var + var_weights[1] * layer_var + var_weights[2] * batch_var

        inv_square_root = tf.math.rsqrt(var + 1e-7)
        # Scale and shift
        inv_square_root *= self.gamma

        output = (input_tensor - mean) * inv_square_root + self.beta

        return tf.reshape(output, original_shape)

    def get_config(self):

        config = super().get_config()

        config.update({'momentum': self.momentum,
                       'trainable': self.trainable,
                       'name': self.name})

        return config


def get_val_from_source(source_matrix, x_coords, y_coords):

    batch_size = tf.shape(x_coords)[0]
    num_elements = tf.shape(x_coords)[1]

    batch_idx = tf.expand_dims(tf.range(batch_size, dtype=tf.float32), axis=-1)
    batch_idx = tf.tile(batch_idx, [1, num_elements])
    batch_idx = tf.reshape(batch_idx, [-1])

    x_coords = tf.reshape(x_coords, [-1])
    y_coords = tf.reshape(y_coords, [-1])

    coords = tf.transpose(tf.stack([batch_idx, y_coords, x_coords]))
    coords = tf.cast(coords, tf.int32)

    values = tf.gather_nd(source_matrix, coords)

    return tf.reshape(values, [batch_size, -1])


def bilinear_interpolation(source_matrix, target_x, target_y):

    height, width = tf.shape(source_matrix)[1:3]

    target_x = 0.5 * ((target_x + 1.0) * tf.cast(width - 2, tf.float32))
    target_y = 0.5 * ((target_y + 1.0) * tf.cast(height - 2, tf.float32))

    x0 = tf.math.floor(target_x)
    x1 = x0 + 1.
    y0 = tf.math.floor(target_y)
    y1 = y0 + 1.

    x0 = tf.clip_by_value(x0, 0., tf.cast(width - 1, tf.float32))
    y0 = tf.clip_by_value(y0, 0., tf.cast(height - 1, tf.float32))
    x1 = tf.clip_by_value(x1, 0., tf.cast(width - 1, tf.float32))
    y1 = tf.clip_by_value(y1, 0., tf.cast(height - 1, tf.float32))

    q00 = get_val_from_source(source_matrix, x0, y0)
    q01 = get_val_from_source(source_matrix, x1, y0)
    q10 = get_val_from_source(source_matrix, x0, y1)
    q11 = get_val_from_source(source_matrix, x1, y1)

    wa = (x1 - target_x) * (y1 - target_y)
    wb = (target_x - x0) * (y1 - target_y)
    wc = (x1 - target_x) * (target_y - y0)
    wd = (target_x - x0) * (target_y - y0)

    lower_left_contrib = wa * q00
    lower_right_contrib = wb * q01
    upper_left_contrib = wc * q10
    upper_right_contrib = wd * q11

    pixel_values = lower_left_contrib + lower_right_contrib + upper_left_contrib + upper_right_contrib

    return tf.reshape(pixel_values, [-1, height, width])


def grid_generator(height, width, transformation_matrix):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.
    - width: desired width of grid/output. Used
      to downsample or upsample.
    - transformation_matrix: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 transformation_matrix parameters of
      the form (2x3) that define the affine transformation T.
    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    batch_size = tf.shape(transformation_matrix)[0]

    # create normalized 2D grid
    cols = tf.linspace(-1.0, 1.0, width)
    rows = tf.linspace(-1.0, 1.0, height)
    col_idx, row_idx = tf.meshgrid(cols, rows)

    # flatten
    row_idx_flat = tf.reshape(row_idx, [-1])
    col_idx_flat = tf.reshape(col_idx, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(row_idx_flat)

    # (3, width * height)
    sampling_grid = tf.stack([col_idx_flat, row_idx_flat, ones])

    # repeat grid batch_size times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)

    # (batch_size, 3, width * height)
    sampling_grid = tf.tile(sampling_grid, [batch_size, 1, 1])

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(transformation_matrix, sampling_grid)

    return batch_grids
