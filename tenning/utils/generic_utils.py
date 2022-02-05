import types
import importlib
import logging
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.python.util import nest
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
from sklearn import manifold
import tensorflow as tf
import numpy as np
import functools
import warnings
import math
import time
import json
import glob
import os


def maybe_load_weights(weights_dir, model, by_name=True):

    latest_weights = None

    if not os.path.exists(weights_dir):
        print(f"The specified directory {weights_dir} does not exists. It will be created now.")
        os.mkdir(weights_dir)

    weight_files = glob.glob(os.path.join(weights_dir, '*.h5'))
    if weight_files:

        # Sorts the list of weight files by last modification date (newer to older)
        weight_files.sort(key=os.path.getmtime, reverse=True)

        print(f"Latest weight file: {weight_files[0]}")

        latest_weights = weight_files[0]

        # Restores latest weights
        model.load_weights(latest_weights, by_name=by_name)
    else:
        save_path = os.path.join(weights_dir, f"starting_weights.h5")
        print(f"No weight file found at {weights_dir}! Creating {save_path} file containing the random initialized weights...")
        # Save the first random weights to enforce all subsequent training to
        # start at the same point
        model.save_weights(save_path, save_format='h5')


def save_plot_as_image(images, titles, file_path):

    fig = plt.figure(figsize=(30, 30))

    num_figures = np.ceil(np.sqrt(titles.shape[0])).astype(np.uint8)

    for index, (image, title) in enumerate(zip(images, titles)):
        ax = plt.subplot(num_figures, num_figures, index + 1)

        ax.set_xticks([])
        ax.set_yticks([])

        image = np.squeeze(image)

        ax.imshow(image, cmap='gray')
        ax.set_title(f"{title}")

    fig.savefig(file_path)


def compare_images(reconstruction, groundtruth):

    reconstruction = np.squeeze(reconstruction)
    groundtruth = np.squeeze(groundtruth)

    ssim = np.mean([structural_similarity(recon, ground) for recon, ground in zip(reconstruction, groundtruth)])
    psnr = np.mean([peak_signal_noise_ratio(recon, ground) for recon, ground in zip(reconstruction, groundtruth)])
    rmse = np.mean([mean_squared_error(recon, ground, squared=False) for recon, ground in zip(reconstruction, groundtruth)])
    mae = np.mean([mean_absolute_error(recon, ground) for recon, ground in zip(reconstruction, groundtruth)])

    print(f"mean SSIM: {ssim}\nmean PSNR: {psnr}\nmean RMSE: {rmse}\nmean MAE: {mae}")


def save_json_model(model, file_path):

    print(f"Saving model '{model.name}' to {file_path} file...")

    with open(file_path, "w") as f:
        json.dump(model.to_json(), f)


def build_json_model(weights_dir, json_path, custom_objects=None):

    with open(json_path, "r") as f:
        model_config = json.load(f)

    print(f"Building an existing model from {json_path} file...")
    model = tf.keras.models.model_from_json(model_config, custom_objects=custom_objects)
    maybe_load_weights(weights_dir, model)

    return model


def get_output_shape(model, include_batch=False):
    """ Gets the shape of last dimension:
            (D,) for 1-D tensors
            (H, W, C) or (C, H, W) for 2-D tensors
    """

    if model is None:
        return [None]

    output_shape = list(model.output_shape)

    # Checks if output_shape is a nested list
    if any(isinstance(i, tuple) for i in output_shape):

        out_shape = []
        for shape in output_shape:
            if include_batch:
                out_shape.append([-1] + list(shape[1:]))
            else:
                out_shape.append(list(shape[1:]))

        return out_shape
    else:
        if include_batch:
            # return [-1] + list(model.layers[-1].output_shape[0][1:])
            return [-1] + list(model.output_shape[1:])
        else:
            # return list(model.layers[-1].output_shape[0][1:])
            return list(model.output_shape[1:])


def get_input_shape(model, include_batch=False):
    """ Gets the shape of last dimension:
            (D,) for 1-D tensors
            (H, W, C) or (C, H, W) for 2-D tensors
    """

    if model is None:
        return [None]

    input_shape = list(model.input_shape)

    # Checks if input_shape is a nested list
    if any(isinstance(i, tuple) for i in input_shape):

        in_shape = []
        for shape in input_shape:
            if include_batch:
                in_shape.append([-1] + list(shape[1:]))
            else:
                in_shape.append(list(shape[1:]))

        return in_shape
    else:
        if include_batch:
            # return list([-1] + model.layers[0].input_shape[0][1:])
            return [-1] + list(model.input_shape[1:])
        else:
            # return list(model.layers[0].input_shape[0][1:])
            return list(model.input_shape[1:])


def get_object_config(o):
    if o is None:
        return None

    config = {
        'class_name': o.__class__.__name__,
        'config': o.get_config()
    }
    return config


def to_numpy_or_python_type(tensors):
    """Converts a structure of `Tensor`s to `NumPy` arrays or Python scalar types.
    For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
    it converts it to a Python type, such as a float or int, by calling
    `result.item()`.
    Numpy scalars are converted, as Python types are often more convenient to deal
    with. This is especially useful for bfloat16 Numpy scalars, which don't
    support as many operations as other Numpy values.
    Args:
      tensors: A structure of tensors.
    Returns:
      `tensors`, but scalar tensors are converted to Python types and non-scalar
      tensors are converted to Numpy arrays.
    """
    def _to_single_numpy_or_python_type(t):
        if isinstance(t, ops.Tensor):
            x = t.numpy()
            return x.item() if np.ndim(x) == 0 else x
        return t  # Don't turn ragged or sparse tensors to NumPy.

    return nest.map_structure(_to_single_numpy_or_python_type, tensors)


def display_image_from_batch(image_batch, title_batch=None, num_images=4):

    images = to_numpy_or_python_type(image_batch)

    titles = [''] * len(images)

    if title_batch is not None:
        titles = to_numpy_or_python_type(title_batch)

    if images.shape[-1] == 1:
        images = images.reshape((images.shape[0], images.shape[1], images.shape[2]))

    # images = np.squeeze(images)

    for i in range(num_images):
        plt.figure(i)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])

    plt.show()


def display_layer_features(model, layer_name, input_data, suptitle=''):

    desired_layer = model.get_layer(layer_name)

    testing_model = tf.keras.Model(inputs=model.input, outputs=desired_layer.output)
    testing_model.build(model.input_shape)

    predictions = testing_model.predict_on_batch(input_data)

    predictions = np.squeeze(predictions).astype(np.float32)

    if predictions.ndim == 2:
        num_channels = 1
    else:
        num_channels = predictions.shape[-1]

    square = np.ceil(np.sqrt(num_channels))

    plt.figure(figsize=(12.8, 9.6))
    plt.suptitle(suptitle)
    for channel in range(num_channels):
        plt.subplot(square, square, channel + 1)
        if predictions.ndim == 2:
            plt.imshow(predictions, cmap='gray')
        else:
            plt.imshow(predictions[:, :, channel], cmap='gray')

    plt.show()


def pair_enum(features, batch_size=1):

    features1 = tf.tile(features, tf.constant([batch_size, 1]))
    features2 = tf.tile(features, tf.constant([1, batch_size]))
    features2 = tf.reshape(features2, tf.constant([-1, features.shape[1]]))

    pairs = tf.concat([features1, features2], axis=1)

    return pairs


def visualize_clusters(Z, labels, num_clusters, filename=None, save_fig=False):
    '''
    TSNE visualization of the points in latent space Z
    :param Z: Numpy array containing points in latent space in which clustering was performed
    :param labels: True labels - used for coloring points
    :param num_clusters: Total number of clusters
    :param filename: filename where the plot should be saved
    :param save_fig: if the figure should be saved into a file
    :return: None - (side effect) saves clustering visualization plot in specified location
    '''
    Z = to_numpy_or_python_type(Z)
    labels = to_numpy_or_python_type(labels)

    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(Z)
    fig = plt.figure(0)
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))

    if save_fig:

        if not filename:
            raise ValueError(f"You must provide a valid filename")

        fig.savefig(filename, dpi=300)

    plt.show()


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def timing(function):

    @functools.wraps(function)
    def wrapper(*args, **kwargs):

        print(f"Estimating execution time for function {function.__name__}...")

        start = time.time()

        result = function(*args, **kwargs)

        end = time.time()

        print(f"Function {function.__name__} took {end - start} seconds")

        return result

    return wrapper


def is_perfect_square(number):
    return math.sqrt(number).is_integer()


class BatchPlotter:

    def __init__(self, figsize=(6, 8), max_per_plot=4, color_map='gray'):

        assert is_perfect_square(max_per_plot), f"The argument 'max_per_plot' must be a perfect square number."

        self._figsize = figsize
        self._max_per_plot = max_per_plot
        self._color_map = color_map
        self._grid_dim = int(np.sqrt(max_per_plot))
        self._fig_id = None

    def _update_id(self):

        if self._fig_id is not None:
            self._fig_id += 1
        else:
            self._fig_id = 0

    @property
    def max_per_plot(self):
        return self._max_per_plot

    @property
    def figsize(self):
        return self._figsize

    @property
    def color_map(self):
        return self._color_map

    @max_per_plot.setter
    def max_per_plot(self, value):

        assert is_perfect_square(value), f"The argument 'value' must be a perfect square number."
        self._max_per_plot = value
        self._grid_dim = int(np.sqrt(value))

    @figsize.setter
    def figsize(self, value):
        self._figsize = value

    @color_map.setter
    def color_map(self, value):
        self._color_map = value

    def num_plots(self, batch):
        """ Gets the number of plots generated based on the batch size and
            the maximum number of images per plot
        """
        return np.ceil(len(batch) / self._max_per_plot).astype(np.int32)

    def gen_plot(self, images):

        plt.figure(self._fig_id, figsize=self._figsize)

        for idx, image in enumerate(images):
            plt.subplot(self._grid_dim, self._grid_dim, idx + 1)
            plt.imshow(image, cmap=self._color_map)

        self._update_id()

    def plot(self, batch):

        batch = to_numpy_or_python_type(batch)

        # Squeezes possible one dimensional axes
        batch = batch.squeeze()

        if batch.ndim == 2:
            # Puts batch into list if it has only one image
            batch = [batch]

        num_plots = self.num_plots(batch)
        batch_indices = np.arange(num_plots)

        for idx in batch_indices:

            images = batch[idx * self._max_per_plot: (idx + 1) * self._max_per_plot]

            self.gen_plot(images)

        plt.show()


def eye_like(matrix):
    """ Returns a identity matrix with the same shape as the original matrix.
        Arguments:
        matrix: A 2D-tensor [height, width] or 3D-tensor [batch_size, height, width].
        Output shape:
        A tensor with the shape as input matrix.
    """

    if tf.rank(matrix) < 3:
        matrix = matrix[tf.newaxis, ...]

    batch_size = tf.shape(matrix)[0]
    height = tf.shape(matrix)[1]
    width = tf.shape(matrix)[2]

    identity = tf.eye(height, num_columns=width, batch_shape=[batch_size])

    return identity


class LazyLoader(types.ModuleType):
    """ Lazily import a module, mainly to avoid pulling in large dependencies.

        Args:
            local_name (str): The name of the module.
            parent_module_globals (dict): A dictionary of the current global symbol table.
            name (name): The path to the module. Example: package.local_name.
            warning (str, optional): A custom warning message. Defaults to None.
    """

    # The lint error here is incorrect.
    def __init__(self, local_name, parent_module_globals, name, warning=None):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._warning = warning

        super().__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Emit a warning if one was specified
        if self._warning:
            logging.warning(self._warning)
            # Make sure to only warn once.
            self._warning = None

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
