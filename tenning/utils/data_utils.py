from itertools import compress
from random import shuffle
from random import sample
from typing import Union
from scipy import sparse
import tensorflow as tf
import pandas as pd
import healpy as hp
import numpy as np
import itertools
import time
import os
from .generic_utils import deprecated
from .generic_utils import eye_like


def normalize_batch(data: Union[tf.Tensor, np.ndarray],
                    old_range: list = [0, 255],
                    new_range: list = [-1, 1]) -> Union[tf.Tensor, np.ndarray]:
    """ Rescales the data to a range specified by 'new_range'

        Args:
            data: The target data.
            old_range: A list that specifies the current numeric range of
                       the data ([lower_bound, upper_bound]).
            new_range: A list that specifies what will be the new numeric 
                       range of the data ([lower_bound, upper_bound]).

        Returns:
            The data in a new range.
    """

    diff = new_range[1] - new_range[0]

    data = diff * ((data - old_range[0]) / (old_range[1] - old_range[0])) + new_range[0]

    return data


def standardize_batch(data: Union[tf.Tensor, np.ndarray]) -> Union[tf.Tensor, np.ndarray]:
    """ Rescales the data to have zero mean and unit standard deviation.

        Args:
            data: The target data.

        Returns:
            The standardized data.
    """

    epsilon = 1e-7  # To avoid numerical issues

    ndim = tf.rank(data)

    if ndim > 1:
        mean = tf.reduce_mean(data, axis=range(1, ndim), keepdims=True)
        stddev = tf.math.reduce_std(data, axis=range(1, ndim), keepdims=True)
    else:
        mean = tf.reduce_mean(data, axis=0, keepdims=True)
        stddev = tf.math.reduce_std(data, axis=0, keepdims=True)

    # Standardize data to have mean = 0 and variance = 1
    standardized_data = (data - mean) / (stddev + epsilon)

    return standardized_data


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class IteratorBuilder:
    """'IteratorBuilder' uses the tensorflow Dataset API to build a dataset iterator.

        It accepts a pandas dataframe or a numpy array as dataset. It can cache the
        dataset to memory or to disk, the latter if a directory is specified. There are
        two methods that must be implemented by the user: 'yielder' and 'post_process'.
        The former can handle dataset manipulation, but the user must have in mind that
        this function will store its return value in memory or disk. Thus, one needs to
        avoid loading bigger files such as images if one wants to save memory. Is advisable
        to leave the hard work for the 'post_process' method to save memory.

        For example, when dealing with an image dataset is advisable to return the image file
        paths in the 'yielder' method and only load the image on the 'post_process' method.
        For this case, it will be an increase in the dataset pipeline but there won't be a
        memory overflow.

        Args:
            kwargs: Any pair of key and values. Some useful keywords can be added, such as
            'batch_size', 'shuffle', 'drop_remainder', 'val_ratio' and 'test_ratio'. The
            'batch_size' parameter specifies the number of samples which will be taken from
            the dataset after each iteration (defaults to 32). If the last batch can't have
            the same size of the other batches it can be dropped by setting 'drop_remainder'
            to True (defaults to True). The dataset gets shuffled each time the iterator is
            restarted if 'shuffle' is set to true (defaults to False). The 'val_ratio' and
            the 'test_ratio' parameters specify the dataset split ratio.
    """

    def __init__(self, **kwargs):

        self._batch_size = kwargs.pop('batch_size', 32)
        self._shuffle = kwargs.pop('shuffle', False)
        self._drop_remainder = kwargs.pop('drop_remainder', True)
        self._val_ratio = kwargs.pop('val_ratio', 0.2)
        self._test_ratio = kwargs.pop('test_ratio', 0.3)

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def set_dataset(self, dataset: Union[pd.DataFrame, np.array]) -> None:
        """ Splits the dataset into train, test and validation datasets and leave them
            prepared so a iterator can be created.

            Args:
                dataset: Input dataset.
        """

        train_dataset, val_dataset, test_dataset = split_dataset(dataset,
                                                                 val_ratio=self._val_ratio,
                                                                 test_ratio=self._test_ratio)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def yielder(self, data) -> Union[list, tuple]:
        """ This method must be implemented by the user and can perform any
            transformation to the dataset. The return of this method will be
            cached to memory or disk.

            * It's advisable to leave the hard work to the 'post_process' method.

            Args:
                params: A Tensor containing a batch from the dataset.

            Returns:
                The return value must be a iterable containing the data. For example:
                if the dataset is the Iris flowers dataset and we want to split this
                data into four sets: 'sepal length', 'sepal width', 'petal length'
                and 'petal width', then we must perform this split in this method and
                return a iterable as follows:

                    sepal_length = dataset[:, 0]
                    sepal_width = dataset[:, 1]
                    petal_length = dataset[:, 2]
                    petal_width = dataset[:, 3]

                    return (sepal_length, sepal_width, petal_length, petal_width)
        """
        raise NotImplementedError()

    def post_process(self, *args):
        raise NotImplementedError()

    def gen_iterator(self, dataset, dataset_size, buffer_size=None, cache_file="", seed=42):

        if (dataset_size < self._batch_size) or (dataset is None):
            return None

        buffer_size = buffer_size or dataset_size * 3

        dataset = tf.data.Dataset.from_tensor_slices(dataset)

        if self._shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)

        dataset = dataset.batch(self._batch_size, drop_remainder=self._drop_remainder)

        # Preprocessing of the dataset before cache. Must be a cheaper funcionality and that
        # doesn't generate bigger files to avoid memory hunger computation
        dataset = dataset.map(self.yielder, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Store cache in HDD or RAM, the former is only performed if a 'cache_file' is given
        dataset = dataset.cache(cache_file)

        # Apply image preprocessing to the whole batch
        dataset = dataset.map(self.post_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Overlaps the preprocessing and model execution of a training step
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def train_iterator(self, buffer_size=None, cache_file="", seed=42):
        return self.gen_iterator(self._train_dataset, len(self.train_dataset), buffer_size=buffer_size, cache_file=cache_file, seed=seed)

    def test_iterator(self, buffer_size=None, cache_file="", seed=42):
        return self.gen_iterator(self._test_dataset, len(self.test_dataset), buffer_size=buffer_size, cache_file=cache_file, seed=seed)

    def val_iterator(self, buffer_size=None, cache_file="", seed=42):
        return self.gen_iterator(self._val_dataset, len(self.val_dataset), buffer_size=buffer_size, cache_file=cache_file, seed=seed)

    def num_batches(self, data_size):
        """Returns number of batches"""

        if self._drop_remainder:
            # Ignores the last batch if it is smaller than the batch size
            return np.floor(data_size / self._batch_size).astype(np.int32)

        return np.ceil(data_size / self._batch_size).astype(np.int32)

    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, train_dataset):

        if isinstance(train_dataset, pd.DataFrame):
            self._train_dataset = dataframe_to_tensor(train_dataset, train_dataset.columns)

        elif isinstance(train_dataset, np.ndarray):
            self._train_dataset = train_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, test_dataset):

        if isinstance(test_dataset, pd.DataFrame):
            self._test_dataset = dataframe_to_tensor(test_dataset, test_dataset.columns)

        elif isinstance(test_dataset, np.ndarray):
            self._test_dataset = test_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @val_dataset.setter
    def val_dataset(self, val_dataset):

        if isinstance(val_dataset, pd.DataFrame):
            self._val_dataset = dataframe_to_tensor(val_dataset, val_dataset.columns)

        elif isinstance(val_dataset, np.ndarray):
            self._val_dataset = val_dataset

    def __len__(self):
        """Returns the total dataset size"""

        return len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)

    def get_config(self):

        train_info = {"size": len(self.train_dataset), "num_batches": self.num_batches(len(self.train_dataset))}
        test_info = {"size": len(self.test_dataset), "num_batches": self.num_batches(len(self.test_dataset))}
        val_info = {"size": len(self.val_dataset), "num_batches": self.num_batches(len(self.val_dataset))}

        config = {'train_info': train_info,
                  'val_info': val_info,
                  'test_info': test_info,
                  'batch_size': self._batch_size}

        return config


def dataframe_to_tensor(dataframe, column_names):

    if len(dataframe) == 0:
        return None

    # Tensors don't accept different data types, so the entire dataframe
    # is converted to string and then transformed to tensor.
    # When manipulating the data you should convert to the right data type
    # by yourself
    dataframe = dataframe.astype('str')  # << NEEDS AJUSTMENT BECAUSE THE RESULTING DATAFRAME
    # IS NOT OF STRING TYPE BUT OBJECT TYPE

    tensors = [tf.convert_to_tensor(dataframe[col].values) for col in column_names]

    return tf.stack(tensors, axis=-1)


@tf.function
def to_bytes(tensor):
    """ Reads files (as bytes) from a tensor containing the directories.
        'tensor' must be a string tensor or sequence of string tensors.
    """
    return tf.map_fn(tf.io.read_file, tensor, dtype=tf.string, swap_memory=True)


@tf.function
def to_img(tensor):
    """ Decodes bytes to a valid image format.
        'tensor' must be a byte representation of the images, usually
        the return values from 'to_bytes()' fits this function.
    """
    return tf.map_fn(tf.io.decode_png, tensor, dtype=tf.uint8, swap_memory=True)


@tf.function
def to_abspath(tensor, root_directory):
    """ Receives a tensor containing relative paths to a file
        starting from 'root_directory' and builds a new tensor
        where each row now has an absolute path to the files
    """

    return tf.map_fn(lambda relative: tf.strings.join([root_directory, relative], separator="/"), tensor, dtype=tf.string, swap_memory=True)


def split_dataset(dataset, val_ratio=0.2, test_ratio=0.2):
    """Splits dataset into train, validation and test datasets"""

    test_m = 1. - test_ratio
    val_m = test_m - val_ratio

    if isinstance(dataset, pd.DataFrame):
        shuffled_data = dataset.sample(frac=1)

        train, validation, test = np.split(shuffled_data, [int(val_m * len(dataset)), int(test_m * len(dataset))])

        train = train.reset_index(drop=True)
        validation = validation.reset_index(drop=True)
        test = test.reset_index(drop=True)

    elif isinstance(dataset, np.ndarray):
        np.random.shuffle(dataset)

        train, validation, test = np.split(dataset, [int(val_m * len(dataset)), int(test_m * len(dataset))])

    else:
        raise TypeError(f"The dataset must be a pandas dataframe or a numpy array, but got {type(dataset)}")

    return train, validation, test


def laplacian_from_healpix(nside=16, lap_type='normalized', dtype=np.float32):
    """Return an unnormalized weight matrix for a graph using the HEALPIX sampling.
    Arguments:
        nside : The healpix nside parameter, must be a integer power of 2, less than 2**30.
        lap_type: If 'normalized' then a normalized laplacian matrix (laplacian = I − D^{−1/2}AD^{−1/2})
                is returned, otherwise return just the combinatorial laplacian matrix (laplacian = Degree - Adjacency).
        dtype : (optional) The desired data type of the weight matrix.
    Output shape:
    A square matrix with dimension equal to [nside**2 * 12].
    """

    if lap_type != 'normalized' or lap_type != 'combinatorial':
        raise ValueError('Unknown Laplacian type {}'.format(lap_type))

    indexes = list(range(nside**2 * 12))
    npix = len(indexes)  # Number of pixels.

    # Get the coordinates.
    x, y, z = hp.pix2vec(nside, indexes, nest=True)
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords, dtype=dtype)
    # Get the 7-8 neighbors.
    neighbors = hp.pixelfunc.get_all_neighbours(nside, indexes, nest=True)

    # Indices of non-zero values in the adjacency matrix.
    col_index = neighbors.T.reshape([-1])
    row_index = np.repeat(indexes, 8)

    keep = [c in indexes for c in col_index]
    col_index = [el for el in col_index[keep]]
    row_index = [el for el in row_index[keep]]

    # Compute Euclidean distances between neighbors.
    distances = np.sum((coords[row_index] - coords[col_index])**2, axis=1)
    # slower: np.linalg.norm(coords[row_index] - coords[col_index], axis=1)**2

    # Compute similarities / edge weights.
    kernel_width = np.mean(distances)
    weights = np.exp(-distances / (2 * kernel_width))

    # Similarity proposed by Renata & Pascal, ICCV 2017.
    # weights = 1 / distances

    # Build the sparse matrix.
    W = sparse.csr_matrix((weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype)

    """Build a Laplacian (tensorflow)."""
    d = np.ravel(W.sum(1))
    if lap_type == 'combinatorial':
        D = sparse.diags(d, 0, dtype=dtype)
        laplacian = (D - W).tocsc()
    elif lap_type == 'normalized':
        d12 = np.power(d, -0.5)
        D12 = sparse.diags(np.ravel(d12), 0, dtype=dtype).tocsc()
        laplacian = sparse.identity(d.shape[0], dtype=dtype) - D12 * W * D12

    return np.asarray(laplacian.todense())


def build_laplacians(nsides, lap_type='normalized', dtype=np.float32):
    """Build a list of Laplacians (and down-sampling factors) from a list of nsides."""
    laplacians = []
    pooling_size = []
    nside_last = None
    for i, nside in enumerate(nsides):
        if i > 0:  # First is input dimension.
            pooling_size.append((nside_last // nside)**2)
        nside_last = nside
        if i < len(nsides) - 1:  # Last does not need a Laplacian.
            laplacian = laplacian_from_healpix(nside=nside, lap_type=lap_type, dtype=dtype)
            laplacians.append(laplacian)
    return laplacians, pooling_size


def rescale_laplacian(laplacian, lmax=2, scale=1):
    """Rescale the Laplacian eigenvalues in [-scale,scale]."""
    identity = eye_like(laplacian)
    laplacian *= 2 * scale / lmax
    laplacian -= identity
    return laplacian


def benchmark_data_pipeline(dataset_iterator, epochs=1):
    """ Evaluates data pipeline performance """
    start_time = time.perf_counter()

    for _ in range(epochs):

        counter = itertools.count()

        # num_batches <- 0
        num_batches = next(counter)

        start_time2 = time.perf_counter()

        for _ in dataset_iterator:

            num_batches = next(counter)

            # Dummy computation time
            time.sleep(0.01)

        print(f"Number of batches: {num_batches}; Elapsed time: {time.perf_counter() - start_time2}")

    print(f"Execution time: {time.perf_counter() - start_time}")
