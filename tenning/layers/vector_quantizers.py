from tensorflow.keras.layers import Layer
import tensorflow as tf


class VectorQuantizer(Layer):

    def __init__(self, embedding_dim, num_embeddings, commitment_cost, trainable=True, name='vq', **kwargs):
        super().__init__(name=name, trainable=trainable)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

    def build(self, input_shape):

        # Instantiate embedding table with initialized weights
        self.emb_weights = self.add_weight(shape=[self.num_embeddings, self.embedding_dim],
                                           initializer=tf.keras.initializers.he_normal(),
                                           trainable=self.trainable,
                                           name=self.name + '/embed')

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        return tuple(shape)

    def call(self, input_tensor):
        '''
        Note: 
            shape_of_inputs=[batch_size, ?, ?, embedding_dim]
        '''
        # Assert last dimension of inputs is same as embedding_dim

        flat_inputs = tf.reshape(input_tensor, [-1, self.embedding_dim])

        distances = tf.reduce_sum(flat_inputs**2, 1, keepdims=True) \
            - 2 * tf.matmul(flat_inputs, tf.transpose(self.emb_weights)) \
            + tf.reduce_sum(tf.transpose(self.emb_weights)**2, 0, keepdims=True)

        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(input_tensor)[:-1])  # shape=[batch_size, ?, ?]

        quantized = tf.nn.embedding_lookup(self.emb_weights, encoding_indices)

        inp_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - input_tensor)**2)
        emb_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(input_tensor))**2)
        loss = emb_latent_loss + self.commitment_cost * inp_latent_loss  # used to optimize self.emb_vq only!

        self.add_loss(loss, inputs=True)

        quantized = input_tensor + tf.stop_gradient(quantized - input_tensor)
        # Important Note:
        #   This step is used to copy the gradient from inputs to quantized.

        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))
        # The perplexity is the exponentiation of the entropy,
        # indicating how many codes are 'active' on average.
        # We hope the perplexity is larger.

        self.add_metric(perplexity, aggregation='mean', name='perplexity')

        return quantized

    def get_config(self):

        config = super().get_config()

        config.update({'embedding_dim': self.embedding_dim,
                       'num_embeddings': self.num_embeddings,
                       'commitment_cost': self.commitment_cost,
                       'trainable': self.trainable,
                       'name': self.name, })

        return config


class VectorQuantizerEMA(Layer):

    def __init__(self,
                 embedding_dim,
                 num_embeddings,
                 commitment_cost=0.25,
                 decay=0.99,
                 epsilon=1e-5,
                 trainable=True,
                 name='vq_ema',
                 **kwargs):
        super().__init__(name=name, trainable=trainable)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

    def build(self, input_shape):

        self.emb_weights = self.add_weight(shape=[self.num_embeddings, self.embedding_dim],
                                           initializer=tf.keras.initializers.he_normal(),
                                           trainable=False,
                                           name=self.name + '/embed',
                                           dtype=tf.float32)

        self.ema_w = self.add_weight(shape=[self.num_embeddings, self.embedding_dim],
                                     initializer=tf.keras.initializers.he_normal(),
                                     trainable=False,
                                     name=self.name + '/ema_w',
                                     dtype=tf.float32)

        self.ema_cluster_size = self.add_weight(shape=[self.num_embeddings],
                                                initializer=tf.keras.initializers.zeros(),
                                                trainable=False,
                                                name=self.name + '/ema_cluster_size',
                                                dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        return tuple(shape)

    def call(self, input_tensor, training=True):
        '''
        Note: 
            shape_of_inputs=[batch_size, ?, ?, embedding_dim]
        '''
        # Assert last dimension of inputs is same as embedding_dim
        flat_inputs = tf.reshape(input_tensor, [-1, self.embedding_dim])

        distances = tf.reduce_sum(flat_inputs**2, 1, keepdims=True) \
            - 2 * tf.matmul(flat_inputs, tf.transpose(self.emb_weights)) \
            + tf.reduce_sum(tf.transpose(self.emb_weights)**2, 0, keepdims=True)

        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings, dtype=input_tensor.dtype)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(input_tensor)[:-1])  # shape=[batch_size, ?, ?]
        quantized = tf.nn.embedding_lookup(self.emb_weights, encoding_indices)
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - input_tensor) ** 2)

        loss = tf.convert_to_tensor(0., dtype=input_tensor.dtype)

        if training:

            cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * tf.reduce_sum(encodings, 0)
            self.ema_cluster_size.assign(tf.cast(cluster_size, tf.float32))

            dw = tf.matmul(flat_inputs, encodings, transpose_a=True)

            ema_w = self.ema_w * self.decay + (1 - self.decay) * tf.transpose(dw)
            self.ema_w.assign(tf.cast(ema_w, tf.float32))

            n = tf.reduce_sum(self.ema_cluster_size)
            updated_ema_cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n

            normalised_updated_ema_w = self.ema_w / tf.reshape(updated_ema_cluster_size, [-1, 1])
            self.emb_weights.assign(tf.cast(normalised_updated_ema_w, tf.float32))

            loss = self.commitment_cost * e_latent_loss

        self.add_loss(loss, inputs=True)

        quantized = input_tensor + tf.stop_gradient(quantized - input_tensor)
        avg_probs = tf.reduce_mean(encodings, 0)

        perplexity = tf.convert_to_tensor(0., dtype=input_tensor.dtype)

        if training and self.trainable:
            perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))

        self.add_metric(perplexity, aggregation='mean', name='perplexity')

        return quantized

    def get_config(self):

        config = super().get_config()

        config.update({'embedding_dim': self.embedding_dim,
                       'num_embeddings': self.num_embeddings,
                       'commitment_cost': self.commitment_cost,
                       'decay': self.decay,
                       'epsilon': self.epsilon,
                       'trainable': self.trainable,
                       'name': self.name, })

        return config
