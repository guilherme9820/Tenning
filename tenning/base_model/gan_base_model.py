from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import Model
from itertools import count
import tensorflow as tf
import numpy as np
import copy
from tenning.generic_utils import to_numpy_or_python_type


class GANBaseModel:

    def __init__(self, *args, **kwargs):

        self.history = None

    def fit(self,
            dataset_iterator,
            epochs,
            batch_size,
            n_critic=5,
            verbose=1,
            samples=None,
            steps_per_epoch=None,
            validation_iterator=None,
            validation_steps=None,
            validation_freq=1,
            callbacks=None):

        callbacks = callbacks or []

        disc_metrics_names = []
        for names in self.discriminator.metrics_names:
            disc_metrics_names.append('disc_' + names)

        gen_metrics_names = []
        for names in self.generator.metrics_names:
            gen_metrics_names.append('gen_' + names)

        if not isinstance(callbacks, callbacks_module.CallbackList):
            self.history = tf.keras.callbacks.History()
            callbacks.append(self.history)

            callbacks = callbacks_module.CallbackList(callbacks)

            params = {
                'batch_size': batch_size,
                'verbose': verbose,
                'metrics': disc_metrics_names + gen_metrics_names,
                'epochs': epochs,
                'steps': steps_per_epoch,
                'samples': samples
            }

            callbacks.set_model(self.generator)
            callbacks.set_params(params)

        callbacks.on_train_begin()

        epoch_counter = count()

        epoch_id = next(epoch_counter)

        did_abort = False

        discriminator_logs = dict.fromkeys(disc_metrics_names, 0)
        generator_logs = dict.fromkeys(gen_metrics_names, 0)

        try:
            while epoch_id < epochs:

                # Enables different behaviours for some layers, such as Batchnorm, dropout...
                tf.keras.backend.set_learning_phase(1)

                self.reset_metrics()

                batch_counter = count()

                batch_id = next(batch_counter)

                callbacks.on_epoch_begin(epoch_id)

                critic_count = 0

                for data in dataset_iterator:

                    callbacks.on_batch_begin(batch_id)

                    discriminator_metrics = self.update_discriminator(**data)

                    if tf.equal(tf.rank(discriminator_metrics), 0):
                        # In case of discriminator_metrics set no None in compile method
                        discriminator_metrics = [discriminator_metrics]

                    discriminator_logs.update(zip(discriminator_logs, discriminator_metrics))
                    discriminator_logs['size'] = batch_size

                    critic_count += 1

                    if critic_count == n_critic:
                        generator_metrics = self.update_generator(**data)

                        if tf.equal(tf.rank(generator_metrics), 0):
                            # In case of generator_metrics set no None in compile method
                            generator_metrics = [generator_metrics]

                        generator_logs.update(zip(generator_logs, generator_metrics))
                        generator_logs['size'] = batch_size

                        critic_count = 0

                        batch_logs = dict(discriminator_logs, **generator_logs)

                        callbacks.on_batch_end(batch_id, logs=batch_logs)

                    batch_id = next(batch_counter)

                epoch_logs = copy.copy(batch_logs)

                # Run validation.
                if validation_iterator and self._should_eval(epoch_id, validation_freq):
                    val_logs = self.evaluate(validation_iterator,
                                             batch_size,
                                             verbose=verbose,
                                             steps=validation_steps,
                                             callbacks=callbacks,
                                             return_dict=True)
                    val_logs = {'val_' + name: val for name, val in val_logs.items()}
                    epoch_logs.update(val_logs)

                callbacks.on_epoch_end(epoch_id, logs=epoch_logs)

                epoch_id = next(epoch_counter)

        except KeyboardInterrupt:

            did_abort = True

        callbacks.on_train_end(logs={'did_abort': did_abort})

        return self.history

    def evaluate(self, dataset_iterator, batch_size, verbose=1, steps=None, callbacks=None, return_dict=False):

        # Disables different behaviours for some layers, such as Batchnorm, dropout...
        tf.keras.backend.set_learning_phase(0)

        callbacks = callbacks or []

        gen_metrics_names = []
        for names in self.generator.metrics_names:
            gen_metrics_names.append('gen_' + names)

        all_metrics = gen_metrics_names

        if not isinstance(callbacks, callbacks_module.CallbackList):
            self.history = tf.keras.callbacks.History()
            callbacks.append(self.history)

            callbacks = callbacks_module.CallbackList(callbacks)

            params = {
                'batch_size': batch_size,
                'verbose': verbose,
                'metrics': all_metrics,
                'epochs': 1,
                'steps': steps
            }

            callbacks.set_model(self.generator)
            callbacks.set_params(params)

        generator_logs = dict.fromkeys(gen_metrics_names, 0)

        callbacks.on_test_begin()

        self.reset_metrics()

        batch_counter = count()

        batch_id = next(batch_counter)

        for data in dataset_iterator:

            callbacks.on_test_batch_begin(batch_id)

            metrics = self.predict_step(**data)

            if tf.equal(tf.rank(metrics), 0):
                metrics = [metrics]

            generator_logs.update(zip(generator_logs, metrics))
            generator_logs['size'] = batch_size

            callbacks.on_test_batch_end(batch_id, logs=generator_logs)

            batch_id = next(batch_counter)

        callbacks.on_test_end()

        generator_logs.pop('size')
        logs = to_numpy_or_python_type(generator_logs)
        if return_dict:
            return logs
        else:
            results = [logs.get(name, None) for name in all_metrics]
            if len(results) == 1:
                return results[0]

            return results

    def update_discriminator(self, **data_dict):
        raise NotImplementedError()

    def update_generator(self, **data_dict):
        raise NotImplementedError()

    def predict_step(self, **data_dict):
        raise NotImplementedError()

    def build_model(self):
        raise NotImplementedError()

    def reset_metrics(self):
        self.discriminator.reset_metrics()
        self.generator.reset_metrics()

    def _should_eval(self, epoch, validation_freq):
        epoch = epoch + 1  # one-index the user-facing epoch.
        if isinstance(validation_freq, int):
            return epoch % validation_freq == 0
        elif isinstance(validation_freq, list):
            return epoch in validation_freq
        else:
            raise ValueError('Expected `validation_freq` to be a list or int.')

    @property
    def discriminator(self):
        return self.__discriminator

    @discriminator.setter
    def discriminator(self, discriminator):
        self.__discriminator = discriminator

    @property
    def generator(self):
        return self.__generator

    @generator.setter
    def generator(self, generator):
        self.__generator = generator
