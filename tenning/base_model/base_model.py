from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import Model
from itertools import count
import tensorflow as tf
import numpy as np
import copy
from tenning.generic_utils import to_numpy_or_python_type


class BaseModel(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.history = None

        self.compiled = False

    def compile(self, **kwargs):

        if not self.built:
            self.init_graph()

        super().compile(**kwargs)

        self.reset_metrics()

        self.compiled = True

    def fit(self,
            dataset_iterator,
            epochs,
            batch_size,
            verbose=1,
            samples=None,
            steps_per_epoch=None,
            validation_iterator=None,
            validation_steps=None,
            validation_freq=1,
            callbacks=None,
            class_weight=None):

        if not self.compiled:
            raise RuntimeError(
                "Your tried to fit your model but it hasn\'t been compiled yet. Please call 'compile()' before 'fit()'.")

        callbacks = callbacks or []

        if not isinstance(callbacks, callbacks_module.CallbackList):
            self.history = tf.keras.callbacks.History()
            callbacks.append(self.history)

            callbacks = callbacks_module.CallbackList(callbacks)

            params = {
                'batch_size': batch_size,
                'verbose': verbose,
                'metrics': self.metrics_names,
                'epochs': epochs,
                'steps': steps_per_epoch,
                'samples': samples
            }

            callbacks.set_model(self)
            callbacks.set_params(params)

        callbacks.on_train_begin()

        epoch_counter = count()

        epoch_id = next(epoch_counter)

        did_abort = False

        batch_logs = dict.fromkeys(self.metrics_names, 0)

        try:
            while epoch_id < epochs:

                # Enables different behaviours for some layers, such as Batchnorm, dropout...
                tf.keras.backend.set_learning_phase(1)

                self.reset_metrics()

                batch_counter = count()

                batch_id = next(batch_counter)

                callbacks.on_epoch_begin(epoch_id)

                for data in dataset_iterator:

                    callbacks.on_batch_begin(batch_id)

                    data.update(class_weight=class_weight)

                    metrics = self.update_step(**data)

                    if tf.equal(tf.rank(metrics), 0):
                        # In case of metrics set no None in compile method
                        metrics = [metrics]

                    batch_logs.update(zip(batch_logs, metrics))
                    batch_logs['size'] = batch_size

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

        if not self.compiled:
            raise RuntimeError(
                "Your tried to fit your agent but it hasn\'t been compiled yet. Please call 'compile()' before 'test()'.")

        callbacks = callbacks or []

        if not isinstance(callbacks, callbacks_module.CallbackList):
            self.history = tf.keras.callbacks.History()
            callbacks.append(self.history)

            callbacks = callbacks_module.CallbackList(callbacks)

            params = {
                'batch_size': batch_size,
                'verbose': verbose,
                'metrics': self.metrics_names,
                'epochs': 1,
                'steps': steps
            }

            callbacks.set_model(self)
            callbacks.set_params(params)

        batch_logs = dict.fromkeys(self.metrics_names, 0)

        callbacks.on_test_begin()

        self.reset_metrics()

        batch_counter = count()

        batch_id = next(batch_counter)

        for data in dataset_iterator:

            callbacks.on_test_batch_begin(batch_id)

            metrics = self.predict_step(**data)

            if tf.equal(tf.rank(metrics), 0):
                metrics = [metrics]

            batch_logs.update(zip(batch_logs, metrics))
            batch_logs['size'] = batch_size

            callbacks.on_test_batch_end(batch_id, logs=batch_logs)

            batch_id = next(batch_counter)

        callbacks.on_test_end()

        batch_logs.pop('size')
        logs = to_numpy_or_python_type(batch_logs)
        if return_dict:
            return logs
        else:
            results = [logs.get(name, None) for name in self.metrics_names]
            if len(results) == 1:
                return results[0]

            return results

    def update_step(self, **data_dict):
        raise NotImplementedError()

    def predict_step(self, **data_dict):
        raise NotImplementedError()

    def build_model(self):
        raise NotImplementedError()

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    def init_graph(self, **kwargs):

        kwargs = self.build_model()

        kwargs.update(trainable=self.trainable,
                      name=self.name)

        # Graph network
        self._init_graph_network(**kwargs)

        tf_utils.assert_no_legacy_layers(self.layers)

    def get_config(self):
        if not self.built:
            raise Exception(f"You must call 'init_graph()' or 'compile()' methods before calling 'get_config()'")

        return super().get_config()

    def _should_eval(self, epoch, validation_freq):
        epoch = epoch + 1  # one-index the user-facing epoch.
        if isinstance(validation_freq, int):
            return epoch % validation_freq == 0
        elif isinstance(validation_freq, list):
            return epoch in validation_freq
        else:
            raise ValueError('Expected `validation_freq` to be a list or int.')
