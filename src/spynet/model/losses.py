import keras
import tensorflow as tf


def _epe_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.norm(y_true - y_pred, ord="euclidean", axis=-1)


class EndpointError(keras.metrics.MeanMetricWrapper):
    """
    Computes the Mean Endpoint Error.
    Inheriting from MeanMetricWrapper ensures Keras automatically applies
    the dataset's sample_weight (valid mask) before calculating the mean.
    """

    def __init__(self, name="endpoint_error", **kwargs):
        super().__init__(fn=_epe_fn, name=name, **kwargs)
