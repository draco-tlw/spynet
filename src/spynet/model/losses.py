import tensorflow as tf


def endpoint_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    epe_per_pixel = tf.norm(y_true - y_pred, ord="euclidean", axis=-1)

    return tf.reduce_mean(epe_per_pixel)
