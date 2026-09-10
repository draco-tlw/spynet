from functools import partial

import keras
import tensorflow as tf

import src.spynet.model.utils as utils


class SPyNetLevel(keras.Model):
    _model: keras.Model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        DefaultConv2D = partial(
            keras.layers.Conv2D,
            kernel_size=7,
            padding="same",
            activation="relu",
        )

        self._model = keras.models.Sequential(
            [
                DefaultConv2D(filters=32),
                DefaultConv2D(filters=64),
                DefaultConv2D(filters=32),
                DefaultConv2D(filters=16),
                DefaultConv2D(filters=2, activation="linear"),
            ]
        )

    @tf.function
    def _prepare_x(
        self,
        I1_k: tf.Tensor,
        I2_k: tf.Tensor,
        V_k_prev_upscaled: tf.Tensor,
    ):
        warped = utils.warp(I2_k, V_k_prev_upscaled)

        x = tf.concat([I1_k, warped, V_k_prev_upscaled], axis=-1)

        return x

    @tf.function
    def _prepare_y(
        self,
        V_k_prev_upscaled: tf.Tensor,
        V_k_hat: tf.Tensor,
    ):
        y = V_k_hat - V_k_prev_upscaled

        return y

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], training=False):
        I1_k, I2_k, V_k_prev_upscaled = inputs
        x = self._prepare_x(I1_k, I2_k, V_k_prev_upscaled)

        return self._model(x, training=training)

    def train_step(
        self, data: tuple[tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]
    ):
        x, y = data
        _, _, V_k_prev_upscaled = x
        V_k_hat = y

        y = self._prepare_y(V_k_prev_upscaled, V_k_hat)

        return super().train_step((x, y))
