from pathlib import Path

import keras
import tensorflow as tf

import src.spynet.model.losses as losses
import src.spynet.model.utils as utils
from src.spynet.model.callbacks import Checkpoint, GradualDecayScheduler
from src.spynet.model.level import SPyNetLevel


@keras.saving.register_keras_serializable(package="SPyNetPackage")
class SPyNet(keras.Model):
    _g: list[keras.Model]
    _max_k: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._g0 = SPyNetLevel()
        self._g1 = SPyNetLevel()
        self._g2 = SPyNetLevel()
        self._g3 = SPyNetLevel()
        self._g4 = SPyNetLevel()

        self._g = [self._g0, self._g1, self._g2, self._g3, self._g4]
        self._max_k = len(self._g) - 1

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], training=False):
        I1, I2 = inputs

        pyramid_I1 = [I1]
        pyramid_I2 = [I2]

        for _ in range(self._max_k):
            pyramid_I1.insert(0, utils.downscale(pyramid_I1[0]))
            pyramid_I2.insert(0, utils.downscale(pyramid_I2[0]))

        shape = tf.shape(pyramid_I1[0])
        V_prev_upscaled = tf.zeros([shape[0], shape[1], shape[2], 2], tf.float32)

        for k in range(self._max_k + 1):
            I1_k = pyramid_I1[k]
            I2_k = pyramid_I2[k]

            v_k = self._g[k]((I1_k, I2_k, V_prev_upscaled), training=training)

            V_k = v_k + V_prev_upscaled

            if k < self._max_k:
                V_prev_upscaled = utils.upscale(V_k, is_flow=True)
            else:
                return V_k

    def train_step(self, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        I1, I2, V_hat = data

        x = I1, I2
        y = V_hat

        return super().train_step((x, y))

    def _compile_levels(self, learning_rate=1e-4, beta_1=0.9, beta_2=0.999):
        for g_k in self._g:
            optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, clipnorm=1.0
            )
            g_k.compile(
                optimizer=optimizer, loss="mse", metrics=[losses.endpoint_error]
            )

    @tf.function
    def _compute_pyramids(self, I1: tf.Tensor, I2: tf.Tensor, V_hat: tf.Tensor):
        pyramid_I1 = [I1]
        pyramid_I2 = [I2]
        pyramid_V_hat = [V_hat]

        for _ in range(self._max_k):
            pyramid_I1.insert(0, utils.downscale(pyramid_I1[0]))
            pyramid_I2.insert(0, utils.downscale(pyramid_I2[0]))
            pyramid_V_hat.insert(0, utils.downscale(pyramid_V_hat[0], is_flow=True))

        return tuple(pyramid_I1), tuple(pyramid_I2), tuple(pyramid_V_hat)

    @tf.function
    def _prepare_level_k_data(
        self,
        k: int,
        pyramid_I1: list[tf.Tensor],
        pyramid_I2: list[tf.Tensor],
        pyramid_V_hat: list[tf.Tensor],
    ):

        I1_k = pyramid_I1[k]
        I2_k = pyramid_I2[k]
        V_hat_k = pyramid_V_hat[k]

        shape = tf.shape(pyramid_I1[0])
        V_prev_upscaled = tf.zeros([shape[0], shape[1], shape[2], 2], tf.float32)

        for i in range(k):
            x = pyramid_I1[i], pyramid_I2[i], V_prev_upscaled
            v_i = self._g[i](x, training=False)
            V_i = v_i + V_prev_upscaled
            V_prev_upscaled = utils.upscale(V_i, is_flow=True)

        x = I1_k, I2_k, V_prev_upscaled
        y = V_hat_k - V_prev_upscaled

        return x, y

    def _create_level_dataset(self, k: int, pyramid_data: tf.data.Dataset):
        sample_x = None
        sample_y = None

        for batch in pyramid_data.take(1):
            p_i1, p_i2, p_v = batch
            sample_x, sample_y = self._prepare_level_k_data(k, p_i1, p_i2, p_v)

        if sample_x is None or sample_y is None:
            raise ValueError("Dataset is empty. Cannot extract tensor specifications.")

        x_spec = tuple(tf.TensorSpec(t.shape, t.dtype) for t in sample_x)
        y_spec = tf.TensorSpec(sample_y.shape, sample_y.dtype)

        def gen():
            for batch in pyramid_data:
                p_i1, p_i2, p_v = batch
                yield self._prepare_level_k_data(k, p_i1, p_i2, p_v)

        ds = tf.data.Dataset.from_generator(gen, output_signature=(x_spec, y_spec))

        ds = ds.apply(
            tf.data.experimental.assert_cardinality(pyramid_data.cardinality())
        )

        return ds.prefetch(tf.data.AUTOTUNE)

    def train_sequential(
        self,
        project_dir: Path,
        data: tf.data.Dataset,
        val_data: tf.data.Dataset | None = None,
        epochs: int = 120,
        lr0=1e-4,
        lrf=1e-1,
        lr_decay_epoch=60,
        beta_1=0.9,
        beta_2=0.999,
        early_stopping_patience: int | None = 20,
        start_level=0,
        start_epoch=0,
    ):
        self._compile_levels(learning_rate=lr0, beta_1=beta_1, beta_2=beta_2)

        project_dir.mkdir(parents=True, exist_ok=True)

        print("[INFO] Building master model architecture for saving...")
        for batch in data.take(1):
            if len(batch) == 3:
                img1, img2, _ = batch
            else:
                (img1, img2), _ = batch

            self([img1, img2], training=False)

        pyramid_data = data.map(
            self._compute_pyramids, num_parallel_calls=tf.data.AUTOTUNE
        )

        pyramid_val_data = None
        if val_data is not None:
            pyramid_val_data = val_data.map(
                self._compute_pyramids, num_parallel_calls=tf.data.AUTOTUNE
            )

        for prior_k in range(start_level):
            self._g[prior_k].trainable = False

        hists = []

        for k in range(start_level, self._max_k + 1):
            print(f"--- Training Level {k} ---")

            current_initial_epoch = start_epoch if k == start_level else 0

            level_data = self._create_level_dataset(k, pyramid_data)

            level_val_data = None
            if val_data is not None and pyramid_val_data is not None:
                level_val_data = self._create_level_dataset(k, pyramid_val_data)

            if k > 0 and current_initial_epoch == 0:
                for x, _ in level_data.take(1):
                    self._g[k](x, training=False)

                self._g[k].set_weights(self._g[k - 1].get_weights())

            callbacks = [
                GradualDecayScheduler(
                    lr0=lr0,
                    lrf=lrf,
                    start_decay_epoch=lr_decay_epoch,
                    total_epochs=epochs,
                ),
                Checkpoint(
                    self, current_level=k, total_epochs=epochs, project_dir=project_dir
                ),
                keras.callbacks.CSVLogger(
                    project_dir / f"results_level_{k}.csv", append=True
                ),
            ]

            if val_data is not None and early_stopping_patience is not None:
                callbacks.append(
                    keras.callbacks.EarlyStopping(
                        monitor="val_endpoint_error",
                        patience=early_stopping_patience,
                        restore_best_weights=True,
                        mode="min",
                    )
                )

            hist = self._g[k].fit(
                level_data,
                validation_data=level_val_data,
                epochs=epochs,
                initial_epoch=current_initial_epoch,
                callbacks=callbacks,
            )
            hists.append(hist)

            self._g[k].trainable = False

        return hists
