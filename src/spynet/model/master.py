from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

import src.spynet.model.losses as losses
import src.spynet.model.utils as utils
from src.spynet.model.callbacks import Checkpoint, SPyNetSequentialSchedule
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

    def train_step(self, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]):
        I1, I2, V_hat, valid = data

        x = I1, I2
        y = V_hat

        return super().train_step((x, y, valid))

    def _compile_levels(self, learning_rate=1e-4, beta_1=0.9, beta_2=0.999):
        for g_k in self._g:
            optimizer = keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, clipnorm=1.0
            )
            g_k.compile(
                optimizer=optimizer, loss="mse", metrics=[losses.EndpointError()]
            )

    @tf.function
    def _compute_pyramids(
        self, I1: tf.Tensor, I2: tf.Tensor, V_hat: tf.Tensor, valid: tf.Tensor
    ):
        pyramid_I1 = [I1]
        pyramid_I2 = [I2]
        pyramid_V_hat = [V_hat]
        pyramid_valid = [valid]

        for _ in range(self._max_k):
            pyramid_I1.insert(0, utils.downscale(pyramid_I1[0]))
            pyramid_I2.insert(0, utils.downscale(pyramid_I2[0]))
            pyramid_V_hat.insert(0, utils.downscale(pyramid_V_hat[0], is_flow=True))
            pyramid_valid.insert(0, utils.downscale(pyramid_valid[0]))

        return (
            tuple(pyramid_I1),
            tuple(pyramid_I2),
            tuple(pyramid_V_hat),
            tuple(pyramid_valid),
        )

    @tf.function
    def _prepare_level_k_data(
        self,
        k: int,
        pyramid_I1: list[tf.Tensor],
        pyramid_I2: list[tf.Tensor],
        pyramid_V_hat: list[tf.Tensor],
        pyramid_valid: list[tf.Tensor],
    ):
        I1_k = pyramid_I1[k]
        I2_k = pyramid_I2[k]
        V_hat_k = pyramid_V_hat[k]
        valid_k = pyramid_valid[k]

        shape = tf.shape(pyramid_I1[0])
        V_prev_upscaled = tf.zeros([shape[0], shape[1], shape[2], 2], tf.float32)

        for i in range(k):
            x = pyramid_I1[i], pyramid_I2[i], V_prev_upscaled
            v_i = self._g[i](x, training=False)
            V_i = v_i + V_prev_upscaled
            V_prev_upscaled = utils.upscale(V_i, is_flow=True)

        x = I1_k, I2_k, V_prev_upscaled
        y = V_hat_k - V_prev_upscaled

        return x, y, valid_k

    def _create_level_dataset(self, k: int, pyramid_data: tf.data.Dataset):
        sample_x = None
        sample_y = None
        sample_w = None

        for batch in pyramid_data.take(1):
            p_i1, p_i2, p_v, p_valid = batch
            sample_x, sample_y, sample_w = self._prepare_level_k_data(
                k, p_i1, p_i2, p_v, p_valid
            )

        if sample_x is None or sample_y is None or sample_w is None:
            raise ValueError("Dataset is empty. Cannot extract tensor specifications.")

        x_spec = tuple(tf.TensorSpec(t.shape, t.dtype) for t in sample_x)
        y_spec = tf.TensorSpec(sample_y.shape, sample_y.dtype)
        w_spec = tf.TensorSpec(sample_w.shape, sample_w.dtype)

        def gen():
            for batch in pyramid_data:
                p_i1, p_i2, p_v, p_valid = batch
                yield self._prepare_level_k_data(k, p_i1, p_i2, p_v, p_valid)

        ds = tf.data.Dataset.from_generator(
            gen, output_signature=(x_spec, y_spec, w_spec)
        )

        ds = ds.apply(
            tf.data.experimental.assert_cardinality(pyramid_data.cardinality())
        )

        return ds.prefetch(tf.data.AUTOTUNE)

    def train_sequential(
        self,
        project_dir: Path,
        data: tf.data.Dataset,
        val_data: tf.data.Dataset | None = None,
        epochs: int = 200,
        initial_lr=1e-4,
        lr_decay_factor=1e-1,
        lr_decay_epoch=60,
        beta_1=0.9,
        beta_2=0.999,
        early_stopping_patience: int | None = 20,
        early_stopping_start_epoch: int | None = 80,
        # resume
        start_level=0,
        start_epoch=0,
        best_epe: float | None = None,
    ):
        self._compile_levels(learning_rate=initial_lr, beta_1=beta_1, beta_2=beta_2)

        project_dir.mkdir(parents=True, exist_ok=True)

        print("[INFO] Building master model architecture for saving...")
        for batch in data.take(1):
            if len(batch) != 4:
                raise ValueError(
                    f"Expected dataset to yield 4 items (img1, img2, flow, valid), "
                    f"but got {len(batch)} items."
                )

            img1, img2, _, _ = batch
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
                for x, _, _ in level_data.take(1):
                    self._g[k](x, training=False)

                self._g[k].set_weights(self._g[k - 1].get_weights())

            tensorboard_logs_path = project_dir / "logs"
            tensorboard_logs_path.mkdir(exist_ok=True)

            level_k_best_epe = (
                best_epe if (start_level == k and start_epoch > 0) else None
            )

            callbacks = [
                SPyNetSequentialSchedule(
                    initial_lr=initial_lr,
                    decay_factor=lr_decay_factor,
                    decay_epoch=lr_decay_epoch,
                ),
                Checkpoint(
                    self,
                    current_level=k,
                    total_epochs=epochs,
                    project_dir=project_dir,
                    best_val_error=(
                        float(level_k_best_epe)
                        if level_k_best_epe is not None
                        else np.inf
                    ),
                ),
                keras.callbacks.CSVLogger(
                    project_dir / f"results_level_{k}.csv", append=True
                ),
                keras.callbacks.TensorBoard(
                    log_dir=str(tensorboard_logs_path / f"level_{k}"), histogram_freq=1
                ),
            ]

            if (
                val_data is not None
                and early_stopping_patience is not None
                and early_stopping_start_epoch is not None
            ):
                callbacks.append(
                    keras.callbacks.EarlyStopping(
                        monitor="val_endpoint_error",
                        patience=early_stopping_patience,
                        start_from_epoch=early_stopping_start_epoch,
                        baseline=level_k_best_epe,
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

            best_model_path = project_dir / "weights" / "best.keras"
            if best_model_path.exists():
                print(
                    f"[INFO] Loading globally best weights from disk for Level {k}..."
                )
                self.load_weights(best_model_path)

            self._g[k].trainable = False

        return hists
