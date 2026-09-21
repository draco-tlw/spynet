import json
from pathlib import Path

import keras
import numpy as np


class SPyNetSequentialSchedule(keras.callbacks.LearningRateScheduler):
    def __init__(
        self,
        initial_lr: float,
        decay_factor: float,
        decay_epoch: int,
    ):
        self.initial_lr = initial_lr
        self.final_lr = initial_lr * decay_factor
        self.decay_epoch = decay_epoch

        super().__init__(schedule=self._calculate_lr)

    def _calculate_lr(self, epoch: int, current_lr: float) -> float:
        if epoch < self.decay_epoch:
            return self.initial_lr
        else:
            return self.final_lr


class Checkpoint(keras.callbacks.Callback):
    def __init__(
        self,
        master_model: keras.Model,
        current_level: int,
        total_epochs: int,
        project_dir: Path,
        best_val_error: float = np.inf,
    ):
        super().__init__()
        self.master_model = master_model
        self.current_level = current_level
        self.total_epochs = total_epochs

        self.project_dir = project_dir
        self.weights_dir = self.project_dir / "weights"
        self.state_file = self.project_dir / "training_state.json"

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_error = best_val_error

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        last_path = self.weights_dir / "last.keras"
        self.master_model.save(last_path)

        current_val_error = logs.get("val_endpoint_error", np.inf)
        if current_val_error < self.best_val_error:
            self.best_val_error = current_val_error
            best_path = self.weights_dir / "best.keras"
            self.master_model.save(best_path)

        next_epoch = epoch + 1
        next_level = self.current_level

        if next_epoch >= self.total_epochs:
            next_epoch = 0
            next_level += 1
            self.best_val_error = None

        state = {
            "level": next_level,
            "epoch": next_epoch,
            "best_epe": (
                float(self.best_val_error) if self.best_val_error is not None else None
            ),
        }
        self.state_file.write_text(json.dumps(state, indent=4))
