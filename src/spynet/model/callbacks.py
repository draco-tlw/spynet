import json
from pathlib import Path

import keras
import numpy as np


class GradualDecayScheduler(keras.callbacks.LearningRateScheduler):

    def __init__(
        self,
        lr0: float,
        lrf: float,
        start_decay_epoch: int,
        total_epochs: int,
    ):
        self.lr0 = lr0
        self.final_lr = lr0 * lrf
        self.start_decay_epoch = start_decay_epoch
        self.total_epochs = total_epochs

        super().__init__(schedule=self._calculate_lr)

    def _calculate_lr(self, epoch: int, current_lr: float) -> float:
        if epoch < self.start_decay_epoch:
            return self.lr0

        decay_duration = max(1, self.total_epochs - 1 - self.start_decay_epoch)
        steps_into_decay = epoch - self.start_decay_epoch

        progress = steps_into_decay / decay_duration

        new_lr = self.lr0 - (progress * (self.lr0 - self.final_lr))
        return max(self.final_lr, new_lr)


class Checkpoint(keras.callbacks.Callback):
    def __init__(
        self,
        master_model,
        current_level: int,
        total_epochs: int,
        project_dir: Path,
    ):
        super().__init__()
        self.master_model = master_model
        self.current_level = current_level
        self.total_epochs = total_epochs

        self.project_dir = project_dir
        self.weights_dir = self.project_dir / "weights"
        self.state_file = self.project_dir / "training_state.json"

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_error = np.inf

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

        state = {"level": next_level, "epoch": next_epoch}
        self.state_file.write_text(json.dumps(state, indent=4))
