import json
from pathlib import Path

import keras

from spynet.data.augmentation import augment_sample
from src.spynet.data.loader import create_dataset, get_official_splits
from src.spynet.model.master import SPyNet

SEED = 42
keras.utils.set_random_seed(SEED)

data_dir = "data/FlyingChairs/data"
split_file = "data/FlyingChairs/FlyingChairs_train_val.txt"

train_lists, val_lists = get_official_splits(data_dir, split_file)

train_dataset = create_dataset(
    *train_lists, batch_size=8, shuffle=True, augment_fn=augment_sample, prefetch=2
)
val_dataset = create_dataset(*val_lists, batch_size=8, shuffle=False, prefetch=2)

project_dir = Path("runs/flying_chairs/base_paper_params")
model_file = project_dir / "weights" / "last.keras"
state_file = project_dir / "training_state.json"

start_level = 0
start_epoch = 0

if model_file.exists() and state_file.exists():
    state = json.loads(state_file.read_text())
    start_level = state["level"]
    start_epoch = state["epoch"]
    print(
        f"\n[INFO] Resuming {project_dir} from Level {start_level}, Epoch {start_epoch}\n"
    )

    model = keras.models.load_model(model_file)
else:
    print(f"\n[INFO] Starting new run in {project_dir}\n")
    model = SPyNet()


model.train_sequential(data=train_dataset, val_data=val_dataset, project_dir=project_dir, start_level=start_level, start_epoch=start_epoch)  # type: ignore
