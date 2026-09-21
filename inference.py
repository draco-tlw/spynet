import math
import random
import typing
from pathlib import Path

import cv2
import keras
import numpy as np

import src.spynet.data.flying_chairs as flying_chairs
import src.spynet.data.loader as loader
import src.spynet.utils as utils
from src.spynet.model.master import SPyNet

PROJECT_DIR = Path("runs/flying_chairs/poc")
MODEL_FILE = PROJECT_DIR / "weights" / "last.keras"

DATA_DIR = Path("data/FlyingChairs/data")
SPLIT_FILE = Path("data/FlyingChairs/FlyingChairs_train_val.txt")

BATCH_SIZE = 4

print("[INFO] Loading Model...")
spynet = typing.cast(SPyNet, keras.models.load_model(MODEL_FILE, compile=False))

print("[INFO] Fetching Random Validation Samples...")
_, val_paths = flying_chairs.get_paths(DATA_DIR, SPLIT_FILE)
i1_paths, i2_paths, flow_paths = val_paths

idx = random.sample(range(len(i1_paths)), BATCH_SIZE)
i1 = [i1_paths[i] for i in idx]
i2 = [i2_paths[i] for i in idx]
flows = [flow_paths[i] for i in idx]

infer_ds = loader.create_dataset(i1, i2, flows, batch_size=BATCH_SIZE, shuffle=False)

batch_i1, batch_i2, batch_gt_flows, batch_valid = next(iter(infer_ds))

print("[INFO] Predicting Optical Flow...")
batch_pred_flows = spynet((batch_i1, batch_i2), training=False)

print("[INFO] Building OpenCV Grid...")

visualization_blocks = []

for i in range(BATCH_SIZE):
    img = batch_i1[i].numpy()  # RGB in [0, 1]
    gt_flow = batch_gt_flows[i].numpy()  # (H, W, 2)
    pred_flow = batch_pred_flows[i].numpy()  # (H, W, 2)
    valid_mask = batch_valid[i].numpy()  # (H, W, 1)

    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    epe_map = np.linalg.norm(pred_flow - gt_flow, axis=-1, keepdims=True)
    mean_epe = np.mean(epe_map[valid_mask == 1.0])

    gt_color = utils.flow_to_color(gt_flow)
    pred_color = utils.flow_to_color(pred_flow)

    cv2.putText(
        gt_color,
        "Ground Truth",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        pred_color,
        f"Prediction | EPE: {mean_epe:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    block = cv2.hconcat([img_bgr, gt_color, pred_color])

    # block = cv2.resize(block, (0, 0), fx=0.5, fy=0.5)
    visualization_blocks.append(block)

cols = math.ceil(math.sqrt(BATCH_SIZE))
rows = math.ceil(BATCH_SIZE / cols)

grid_rows = []
block_h, block_w, _ = visualization_blocks[0].shape

for r in range(rows):
    current_row_blocks = []
    for c in range(cols):
        idx = r * cols + c
        if idx < BATCH_SIZE:
            current_row_blocks.append(visualization_blocks[idx])
        else:
            blank_block = np.zeros((block_h, block_w, 3), dtype=np.uint8)
            current_row_blocks.append(blank_block)

    grid_rows.append(cv2.hconcat(current_row_blocks))

final_grid = cv2.vconcat(grid_rows)

print("[INFO] Displaying Output. Press 'q' or any key to close the window.")
cv2.imshow("SPyNet Inference", final_grid)

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q") or key == 27:  # q / ESC -> quit
        break
cv2.destroyAllWindows()
