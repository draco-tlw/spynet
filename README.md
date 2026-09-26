# SPyNet: Spatial Pyramid Network for Optical Flow

An implementation of the SPyNet (Spatial Pyramid Network) architecture for optical flow estimation, built using Keras 3 and TensorFlow.

This repository provides an end-to-end pipeline for training and inference, currently configured for the FlyingChairs dataset.

> **Reference:**
> [Optical Flow Estimation using a Spatial Pyramid Network](https://arxiv.org/abs/1611.00850) (Anurag Ranjan, Michael J. Black, CVPR 2017).

_Note: The model weights included in this repository are from a proof-of-concept run due to local GPU memory constraints. The training pipeline is fully functional for users intending to train the model to convergence._

## Repository Features

- **Sequential Level Training:** Implements the 5-level spatial pyramid architecture. The training script automatically manages the sequential freezing and unfreezing of pyramid levels ($L_0$ through $L_4$) as described in the original paper.
- **State Management:** Training states (current level, epoch, and best EPE metrics) are tracked via JSON. The training loop can be interrupted and resumed with continuity for patience counters and learning rate schedules.
- **Data Pipeline:** Utilizes `tf.data` for dynamic parsing, cropping, and batching of `.ppm` and `.flo` file formats.
- **Augmentation Strategy:** Includes photometric jitter (brightness, contrast, hue, saturation, noise) and geometric transformations (scaling, rotation, translation) applied during the `tf.data` map operations.
- **Inference Grid:** An OpenCV-based inference script evaluates model output against ground truth data, calculating Mean Endpoint Error (EPE) and rendering flow vectors using HSV color mapping.

## Dependencies and Setup

This project uses `uv` for environment and dependency management. Requires Python $\ge$ 3.13.

1.  **Clone and Sync:**

    ```bash
    git clone https://github.com/yourusername/spynet.git
    cd spynet
    uv sync
    ```

2.  **Data Preparation:**
    Download the FlyingChairs dataset. Extract the contents into the `data/` directory. The structure must match the following exactly:
    ```text
    data/
      FlyingChairs/
        data/
          00001_img1.ppm
          00001_img2.ppm
          00001_flow.flo
          ...
        FlyingChairs_train_val.txt
    ```

## Usage

### Training

To initiate or resume the training pipeline:

```bash
uv run train.py
```

- **Resumption:** If the script exits, running `train.py` again will load `training_state.json` and restore the `.keras` weights from the last successful epoch, bypassing completed levels.
- **Logging:** TensorBoard event files and CSV logs are written to `runs/flying_chairs/poc/logs/`.

### Inference

To evaluate a random subset of the validation split:

```bash
uv run inference.py
```

The inference script executes the following:

1.  Samples a batch of validation image pairs and their corresponding ground truth flows.
2.  Computes the predicted optical flow using the weights located at `runs/flying_chairs/poc/weights/last.keras`.
3.  Calculates the valid Mean EPE.
4.  Renders a 2D OpenCV comparison grid displaying the input image, ground truth flow, and predicted flow. (Press `q` or `ESC` to close the visualization window).
