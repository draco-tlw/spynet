import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.spynet.model.utils import downscale, upscale, warp


def create_checkerboard(h=128, w=128, square_size=16):
    """Creates a high-contrast checkerboard pattern."""
    y, x = np.ogrid[:h, :w]
    board = ((x // square_size) + (y // square_size)) % 2
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[board == 1] = [0.9, 0.9, 0.9]
    img[board == 0] = [0.1, 0.2, 0.4]
    return img


def test_visual_all():
    h, w = 128, 128
    dx, dy = 16.0, 16.0

    # 1. Frame 1: Full checkerboard
    f1_np = create_checkerboard(h, w, square_size=16)

    # 2. Frame 2: Shifted with strict cutoff (Black borders mimic off-screen data)
    f2_np = np.zeros_like(f1_np)

    # Calculate valid slice ranges for a down-right shift
    src_y, src_x = 0, 0
    dst_y, dst_x = int(dy), int(dx)
    crop_h, crop_w = h - int(dy), w - int(dx)

    # Paste the cropped image into the shifted position
    f2_np[dst_y : dst_y + crop_h, dst_x : dst_x + crop_w] = f1_np[
        src_y : src_y + crop_h, src_x : src_x + crop_w
    ]

    # Ground truth flow: points from Frame 1 to Frame 2 everywhere
    flow_np = np.zeros((1, h, w, 2), dtype=np.float32)
    flow_np[:, :, :, 0] = dx
    flow_np[:, :, :, 1] = dy

    f1 = tf.convert_to_tensor(f1_np[np.newaxis, ...], dtype=tf.float32)
    f2 = tf.convert_to_tensor(f2_np[np.newaxis, ...], dtype=tf.float32)
    flow = tf.convert_to_tensor(flow_np, dtype=tf.float32)

    # Run Functions
    warped_f2 = warp(f2, flow)
    diff = tf.abs(f1 - warped_f2)

    f1_down = downscale(f1, is_flow=False)
    f1_reconstructed = upscale(f1_down, is_flow=False)

    flow_down = downscale(flow, is_flow=True)
    flow_reconstructed = upscale(flow_down, is_flow=True)

    # 4. COMPREHENSIVE VISUALIZATION
    fig = plt.figure(figsize=(16, 9))

    # --- ROW 1: WARPING VERIFICATION ---
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(f1[0].numpy())
    ax1.set_title("Frame 1 (Reference)")
    ax1.axis("off")

    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(f2[0].numpy())
    ax2.set_title(f"Frame 2 (Shifted +{dx}, +{dy})\nBlack = Real Occlusion")
    ax2.axis("off")

    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(warped_f2[0].numpy())
    ax3.set_title("Warped Frame 2\n(Notice the smeared borders!)")
    ax3.axis("off")

    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(diff[0].numpy(), cmap="hot")
    ax4.set_title("Warp Absolute Error\n(Interior is 0, edges have errors)")
    ax4.axis("off")

    # --- ROW 2: UPSCALE / DOWNSCALE VERIFICATION ---
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(f1_down[0].numpy())
    ax5.set_title(f"Downscaled Image\n({f1_down.shape[1]}x{f1_down.shape[2]})")
    ax5.axis("off")

    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(f1_reconstructed[0].numpy())
    ax6.set_title(f"Upscaled Image\n(Softened edges)")
    ax6.axis("off")

    # Visualizing Flow Vectors
    step = 8
    y_q, x_q = np.mgrid[0:h:step, 0:w:step]

    ax7 = plt.subplot(2, 4, 7)
    u_orig = flow[0, ::step, ::step, 0].numpy()
    v_orig = flow[0, ::step, ::step, 1].numpy()
    ax7.quiver(
        x_q, y_q, u_orig, -v_orig, color="blue", angles="xy", scale_units="xy", scale=1
    )
    ax7.set_xlim(0, w)
    ax7.set_ylim(h, 0)
    ax7.set_title("Original Flow Vectors")
    ax7.set_aspect("equal")

    step_down = 4
    h_d, w_d = h // 2, w // 2
    y_qd, x_qd = np.mgrid[0:h_d:step_down, 0:w_d:step_down]
    u_d = flow_down[0, ::step_down, ::step_down, 0].numpy()
    v_d = flow_down[0, ::step_down, ::step_down, 1].numpy()

    ax8 = plt.subplot(2, 4, 8)
    ax8.quiver(
        x_qd, y_qd, u_d, -v_d, color="red", angles="xy", scale_units="xy", scale=1
    )
    ax8.set_xlim(0, w_d)
    ax8.set_ylim(h_d, 0)
    ax8.set_title("Downscaled Flow Vectors")
    ax8.set_aspect("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_visual_all()
