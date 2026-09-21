import math

import cv2
import numpy as np
import tensorflow as tf


def _geometric_augmentation(
    i1: np.ndarray,
    i2: np.ndarray,
    flow: np.ndarray,
    valid: np.ndarray,
    scale=(0.5, 2.0),
    angle=(-17.0, 17.0),
    translate_x=(-20.0, 20.0),
    translate_y=(-20.0, 20.0),
):
    h, w = i1.shape[:2]
    center = (w / 2.0, h / 2.0)

    scale = np.random.uniform(*scale)
    angle = np.random.uniform(*angle)
    tx = np.random.uniform(*translate_x)
    ty = np.random.uniform(*translate_y)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    M[0, 2] += tx
    M[1, 2] += ty

    i1_warped = cv2.warpAffine(
        i1,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    i2_warped = cv2.warpAffine(
        i2,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    flow_warped = cv2.warpAffine(
        flow,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_warped = cv2.warpAffine(
        valid,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    A = M[:, :2]

    flow_flat = flow_warped.reshape(-1, 2)
    flow_corrected = np.dot(flow_flat, A.T)
    flow_warped = flow_corrected.reshape(h, w, 2)

    if len(valid_warped.shape) == 2:
        valid_warped = np.expand_dims(valid_warped, axis=-1)

    return (
        i1_warped.astype(np.float32),
        i2_warped.astype(np.float32),
        flow_warped.astype(np.float32),
        valid_warped.astype(np.float32),
    )


@tf.function
def _tf_geometric_augmentation(
    i1: tf.Tensor,
    i2: tf.Tensor,
    flow: tf.Tensor,
    valid: tf.Tensor,
    scale_range=(0.5, 2.0),
    angle_range=(-17.0, 17.0),
    tx_range=(-20.0, 20.0),
    ty_range=(-20.0, 20.0),
):
    shape = tf.shape(i1)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    cx = w / 2.0
    cy = h / 2.0

    scale = tf.random.uniform([], scale_range[0], scale_range[1], dtype=tf.float32)
    angle_deg = tf.random.uniform([], angle_range[0], angle_range[1], dtype=tf.float32)
    tx = tf.random.uniform([], tx_range[0], tx_range[1], dtype=tf.float32)
    ty = tf.random.uniform([], ty_range[0], ty_range[1], dtype=tf.float32)

    angle_rad = angle_deg * (math.pi / 180.0)
    alpha = scale * tf.cos(angle_rad)
    beta = scale * tf.sin(angle_rad)

    m00 = alpha
    m01 = beta
    m02 = (1.0 - alpha) * cx - beta * cy + tx

    m10 = -beta
    m11 = alpha
    m12 = beta * cx + (1.0 - alpha) * cy + ty

    M_forward = tf.convert_to_tensor(
        [[m00, m01, m02], [m10, m11, m12], [0.0, 0.0, 1.0]], dtype=tf.float32
    )

    M_inverse = tf.linalg.inv(M_forward)
    transform_params = tf.reshape(M_inverse, [-1])[:8]
    transform_batch = tf.expand_dims(transform_params, 0)
    out_shape = tf.convert_to_tensor([shape[0], shape[1]], dtype=tf.int32)

    i1_w = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(i1, 0),
        transforms=transform_batch,
        output_shape=out_shape,
        fill_mode="CONSTANT",
        interpolation="BILINEAR",
    )[0]

    i2_w = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(i2, 0),
        transforms=transform_batch,
        output_shape=out_shape,
        fill_mode="CONSTANT",
        interpolation="BILINEAR",
    )[0]

    flow_w = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(flow, 0),
        transforms=transform_batch,
        output_shape=out_shape,
        fill_mode="CONSTANT",
        interpolation="BILINEAR",
    )[0]

    valid_w = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(valid, 0),
        transforms=transform_batch,
        output_shape=out_shape,
        fill_mode="CONSTANT",
        interpolation="NEAREST",
    )[0]

    A = tf.convert_to_tensor([[m00, m01], [m10, m11]], dtype=tf.float32)
    flow_corrected = tf.einsum("hwi,ji->hwj", flow_w, A)

    return i1_w, i2_w, flow_corrected, valid_w


@tf.function
def _apply_photometric_jitter(
    i1: tf.Tensor,
    i2: tf.Tensor,
    brightness=(-0.2, 0.2),
    contrast=(0.8, 1.2),
    hue=(-0.1, 0.1),
    saturation=(0.8, 1.2),
    gaussian_noise_mean=0.0,
    gaussian_noise_stddev=0.04,
):
    # 1. Random Brightness
    if tf.random.uniform([]) > 0.5:
        delta = tf.random.uniform([], *brightness)
        i1 = i1 + delta
        i2 = i2 + delta

    # 2. Random Contrast
    if tf.random.uniform([]) > 0.5:
        factor = tf.random.uniform([], *contrast)
        i1 = tf.image.adjust_contrast(i1, factor)
        i2 = tf.image.adjust_contrast(i2, factor)

    # 3. Color Shifts (Hue & Saturation)
    if tf.random.uniform([]) > 0.5:
        hue_delta = tf.random.uniform([], *hue)
        sat_factor = tf.random.uniform([], *saturation)

        i1 = tf.image.adjust_hue(i1, hue_delta)
        i1 = tf.image.adjust_saturation(i1, sat_factor)

        i2 = tf.image.adjust_hue(i2, hue_delta)
        i2 = tf.image.adjust_saturation(i2, sat_factor)

    # 4. Additive Gaussian Noise
    if tf.random.uniform([]) > 0.5:
        i1 = i1 + tf.random.normal(
            shape=tf.shape(i1), mean=gaussian_noise_mean, stddev=gaussian_noise_stddev
        )
        i2 = i2 + tf.random.normal(
            shape=tf.shape(i2), mean=gaussian_noise_mean, stddev=gaussian_noise_stddev
        )

    i1 = tf.clip_by_value(i1, 0.0, 1.0)
    i2 = tf.clip_by_value(i2, 0.0, 1.0)

    return i1, i2


@tf.function
def augment_sample(
    i1: tf.Tensor,
    i2: tf.Tensor,
    flow: tf.Tensor,
    valid: tf.Tensor,
    scale=(0.5, 2.0),
    angle=(-17.0, 17.0),
    translate_x=(-20.0, 20.0),
    translate_y=(-20.0, 20.0),
    brightness=(-0.2, 0.2),
    contrast=(0.8, 1.2),
    hue=(-0.1, 0.1),
    saturation=(0.8, 1.2),
    gaussian_noise_mean=0.0,
    gaussian_noise_stddev=0.04,
):

    i1, i2 = _apply_photometric_jitter(
        i1,
        i2,
        brightness,
        contrast,
        hue,
        saturation,
        gaussian_noise_mean,
        gaussian_noise_stddev,
    )

    # Random Horizontal Flip
    if tf.random.uniform([]) > 0.5:
        i1 = tf.image.flip_left_right(i1)
        i2 = tf.image.flip_left_right(i2)
        flow = tf.image.flip_left_right(flow)
        valid = tf.image.flip_left_right(valid)
        flow = flow * tf.constant([-1.0, 1.0], dtype=tf.float32)

    # Random Vertical Flip
    if tf.random.uniform([]) > 0.5:
        i1 = tf.image.flip_up_down(i1)
        i2 = tf.image.flip_up_down(i2)
        flow = tf.image.flip_up_down(flow)
        valid = tf.image.flip_up_down(valid)
        flow = flow * tf.constant([1.0, -1.0], dtype=tf.float32)

    # Random Affine Transforms
    if tf.random.uniform([]) > 0.5:
        i1, i2, flow, valid = _tf_geometric_augmentation(
            i1, i2, flow, valid, scale, angle, translate_x, translate_y
        )

    return i1, i2, flow, valid
