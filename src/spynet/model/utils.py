import tensorflow as tf


def downscale(tensor: tf.Tensor, is_flow: bool = False):
    shape = tf.shape(tensor)

    h, w = shape[1] // 2, shape[2] // 2

    resized = tf.image.resize(tensor, [h, w], method="bilinear")

    return resized * 0.5 if is_flow else resized


def upscale(tensor: tf.Tensor, is_flow: bool = False):
    shape = tf.shape(tensor)

    h, w = shape[1] * 2, shape[2] * 2

    resized = tf.image.resize(tensor, [h, w], method="bilinear")

    return resized * 2 if is_flow else resized


def warp(image: tf.Tensor, flow: tf.Tensor):
    shape = tf.shape(image)
    bs, h, w = shape[0], shape[1], shape[2]

    y_grid, x_grid = tf.meshgrid(tf.range(h), tf.range(w), indexing="ij")
    y_grid = tf.cast(tf.broadcast_to(y_grid, [bs, h, w]), tf.float32)
    x_grid = tf.cast(tf.broadcast_to(x_grid, [bs, h, w]), tf.float32)

    u, v = tf.unstack(flow, axis=-1)

    x_source = x_grid + u
    y_source = y_grid + v

    x_max = tf.cast(w - 1, tf.float32)
    y_max = tf.cast(h - 1, tf.float32)
    x_source = tf.clip_by_value(x_source, 0.0, x_max)
    y_source = tf.clip_by_value(y_source, 0.0, y_max)

    x0 = tf.floor(x_source)
    x1 = tf.clip_by_value(x0 + 1.0, 0.0, x_max)
    x0 = tf.clip_by_value(x0, 0.0, x_max)

    y0 = tf.floor(y_source)
    y1 = tf.clip_by_value(y0 + 1.0, 0.0, y_max)
    y0 = tf.clip_by_value(y0, 0.0, y_max)

    w_x1 = x_source - x0
    w_x0 = 1.0 - w_x1
    w_y1 = y_source - y0
    w_y0 = 1.0 - w_y1

    b_idx = tf.broadcast_to(tf.range(bs)[:, tf.newaxis, tf.newaxis], [bs, h, w])

    idx00 = tf.stack([b_idx, tf.cast(y0, tf.int32), tf.cast(x0, tf.int32)], axis=-1)
    idx01 = tf.stack([b_idx, tf.cast(y0, tf.int32), tf.cast(x1, tf.int32)], axis=-1)
    idx10 = tf.stack([b_idx, tf.cast(y1, tf.int32), tf.cast(x0, tf.int32)], axis=-1)
    idx11 = tf.stack([b_idx, tf.cast(y1, tf.int32), tf.cast(x1, tf.int32)], axis=-1)

    i00 = tf.gather_nd(image, idx00)
    i01 = tf.gather_nd(image, idx01)
    i10 = tf.gather_nd(image, idx10)
    i11 = tf.gather_nd(image, idx11)

    w00 = tf.expand_dims(w_y0 * w_x0, axis=-1)
    w01 = tf.expand_dims(w_y0 * w_x1, axis=-1)
    w10 = tf.expand_dims(w_y1 * w_x0, axis=-1)
    w11 = tf.expand_dims(w_y1 * w_x1, axis=-1)

    return (i00 * w00) + (i01 * w01) + (i10 * w10) + (i11 * w11)
