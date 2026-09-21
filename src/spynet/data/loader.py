"""tf.data-friendly readers for optical-flow training/eval data.

Flow formats
    .flo  Middlebury / Sintel / FlyingChairs   "PIEH", int32 W, int32 H, float32 (u,v) interleaved
    .pfm  FlyingThings3D / Monkaa / Driving    "PF\\nW H\\nscale\\n" + float32 RGB, rows bottom-to-top
    .png  KITTI 2012/2015, HD1K                16-bit RGB = (u*64 + 2**15, v*64 + 2**15, valid)
Image formats
    .ppm  FlyingChairs (binary P6, 8-bit)      everything else via tf.io.decode_image

Returned conventions
    flow   float32 [H, W, 2]  (u, v) in pixels, u -> right, v -> down, frame 1 -> frame 2
    valid  float32 [H, W, 1]  1.0 where ground truth exists, else 0.0 (flow is 0 there)
    image  float32 [H, W, 3]  RGB in [0, 1]
"""

import tensorflow as tf

_UNKNOWN_FLOW_THRESH = 1e9  # Middlebury: |u| or |v| above this (or NaN) means "unknown"


def _decode_flo(file_bytes, path):
    tf.debugging.Assert(
        tf.strings.substr(file_bytes, 0, 4) == "PIEH",
        ["Not a .flo file (bad 'PIEH' tag):", path],
    )
    w = tf.io.decode_raw(tf.strings.substr(file_bytes, 4, 4), tf.int32)[0]
    h = tf.io.decode_raw(tf.strings.substr(file_bytes, 8, 4), tf.int32)[0]
    data = tf.io.decode_raw(tf.strings.substr(file_bytes, 12, -1), tf.float32)
    flow = tf.reshape(data, [h, w, 2])

    known = tf.reduce_all(
        tf.abs(flow) <= _UNKNOWN_FLOW_THRESH, axis=-1, keepdims=True
    )  # NaN -> False
    return tf.where(known, flow, 0.0), tf.cast(known, tf.float32)


def _decode_pfm(file_bytes, path):
    # 3 header lines: "PF" / "W H" / scale (sign = endianness), then the raster.
    parts = tf.strings.split(file_bytes, "\n", maxsplit=3)
    tf.debugging.Assert(
        tf.strings.strip(parts[0]) == "PF", ["Expected a 3-channel 'PF' .pfm:", path]
    )
    wh = tf.strings.to_number(tf.strings.split(tf.strings.strip(parts[1])), tf.int32)
    w, h = wh[0], wh[1]
    scale = tf.strings.to_number(tf.strings.strip(parts[2]), tf.float32)

    raster = parts[3]
    data = tf.cond(
        scale < 0,  # negative scale -> little-endian, positive -> big-endian
        lambda: tf.io.decode_raw(raster, tf.float32, little_endian=True),
        lambda: tf.io.decode_raw(raster, tf.float32, little_endian=False),
    )
    flow = tf.reshape(data, [h, w, 3])[:, :, :2]  # 3rd channel is unused
    flow = tf.image.flip_up_down(flow)  # PFM stores the bottom row first
    return flow, tf.ones_like(flow[..., :1])


def _decode_kitti_png(file_bytes):
    img = tf.cast(tf.io.decode_png(file_bytes, channels=3, dtype=tf.uint16), tf.float32)
    flow = (img[..., :2] - 32768.0) / 64.0
    valid = tf.cast(img[..., 2:3] > 0, tf.float32)
    return flow * valid, valid


@tf.function
def read_flow(filepath: tf.Tensor):
    file_bytes = tf.io.read_file(filepath)
    ext = tf.strings.lower(filepath)

    flow, valid = tf.case(
        [
            (
                tf.strings.regex_full_match(ext, r".*\.pfm"),
                lambda: _decode_pfm(file_bytes, filepath),
            ),
            (
                tf.strings.regex_full_match(ext, r".*\.png"),
                lambda: _decode_kitti_png(file_bytes),
            ),
        ],
        default=lambda: _decode_flo(file_bytes, filepath),
    )
    flow.set_shape([None, None, 2])
    valid.set_shape([None, None, 1])
    return flow, valid


def _decode_ppm(file_bytes, path):
    total_len = tf.strings.length(file_bytes)

    head_len = tf.minimum(1024, total_len)
    head = tf.strings.substr(file_bytes, 0, head_len)

    head_no_comments = tf.strings.regex_replace(head, r"#[^\n]*\n", " ")
    head_clean = tf.strings.regex_replace(head_no_comments, r"\s+", " ")
    head_clean = tf.strings.strip(head_clean)

    tokens = tf.strings.split(head_clean, " ")

    tf.debugging.Assert(
        tf.strings.regex_full_match(tokens[0], "P6"),
        ["Not a valid P6 PPM file (missing P6 tag):", path],
    )

    w = tf.strings.to_number(tokens[1], tf.int32)
    h = tf.strings.to_number(tokens[2], tf.int32)

    raster_size = w * h * 3

    tf.debugging.Assert(
        total_len >= raster_size,
        ["PPM file is truncated or corrupted. Missing bytes:", path],
    )

    pixel_bytes = tf.strings.substr(file_bytes, total_len - raster_size, raster_size)
    image = tf.io.decode_raw(pixel_bytes, out_type=tf.uint8)

    return tf.reshape(image, [h, w, 3])


@tf.function
def read_image(filepath: tf.Tensor) -> tf.Tensor:
    file_bytes = tf.io.read_file(filepath)
    is_ppm = tf.strings.regex_full_match(tf.strings.lower(filepath), r".*\.ppm")
    image = tf.cond(
        is_ppm,
        lambda: _decode_ppm(file_bytes, filepath),
        lambda: tf.io.decode_image(file_bytes, channels=3, expand_animations=False),
    )
    image.set_shape([None, None, 3])
    return tf.image.convert_image_dtype(image, tf.float32)  # uint8 -> [0, 1]


@tf.function
def load_sample(i1_path: tf.Tensor, i2_path: tf.Tensor, flow_path: tf.Tensor):
    flow, valid = read_flow(flow_path)
    return read_image(i1_path), read_image(i2_path), flow, valid


def create_dataset(
    i1_paths,
    i2_paths,
    flow_paths,
    batch_size=32,
    shuffle=True,
    augment_fn=None,
    prefetch=tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((i1_paths, i2_paths, flow_paths))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(i1_paths), reshuffle_each_iteration=True
        )

    dataset = dataset.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    if augment_fn is not None:
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(prefetch)

    return dataset
