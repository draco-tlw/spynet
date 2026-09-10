import os

import tensorflow as tf


@tf.function
def read_flo(filepath: tf.Tensor) -> tf.Tensor:
    flo_bytes = tf.io.read_file(filepath)
    w_bytes = tf.strings.substr(flo_bytes, 4, 4)
    h_bytes = tf.strings.substr(flo_bytes, 8, 4)
    data_bytes = tf.strings.substr(flo_bytes, 12, -1)

    w = tf.io.decode_raw(w_bytes, out_type=tf.int32)[0]
    h = tf.io.decode_raw(h_bytes, out_type=tf.int32)[0]

    flow_data = tf.io.decode_raw(data_bytes, out_type=tf.float32)
    return tf.reshape(flow_data, [h, w, 2])


@tf.function
def read_ppm(filepath: tf.Tensor) -> tf.Tensor:
    file_bytes = tf.io.read_file(filepath)

    # Calculate the exact offset for the pixel data
    total_len = tf.strings.length(file_bytes)
    pixel_bytes = tf.strings.substr(file_bytes, total_len - 589824, 589824)

    # Decode the raw binary into a uint8 tensor
    image = tf.io.decode_raw(pixel_bytes, out_type=tf.uint8)
    image = tf.reshape(image, [384, 512, 3])

    # Convert to float32 [0.0, 1.0] for the neural network
    return tf.image.convert_image_dtype(image, tf.float32)


@tf.function
def load_sample(i1_path: tf.Tensor, i2_path: tf.Tensor, flo_path: tf.Tensor):
    i1 = read_ppm(i1_path)
    i2 = read_ppm(i2_path)
    flow = read_flo(flo_path)

    return i1, i2, flow


def get_official_splits(data_dir: str, split_file_path: str):
    with open(split_file_path, "r") as f:
        split_flags = [int(line.strip()) for line in f.readlines()]

    train_lists = ([], [], [])
    val_lists = ([], [], [])

    # Flying Chairs uses 1-based indexing (00001, 00002, etc.)
    for i, flag in enumerate(split_flags):
        idx_str = f"{i + 1:05d}"

        i1 = os.path.join(data_dir, f"{idx_str}_img1.ppm")
        i2 = os.path.join(data_dir, f"{idx_str}_img2.ppm")
        flo = os.path.join(data_dir, f"{idx_str}_flow.flo")

        # 1 = Train, 2 = Val
        if flag == 1:
            train_lists[0].append(i1)
            train_lists[1].append(i2)
            train_lists[2].append(flo)
        elif flag == 2:
            val_lists[0].append(i1)
            val_lists[1].append(i2)
            val_lists[2].append(flo)

    print(f"Loaded {len(train_lists[0])} Training samples.")
    print(f"Loaded {len(val_lists[0])} Validation samples.")

    return train_lists, val_lists


def create_dataset(
    i1_paths,
    i2_paths,
    flo_paths,
    batch_size=32,
    shuffle=True,
    augment_fn=None,
    prefetch=tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((i1_paths, i2_paths, flo_paths))

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
