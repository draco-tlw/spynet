from pathlib import Path


def get_paths(data_dir: Path, split_file_path: Path):
    with open(split_file_path, "r") as f:
        split_flags = [int(line.strip()) for line in f.readlines()]

    train_lists: tuple[list[str], list[str], list[str]] = ([], [], [])
    val_lists: tuple[list[str], list[str], list[str]] = ([], [], [])

    # Flying Chairs uses 1-based indexing (00001, 00002, etc.)
    for i, flag in enumerate(split_flags):
        idx_str = f"{i + 1:05d}"

        i1 = data_dir / f"{idx_str}_img1.ppm"
        i2 = data_dir / f"{idx_str}_img2.ppm"
        flo = data_dir / f"{idx_str}_flow.flo"

        # 1 = Train, 2 = Val
        if flag == 1:
            train_lists[0].append(str(i1))
            train_lists[1].append(str(i2))
            train_lists[2].append(str(flo))
        elif flag == 2:
            val_lists[0].append(str(i1))
            val_lists[1].append(str(i2))
            val_lists[2].append(str(flo))

    print(f"Loaded {len(train_lists[0])} Training samples.")
    print(f"Loaded {len(val_lists[0])} Validation samples.")

    return train_lists, val_lists
