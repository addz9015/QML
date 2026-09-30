"""Build clean cats-vs-dogs train + held-out test sets.

Source (not shipped in the repo - supply your own images, any folder depth):

    data/raw_full/class0/**.jpg      cats
    data/raw_full/class1/**.jpg      dogs

Output - flat, class-balanced, and disjoint (no image is in both splits):

    data/train/class0|class1/*.jpg   CFG.per_class images per class
    data/test/class0|class1/*.jpg    CFG.test_per_class images per class

Quality filters: folders named like "trimaps" (segmentation masks), unreadable
files, tiny images and near-uniform images (<= 8 distinct colours, i.e. masks)
are skipped. Candidates are taken round-robin across sub-folders (breeds) so no
breed dominates. The demo images data/test/cat.jpg and data/test/dog.jpg are left
untouched.

Run:  python -m src.build_dataset
"""

from __future__ import annotations

import os
import sys
import random

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CFG

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ("class0", "class1")
EXCLUDE_DIR_KEYWORDS = ("trimap",)
MIN_SIDE = 64
MIN_COLOURS = 9


def find_images(root: str) -> list[str]:
    out = []
    for dirpath, dirnames, names in os.walk(root):
        dirnames[:] = [d for d in dirnames if not any(k in d.lower() for k in EXCLUDE_DIR_KEYWORDS)]
        for n in names:
            if os.path.splitext(n)[1].lower() in IMG_EXTS:
                out.append(os.path.join(dirpath, n))
    return sorted(out)


def spread_order(paths: list[str], rng: random.Random) -> list[str]:
    """All paths, ordered round-robin across their parent folders (breeds)."""
    by_folder: dict[str, list[str]] = {}
    for p in paths:
        by_folder.setdefault(os.path.dirname(p), []).append(p)
    for v in by_folder.values():
        rng.shuffle(v)
    folders = sorted(by_folder)
    rng.shuffle(folders)
    order: list[str] = []
    while any(by_folder.values()):
        for f in folders:
            if by_folder[f]:
                order.append(by_folder[f].pop())
    return order


def load_photo(path: str):
    """Return an RGB PIL image, or None if the file is not a usable photo."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.load()
    except Exception:  # noqa: BLE001 - unreadable / corrupt
        return None
    if min(im.size) < MIN_SIDE:
        return None
    small = np.asarray(im.resize((64, 64)))
    if len(np.unique(small.reshape(-1, 3), axis=0)) < MIN_COLOURS:
        return None
    return im


def reset_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        if os.path.isfile(fp):
            os.remove(fp)


def main() -> None:
    rng = random.Random(CFG.seed)
    need = CFG.per_class + CFG.test_per_class

    for folder in CLASSES:
        raw_dir = os.path.join(CFG.raw_dir, folder)
        if not os.path.isdir(raw_dir):
            raise SystemExit(f"Missing {raw_dir}. Put cat images in data/raw_full/class0 "
                             "and dog images in data/raw_full/class1.")
        candidates = spread_order(find_images(raw_dir), rng)

        photos, skipped = [], 0
        for path in candidates:
            im = load_photo(path)
            if im is None:
                skipped += 1
                continue
            photos.append(im)
            if len(photos) == need:
                break
        if len(photos) < need:
            raise SystemExit(f"{folder}: only {len(photos)} usable images, need {need}")

        order = list(range(need))
        rng.shuffle(order)
        test_idx, train_idx = order[:CFG.test_per_class], order[CFG.test_per_class:]

        for split_dir, idxs in ((CFG.test_dir, test_idx), (CFG.train_dir, train_idx)):
            out_dir = os.path.join(split_dir, folder)
            reset_dir(out_dir)
            for n, i in enumerate(idxs):
                im = photos[i].copy()
                im.thumbnail((512, 512))
                im.save(os.path.join(out_dir, f"{folder}_{n:03d}.jpg"), "JPEG", quality=90)
        print(f"{folder}: {len(candidates)} candidates, {skipped} skipped as unusable "
              f"-> train {len(train_idx)}, test {len(test_idx)}")

    # Invalidate every cache derived from the images.
    for cache in (CFG.feature_cache, CFG.test_feature_cache):
        if os.path.exists(cache):
            os.remove(cache)
            print("Removed stale cache:", cache)
    if os.path.isdir(CFG.cache_dir):
        for name in os.listdir(CFG.cache_dir):
            os.remove(os.path.join(CFG.cache_dir, name))
        print("Cleared", CFG.cache_dir)


if __name__ == "__main__":
    main()
