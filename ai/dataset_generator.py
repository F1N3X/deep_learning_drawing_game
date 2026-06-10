import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ==========================================================
# CONFIG
# ==========================================================

NDJSON_DIR = "quickdraw_ndjson"
OUTPUT_DIR = "./AI/dataset"

IMG_SIZE = 64

LINE_THICKNESS_MIN = 1
LINE_THICKNESS_MAX = 3

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==========================================================
# QUICKDRAW -> IMAGE
# ==========================================================

def drawing_to_image(drawing):

    canvas = np.full(
        (IMG_SIZE, IMG_SIZE),
        255,
        dtype=np.uint8
    )

    all_x = []
    all_y = []

    for stroke in drawing:
        all_x.extend(stroke[0])
        all_y.extend(stroke[1])

    if len(all_x) == 0:
        return canvas

    xmin = min(all_x)
    xmax = max(all_x)

    ymin = min(all_y)
    ymax = max(all_y)

    width = max(xmax - xmin, 1)
    height = max(ymax - ymin, 1)

    padding = 4

    scale = min(
        (IMG_SIZE - 2 * padding) / width,
        (IMG_SIZE - 2 * padding) / height
    )

    offset_x = (
        IMG_SIZE - width * scale
    ) / 2

    offset_y = (
        IMG_SIZE - height * scale
    ) / 2

    thickness = random.randint(
        LINE_THICKNESS_MIN,
        LINE_THICKNESS_MAX
    )

    for stroke in drawing:

        xs = stroke[0]
        ys = stroke[1]

        for i in range(len(xs) - 1):

            x1 = int(
                (xs[i] - xmin) * scale
                + offset_x
            )

            y1 = int(
                (ys[i] - ymin) * scale
                + offset_y
            )

            x2 = int(
                (xs[i + 1] - xmin) * scale
                + offset_x
            )

            y2 = int(
                (ys[i + 1] - ymin) * scale
                + offset_y
            )

            cv2.line(
                canvas,
                (x1, y1),
                (x2, y2),
                0,
                thickness,
                cv2.LINE_AA
            )

    return canvas

# ==========================================================
# AUGMENTATIONS
# ==========================================================

def random_rotation(img):

    angle = random.uniform(-15, 15)

    h, w = img.shape

    M = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        angle,
        1.0
    )

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        borderValue=255
    )


def random_scale(img):

    scale = random.uniform(0.85, 1.15)

    h, w = img.shape

    M = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        0,
        scale
    )

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        borderValue=255
    )


def random_translation(img):

    h, w = img.shape

    tx = random.randint(-5, 5)
    ty = random.randint(-5, 5)

    M = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        borderValue=255
    )


def random_shear(img):

    h, w = img.shape

    shear = random.uniform(-0.15, 0.15)

    M = np.float32([
        [1, shear, 0],
        [0, 1, 0]
    ])

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        borderValue=255
    )


def random_morphology(img):

    kernel_size = random.choice([2, 3])

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    op = random.choice([
        "dilate",
        "erode",
        "none"
    ])

    if op == "dilate":
        img = cv2.dilate(
            img,
            kernel,
            iterations=1
        )

    elif op == "erode":
        img = cv2.erode(
            img,
            kernel,
            iterations=1
        )

    return img


def add_gaussian_noise(img):

    sigma = random.uniform(2, 8)

    noise = np.random.normal(
        0,
        sigma,
        img.shape
    )

    noisy = img.astype(np.float32) + noise

    return np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)


def add_paper_texture(img):

    texture = np.random.normal(
        0,
        6,
        img.shape
    )

    texture = cv2.GaussianBlur(
        texture.astype(np.float32),
        (0, 0),
        sigmaX=3
    )

    result = img.astype(np.float32) + texture

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def random_blur(img):

    if random.random() < 0.4:

        k = random.choice([3, 5])

        img = cv2.GaussianBlur(
            img,
            (k, k),
            0
        )

    return img


def random_dropout(img):

    if random.random() < 0.5:

        count = random.randint(1, 5)

        for _ in range(count):

            x = random.randint(
                0,
                img.shape[1] - 1
            )

            y = random.randint(
                0,
                img.shape[0] - 1
            )

            r = random.randint(2, 8)

            cv2.circle(
                img,
                (x, y),
                r,
                255,
                -1
            )

    return img


def augment_image(img):

    img = random_rotation(img)

    img = random_scale(img)

    img = random_translation(img)

    img = random_shear(img)

    img = random_morphology(img)

    img = random_dropout(img)

    img = random_blur(img)

    img = add_gaussian_noise(img)

    img = add_paper_texture(img)

    return img

# ==========================================================
# SAVE CLASS
# ==========================================================

def export_original_images():

    counts = {}

    ndjson_files = sorted(
        Path(NDJSON_DIR).glob("*.ndjson")
    )

    if not ndjson_files:
        raise FileNotFoundError(f"Aucun fichier .ndjson trouvé dans : {Path(NDJSON_DIR).resolve()}")

    for file in tqdm(
        ndjson_files,
        desc="Export classes"
    ):

        class_name = file.stem

        class_dir = (
            Path(OUTPUT_DIR)
            / class_name
        )

        class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        count = 0

        with open(file, "r", encoding="utf-8") as f:

            for line in f:

                item = json.loads(line)

                if not item.get("recognized", True):
                    continue

                drawing = item["drawing"]

                img = drawing_to_image(drawing)

                filename = (
                    class_dir /
                    f"{count:07d}.png"
                )

                cv2.imwrite(
                    str(filename),
                    img
                )

                count += 1

        counts[class_name] = count

    return counts

# ==========================================================
# BALANCE DATASET
# ==========================================================

def balance_dataset(counts):

    if not counts:
        raise RuntimeError("Aucune classe trouvée.")

    target_count = min(
        max(counts.values()),
        50000
    )

    print(
        f"\nTarget images per class = {target_count}"
    )

    for class_name, count in tqdm(
        counts.items(),
        desc="Balancing"
    ):

        if count >= target_count:
            continue

        class_dir = (
            Path(OUTPUT_DIR)
            / class_name
        )

        images = sorted(class_dir.glob("*.png"))
        
        source = images[
            random.randint(0, len(images)-1)
        ]

        current = count

        while current < target_count:

            source = random.choice(images)

            img = cv2.imread(
                str(source),
                cv2.IMREAD_GRAYSCALE
            )

            aug = augment_image(img)

            filename = (
                class_dir
                / f"{current:07d}.png"
            )

            cv2.imwrite(
                str(filename),
                aug
            )

            current += 1

# ==========================================================
# MAIN
# ==========================================================

def main():

    Path(OUTPUT_DIR).mkdir(
        exist_ok=True
    )

    counts = export_original_images()

    print("\nInitial counts:")
    for k, v in counts.items():
        print(k, v)

    balance_dataset(counts)

    print("\nDone.")
    print(f"Dataset saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()