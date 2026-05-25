import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from zipfile import ZipFile


def unzip(zip_path: str, image_path: str):
    image_path = Path(image_path)
    image_path.mkdir(parents=True, exist_ok=True)
    if len(os.listdir(image_path)) > 0:
        print(f"Images already in {image_path}")
        return
    with ZipFile(zip_path, 'r') as zf:
        print('Unzipping data...')
        zf.extractall(image_path)


def augment_images(in_path: str, out_path: str, subclass: str):
    subclass_dir = Path(out_path) / subclass
    subclass_dir.mkdir(parents=True, exist_ok=True)

    if len(os.listdir(subclass_dir)) > 0:
        print(f"Files already in {subclass_dir}")
        return

    for image_path in glob(os.path.join(in_path, "*/*.png")):
        name = Path(image_path).stem
        image = cv2.imread(image_path)
        for scale in [1.0, 0.9, 0.8, 0.7, 0.6]:
            for rotation in [0, 90, 180, 270]:
                scale_tag = '1' if scale == 1.0 else str(scale).split('.')[1]
                out_file = subclass_dir / f"{name}_{scale_tag}_{rotation}.png"
                resized = cv2.resize(image,
                                     (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                                     interpolation=cv2.INTER_CUBIC)
                center = tuple(np.array(resized.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(center, rotation, 1.0)
                rotated = cv2.warpAffine(resized, rot_mat, resized.shape[1::-1], flags=cv2.INTER_CUBIC)
                cv2.imwrite(str(out_file), rotated)

    print(f"Finished augmenting into {subclass_dir}")


def calc_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10(1.0 / mse)
