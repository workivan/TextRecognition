import random

import cv2
import numpy as np

"""
The Preprocessor class is designed to (pre)process images.
"""


class Preprocessor:
    def __init__(self, image: np.ndarray):
        if image is None:
            raise ValueError("image arg is null!!")

        self._image = image

    def process(self) -> np.ndarray:
        img = self._image.astype(np.float)

        img = self.photometric_augmentation(img)
        img = self.geometric_img_augmentation(img)

        img = cv2.transpose(img)
        # convert to range [-1, 1]
        img = img / 255 - 0.5
        return img

    @staticmethod
    def geometric_img_augmentation(img: np.ndarray) -> np.ndarray:
        wt, ht, _ = img.shape
        h, w = img.shape
        f = min(wt / w, ht / h)
        fx = f * np.random.uniform(0.75, 1.05)
        fy = f * np.random.uniform(0.75, 1.05)

        # random position around center
        txc = (wt - w * fx) / 2
        tyc = (ht - h * fy) / 2
        freedom_x = max((wt - fx * w) / 2, 0)
        freedom_y = max((ht - fy * h) / 2, 0)
        tx = txc + np.random.uniform(-freedom_x, freedom_x)
        ty = tyc + np.random.uniform(-freedom_y, freedom_y)

        # map image into target image
        m = np.float32([[fx, 0, tx], [0, fy, ty]])
        target = np.ones((ht, wt)) * 255
        img = cv2.warpAffine(img, m, dsize=(wt, ht), dst=target, borderMode=cv2.BORDER_TRANSPARENT)

        return img

    @staticmethod
    def photometric_augmentation(img: np.ndarray) -> np.ndarray:
        def rand_odd():
            return random.randint(1, 3) * 2 + 1

        img = cv2.GaussianBlur(img, (rand_odd(), rand_odd()), 0)

        return img
