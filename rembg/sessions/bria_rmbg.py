import os
from typing import List

import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class BriaRmBgSession(BaseSession):
    """
    This class represents a Bria-rmbg-2.0 session, which is a subclass of BaseSession.
    """

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """
        Predicts the output masks for the input image using the inner session.

        Parameters:
            img (PILImage): The input image.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[PILImage]: The list of output masks.
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (1024, 1024)
            ),
        )

        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)

        refined = self._refine_mask(np.asarray(mask, dtype=np.float32) / 255.0)
        refined_mask = Image.fromarray((refined * 255).astype("uint8"), mode="L")

        return [refined_mask]

    @staticmethod
    def _refine_mask(mask: np.ndarray) -> np.ndarray:
        """Apply lightweight morphological smoothing to reduce aliasing artifacts."""

        if mask.ndim != 2:
            return mask

        # Morphological closing to fill small gaps and reduce jagged artifacts
        dilated = BriaRmBgSession._dilate(mask)
        closed = BriaRmBgSession._erode(dilated)

        # Edge-preserving smoothing kernel (approximate gaussian)
        smoothed = BriaRmBgSession._smooth(closed)

        # Blend the smoothed result with the original mask to retain fine details
        refined = (0.6 * smoothed) + (0.4 * mask)

        return np.clip(refined, 0.0, 1.0)

    @staticmethod
    def _smooth(mask: np.ndarray) -> np.ndarray:
        kernel = np.array(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float32
        )
        kernel /= kernel.sum()

        padded = np.pad(mask, 1, mode="edge")

        smoothed = (
            kernel[0, 0] * padded[:-2, :-2]
            + kernel[0, 1] * padded[:-2, 1:-1]
            + kernel[0, 2] * padded[:-2, 2:]
            + kernel[1, 0] * padded[1:-1, :-2]
            + kernel[1, 1] * padded[1:-1, 1:-1]
            + kernel[1, 2] * padded[1:-1, 2:]
            + kernel[2, 0] * padded[2:, :-2]
            + kernel[2, 1] * padded[2:, 1:-1]
            + kernel[2, 2] * padded[2:, 2:]
        )

        return smoothed

    @staticmethod
    def _dilate(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask, 1, mode="edge")

        slices = [
            padded[:-2, :-2],
            padded[:-2, 1:-1],
            padded[:-2, 2:],
            padded[1:-1, :-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, :-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]

        result = slices[0]
        for slc in slices[1:]:
            result = np.maximum(result, slc)

        return result

    @staticmethod
    def _erode(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask, 1, mode="edge")

        slices = [
            padded[:-2, :-2],
            padded[:-2, 1:-1],
            padded[:-2, 2:],
            padded[1:-1, :-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, :-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]

        result = slices[0]
        for slc in slices[1:]:
            result = np.minimum(result, slc)

        return result

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Downloads the BRIA-RMBG 2.0 model file from a specific URL and saves it.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path to the downloaded model file.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/bria-rmbg-2.0.onnx",
            (
                None
                if cls.checksum_disabled(*args, **kwargs)
                else "sha256:5b486f08200f513f460da46dd701db5fbb47d79b4be4b708a19444bcd4e79958"
            ),
            fname=fname,
            path=cls.u2net_home(*args, **kwargs),
            progressbar=True,
        )

        return os.path.join(cls.u2net_home(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the Bria-rmbg session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "bria-rmbg"
