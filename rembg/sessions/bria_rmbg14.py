import os
from typing import List

import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession


class BriaRmBg14Session(BaseSession):
    """Inference session for the BRIA RMBG v1.4 model."""

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """Run inference and return the predicted mask."""
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

        return [mask]

    @classmethod
    def download_models(cls, *args, **kwargs):
        """Return the local path to the RMBG v1.4 ONNX model."""
        model_path = kwargs.get("model_path")
        if model_path:
            resolved = os.path.abspath(os.path.expanduser(model_path))
            if not os.path.exists(resolved):
                raise FileNotFoundError(
                    f"BRIA RMBG 1.4 model not found at the provided path: {resolved}"
                )
            return resolved

        base_dir = cls.u2net_home(*args, **kwargs)
        os.makedirs(base_dir, exist_ok=True)

        preferred_names = [
            f"{cls.name(*args, **kwargs)}.onnx",
            "rmbg14.onnx",
            "rmbg-1.4.onnx",
            "RMBG-1.4.onnx",
        ]

        for fname in preferred_names:
            candidate = os.path.join(base_dir, fname)
            if os.path.exists(candidate):
                return candidate

        fname = f"{cls.name(*args, **kwargs)}.onnx"

        try:
            pooch.retrieve(
                "https://huggingface.co/briaai/RMBG-1.4/resolve/main/rmbg-1.4.onnx?download=1",
                None,
                fname=fname,
                path=base_dir,
                progressbar=True,
            )
        except Exception as exc:  # pragma: no cover - network failures handled gracefully
            raise FileNotFoundError(
                "Unable to download the BRIA RMBG 1.4 model automatically. "
                "Download it manually and provide the path via extras.model_path."
            ) from exc

        return os.path.join(base_dir, fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """Return the identifier for this session."""
        return "bria-rmbg14"
