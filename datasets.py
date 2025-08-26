import os
import glob
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
from PIL import Image


def is_dicom(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        return True
    # Some DICOMs do not have .dcm extension
    try:
        with open(path, 'rb') as f:
            preamble = f.read(132)
        return preamble[128:132] == b'DICM'
    except Exception:
        return False


def load_image_16bit(path: str) -> np.ndarray:
    """Load image as 16-bit numpy array (H, W). Supports DICOM and common image formats.
    For non-16-bit sources, converts to uint16 by scaling.
    """
    if is_dicom(path):
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        # Normalize to [0, 65535] and cast to uint16
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 65535.0).clip(0, 65535).astype(np.uint16)
        return arr
    else:
        img = Image.open(path).convert('L')  # grayscale
        arr8 = np.array(img, dtype=np.uint8)
        # Promote to 16-bit by scaling
        arr16 = (arr8.astype(np.uint16) * 257).astype(np.uint16)
        return arr16


def normalize_to_minus1_1(arr16: np.ndarray) -> torch.Tensor:
    arr = arr16.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val == min_val:
        arr = np.zeros_like(arr, dtype=np.float32)
    else:
        arr = 2.0 * (arr - min_val) / (max_val - min_val) - 1.0
    t = torch.from_numpy(arr).float().unsqueeze(0)  # (1, H, W)
    return t


class MedicalImageFolder(Dataset):
    """
    Recursively load images from a directory. Supports DICOM, PNG, JPG, TIFF.
    Provides random crops of size patch_size for training.
    """
    def __init__(self, root_dir: str, patch_size: int = 128):
        self.root_dir = root_dir
        self.patch_size = patch_size
        # Gather files
        patterns = ["**/*.dcm", "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.tif", "**/*.tiff", "**/*.bmp"]
        files: List[str] = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(root_dir, pat), recursive=True))
        # Fallback: include files without extension for possible DICOM
        files.extend([p for p in glob.glob(os.path.join(root_dir, "**/*"), recursive=True) if os.path.isfile(p)])
        # Deduplicate preserving order
        seen = set()
        self.files = []
        for f in files:
            if f not in seen:
                seen.add(f)
                self.files.append(f)
        if len(self.files) == 0:
            raise ValueError(f"No images found under {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        try:
            arr16 = load_image_16bit(path)
        except Exception:
            # Try as DICOM without extension
            ds = pydicom.dcmread(path)
            arr16 = ds.pixel_array.astype(np.uint16)
        img = normalize_to_minus1_1(arr16)  # (1, H, W)
        _, H, W = img.shape
        ps = self.patch_size
        if H < ps or W < ps:
            # Center pad to patch size
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            img = torch.nn.functional.pad(img, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='reflect')
            _, H, W = img.shape
        # Random crop
        top = torch.randint(0, H - ps + 1, (1,)).item()
        left = torch.randint(0, W - ps + 1, (1,)).item()
        patch = img[:, top:top + ps, left:left + ps]
        return patch  # (1, ps, ps)