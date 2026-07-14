from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
import torchvision.transforms.functional as TF

_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights, 512),
    "resnet50": (resnet50, ResNet50_Weights, 2048),
}

# ImageNet normalization constants
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FrozenBackbone:

    def __init__(self, name="resnet18", device=None):
        if name not in _BACKBONES:
            raise ValueError(f"backbone must be one of {sorted(_BACKBONES)}, got {name!r}")
        ctor, weights, self.out_dim = _BACKBONES[name]
        self.name = name
        self.device = device or pick_device()
        model = ctor(weights=weights.DEFAULT)
        model.fc = nn.Identity()  # expose the penultimate feature vector
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _preprocess(self, batch_uint8):
        x = torch.from_numpy(np.ascontiguousarray(batch_uint8)).float().div(255.0)
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        x = TF.resize(x, [224, 224], antialias=True)
        return (x - _MEAN) / _STD

    @torch.no_grad()
    def extract_features(self, images_uint8, batch_size=128, progress=False):
        feats = []
        n = len(images_uint8)
        for i in range(0, n, batch_size):
            xb = self._preprocess(images_uint8[i:i + batch_size]).to(self.device)
            feats.append(self.model(xb).cpu().numpy())
            if progress and (i // batch_size) % 20 == 0:
                print(f"    backbone {i}/{n}", flush=True)
        return np.concatenate(feats, axis=0).astype(np.float32)
