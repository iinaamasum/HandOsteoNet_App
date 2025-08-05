import torch
import torch.nn as nn
from .seg import BoneAgeSegModel
from .bone_age_model import HandOsteoNetModel

class BoneAgeFullModel(nn.Module):
    def __init__(self, input_size=480):
        super().__init__()
        self.seg_model = BoneAgeSegModel(input_size=input_size)
        self.bone_age_model = HandOsteoNetModel(img_size=input_size)

    def forward(self, img, gender):
        if img.shape[1:] != (3, 480, 480):
            raise ValueError(f"Expected image shape [B, 3, 480, 480], got {img.shape}")
        if gender.ndim == 1:
            gender = gender.unsqueeze(1)
        if gender.shape[1:] != (1,):
            raise ValueError(f"Expected gender shape [B, 1], got {gender.shape}")
        features = self.seg_model(img)
        img_features = torch.cat([img, features], dim=1)
        return self.bone_age_model(img_features, gender) 