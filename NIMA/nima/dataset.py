#----------------------------------------------------------------------------------------------------------
# Code for dataloader adapted from:
# Hossein Talebi and Peyman Milanfar. Neural image assessment (2018)
# at https://github.com/truskovskiyk/nima.pytorch
#----------------------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class AVADataset(Dataset):
    def __init__(self, path_to_csv: Path, images_path: Path, transform):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, np.ndarray]:
        row = self.df.iloc[item]

        image_id = row["image_id"]
        image_path = self.images_path +"/" + f"{image_id}.jpg"
        image = default_loader(image_path)
        x = self.transform(image)

        y = row[1:].values.astype("float32")
        p = y / y.sum()
        ave = (p*[1,2,3,4,5,6,7,8,9,10]).sum()

        return x, ave
