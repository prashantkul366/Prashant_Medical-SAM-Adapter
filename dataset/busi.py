import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import random_click

class BUSI(Dataset):
    def __init__(self, args, data_path, mode="train", prompt="click"):
        self.image_dir = os.path.join(data_path, mode, "images")
        self.mask_dir  = os.path.join(data_path, mode, "masks")

        self.names = sorted(os.listdir(self.image_dir))
        self.img_size = args.image_size
        self.prompt = prompt
        
        # ===== PRINT DATASET INFO =====
        print(f"[BUSI Dataset]")
        print(f"Mode        : {mode}")
        print(f"Data path   : {data_path}")
        print(f"Image dir   : {self.image_dir}")
        print(f"Mask dir    : {self.mask_dir}")
        print(f"Total samples: {len(self.names)}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")

        img = img.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size))

        img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.
        mask = torch.from_numpy(np.array(mask) > 0).unsqueeze(0).float()

        point_label = 1
        pt = torch.tensor([[self.img_size//2, self.img_size//2]])

        if self.prompt == "click":
            point_label, pt = random_click(mask.squeeze().numpy(), point_label)

        return {
            "image": img,
            "label": mask,
            "p_label": point_label,
            "pt": pt,
            "image_meta_dict": {"filename_or_obj": name}
        }
