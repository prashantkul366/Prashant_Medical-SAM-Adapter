import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import random_click
import cv2

class BUSI(Dataset):
    def __init__(self, args, data_path, mode="train", prompt="click"):
        self.image_dir = os.path.join(data_path, mode, "images")
        self.mask_dir  = os.path.join(data_path, mode, "masks")

        self.names = sorted(os.listdir(self.image_dir))
        self.img_size = args.image_size
        self.prompt = prompt
        self.out_size = args.out_size
        self.prompt = prompt
        
        # ===== PRINT DATASET INFO =====
        print(f"[BUSI / Kvasir Dataset]")
        print(f"Mode        : {mode}")
        print(f"Data path   : {data_path}")
        print(f"Image dir   : {self.image_dir}")
        print(f"Mask dir    : {self.mask_dir}")
        print(f"Total samples: {len(self.names)}")

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):

        name = self.names[idx]          # e.g. ISIC_0012255.npy

        # ---------- IMAGE (.npy → 4 channel) ----------
        base_name = os.path.splitext(name)[0]
        npy_name = base_name + ".npy"

        img_np = np.load(os.path.join(self.image_dir, npy_name))   # H,W,4

        # resize each channel
        img_np_resized = np.zeros((self.img_size, self.img_size, 4), dtype=np.float32)
        for c in range(4):
            img_np_resized[:, :, c] = cv2.resize(
                img_np[:, :, c],
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_LINEAR
            )

        img = torch.from_numpy(img_np_resized).permute(2, 0, 1).float()


        # ---------- MASK (.png) ----------
        mask_name = base_name + ".png"
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")
        mask = mask.resize((self.out_size, self.out_size))

        mask = torch.from_numpy(np.array(mask) > 0).unsqueeze(0).float()


        # ---------- PRINT SIZE ----------
        print("Image shape :", img.shape)
        print("Mask shape  :", mask.shape)


        # ---------- PROMPT ----------
        point_label = 1
        pt = torch.tensor([[self.img_size // 2, self.img_size // 2]])

        if self.prompt == "click":
            point_label, pt = random_click(mask.squeeze().numpy(), point_label)

        return {
            "image": img,
            "label": mask,
            "p_label": point_label,
            "pt": pt,
            "image_meta_dict": {"filename_or_obj": name}
        }


    # def __getitem__(self, idx):
    #     name = self.names[idx]

    #     img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
    #     mask = Image.open(os.path.join(self.mask_dir, name)).convert("L")

    #     # img = img.resize((self.img_size, self.img_size))
    #     # mask = mask.resize((self.img_size, self.img_size))

    #     img  = img.resize((self.img_size, self.img_size))
    #     mask = mask.resize((self.out_size, self.out_size))

    #     img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.
    #     mask = torch.from_numpy(np.array(mask) > 0).unsqueeze(0).float()

    #     point_label = 1
    #     pt = torch.tensor([[self.img_size//2, self.img_size//2]])

    #     if self.prompt == "click":
    #         point_label, pt = random_click(mask.squeeze().numpy(), point_label)

    #     return {
    #         "image": img,
    #         "label": mask,
    #         "p_label": point_label,
    #         "pt": pt,
    #         "image_meta_dict": {"filename_or_obj": name}
    #     }
