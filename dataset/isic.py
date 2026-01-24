import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click

class ISIC2016(Dataset):
    def __init__(
        self,
        args,
        data_path,
        transform=None,
        transform_msk=None,
        mode="train",
        prompt="click"
    ):
        self.img_dir = os.path.join(data_path, mode, "images")
        self.mask_dir = os.path.join(data_path, mode, "masks")

        self.img_list = sorted(os.listdir(self.img_dir))
        self.mask_list = sorted(os.listdir(self.mask_dir))

        assert len(self.img_list) == len(self.mask_list)

        self.transform = transform
        self.transform_msk = transform_msk
        self.prompt = prompt
        self.img_size = args.image_size

        # ===== PRINT DATASET INFO =====
        print(f"[isic Dataset]")
        print(f"Mode        : {mode}")
        print(f"Data path   : {data_path}")
        print(f"Image dir   : {self.image_dir}")
        print(f"Mask dir    : {self.mask_dir}")
        print(f"Total samples: {len(self.names)}")

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_name = self.img_list[index]
        mask_name = self.mask_list[index]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask = mask.resize((self.img_size, self.img_size))

        # -------- PROMPT (SAM) --------
        point_label = 1
        pt = None

        if self.prompt == "click":
            point_label, pt = random_click(
                np.array(mask) / 255,
                point_label
            )

        # -------- TRANSFORMS --------
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask).int()

        # -------- FIX SHAPES (CRITICAL) --------
        if isinstance(pt, np.ndarray):
            pt = torch.from_numpy(pt).float()
        pt = pt.unsqueeze(0)          # [1,2]
        point_label = torch.tensor([point_label]).long()  # [1]

        return {
            "image": img,
            "label": mask,
            "pt": pt,
            "p_label": point_label,
            "image_meta_dict": {
                "filename_or_obj": img_name
            }
        }


# class ISIC2016(Dataset):
#     def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

#         df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
#         self.name_list = df.iloc[:,1].tolist()
#         self.label_list = df.iloc[:,2].tolist()
#         self.data_path = data_path
#         self.mode = mode
#         self.prompt = prompt
#         self.img_size = args.image_size

#         self.transform = transform
#         self.transform_msk = transform_msk

#     def __len__(self):
#         return len(self.name_list)

#     def __getitem__(self, index):
#         # if self.mode == 'Training':
#         #     point_label = random.randint(0, 1)
#         #     inout = random.randint(0, 1)
#         # else:
#         #     inout = 1
#         #     point_label = 1
#         point_label = 1

#         """Get the images"""
#         name = self.name_list[index]
#         img_path = os.path.join(self.data_path, name)
        
#         mask_name = self.label_list[index]
#         msk_path = os.path.join(self.data_path, mask_name)

#         img = Image.open(img_path).convert('RGB')
#         mask = Image.open(msk_path).convert('L')

#         # if self.mode == 'Training':
#         #     label = 0 if self.label_list[index] == 'benign' else 1
#         # else:
#         #     label = int(self.label_list[index])

#         newsize = (self.img_size, self.img_size)
#         mask = mask.resize(newsize)

#         if self.prompt == 'click':
#             point_label, pt = random_click(np.array(mask) / 255, point_label)

#         if self.transform:
#             state = torch.get_rng_state()
#             img = self.transform(img)
#             torch.set_rng_state(state)


#             if self.transform_msk:
#                 mask = self.transform_msk(mask).int()
                
#             # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
#             #     mask = 1 - mask
#         name = name.split('/')[-1].split(".jpg")[0]
#         image_meta_dict = {'filename_or_obj':name}
#         return {
#             'image':img,
#             'label': mask,
#             'p_label':point_label,
#             'pt':pt,
#             'image_meta_dict':image_meta_dict,
#         }