import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import cfg
from dataset import BUSI
from utils import *
from function import validation_sam

# -----------------------------
# Metric helpers
# -----------------------------
# def dice_score(pred, gt, eps=1e-7):
#     pred = pred.astype(np.bool_)
#     gt   = gt.astype(np.bool_)
#     inter = (pred & gt).sum()
#     return (2. * inter) / (pred.sum() + gt.sum() + eps)

# def iou_score(pred, gt, eps=1e-7):
#     pred = pred.astype(np.bool_)
#     gt   = gt.astype(np.bool_)
#     inter = (pred & gt).sum()
#     union = (pred | gt).sum()
#     return inter / (union + eps)

# def specificity_score(pred, gt, eps=1e-7):
#     tn = ((pred == 0) & (gt == 0)).sum()
#     fp = ((pred == 1) & (gt == 0)).sum()
#     return tn / (tn + fp + eps)

# def accuracy_score_bin(pred, gt):
#     return (pred == gt).mean()

def dice_uc(pred, gt, eps=1e-5):
    pred = pred.astype(np.float32)
    gt   = gt.astype(np.float32)
    return 2.0 * np.sum(pred * gt) / (np.sum(pred) + np.sum(gt) + eps)

def iou_uc(pred, gt, eps=1e-7):
    pred = pred.astype(np.bool_)
    gt   = gt.astype(np.bool_)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / (union + eps)

def confusion_uc(pred, gt):
    TP = ((pred == 1) & (gt == 1)).sum()
    TN = ((pred == 0) & (gt == 0)).sum()
    FP = ((pred == 1) & (gt == 0)).sum()
    FN = ((pred == 0) & (gt == 1)).sum()
    return TP, TN, FP, FN

# -----------------------------
# Main
# -----------------------------
def main():
    args = cfg.parse_args()

    device = torch.device(f"cuda:{args.gpu_device}")

    # -------- Load model --------
    net = get_network(
        args, args.net,
        use_gpu=True,
        gpu_device=device,
        distribution=args.distributed
    )

    print("=> loading checkpoint from", args.weights)
    ckpt = torch.load(args.weights, map_location=device)
    # print("=> loaded checkpoint")
    net.load_state_dict(ckpt["state_dict"], strict=False)
    net.eval()

    print(f"Loaded checkpoint from: {args.weights}")

    # -------- Dataset --------
    test_set = BUSI(args, args.data_path, mode="test")
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False
    )

    dice_list = []
    iou_list = []
    spec_list = []
    acc_list = []


    # -------- Inference --------
    with torch.no_grad():
        for pack in tqdm(test_loader, desc="Testing"):
            imgs = pack["image"].to(device)
            masks = pack["label"].to(device)
            # pt = pack["pt"]
            # pl = pack["p_label"]

            pt = pack["pt"].to(device)
            pl = pack["p_label"].to(device)

            if pt.dim() == 2:      # [B,2] → [B,1,2]
                pt = pt.unsqueeze(1)

            if pl.dim() == 1:      # [B] → [B,1]
                pl = pl.unsqueeze(1)


            # SAM forward
            imgs_p = net.preprocess(imgs)
            emb = net.image_encoder(imgs_p)

            se, de = net.prompt_encoder(
                points=(pt.to(device), pl.to(device)),
                boxes=None,
                masks=None
            )

            pred, _ = net.mask_decoder(
                image_embeddings=emb,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False
            )

            pred = torch.nn.functional.interpolate(
                pred, size=(args.out_size, args.out_size),
                mode="bilinear", align_corners=False
            )

            prob = torch.sigmoid(pred)
            pred_bin = (prob > 0.5).float()

            gt = masks.cpu().numpy().squeeze()
            pd = pred_bin.cpu().numpy().squeeze()

            # dice_list.append(dice_score(pd, gt))
            # iou_list.append(iou_score(pd, gt))
            # spec_list.append(specificity_score(pd, gt))
            # acc_list.append(accuracy_score_bin(pd, gt))
                    
            TP, TN, FP, FN = confusion_uc(pd, gt)

            dice_list.append(dice_uc(pd, gt))
            iou_list.append(iou_uc(pd, gt))
            spec_list.append(TN / (TN + FP + 1e-7))
            acc_list.append((TP + TN) / (TP + TN + FP + FN + 1e-7))

    # -------- Results --------
    print("\n===== TEST RESULTS =====")
    print(f"Dice        : {np.mean(dice_list):.4f}")
    print(f"IoU         : {np.mean(iou_list):.4f}")
    print(f"Specificity : {np.mean(spec_list):.4f}")
    print(f"Accuracy    : {np.mean(acc_list):.4f}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3

# import os
# import torch
# import numpy as np
# from tqdm import tqdm

# import cfg
# from dataset import get_dataloader
# from utils import *
# import function

# # =========================
# # Metric helpers (FROM YOUR CODE)
# # =========================
# def compute_metrics(pred, gt):
#     """
#     pred, gt: numpy arrays (0/1), same shape
#     """
#     pred = pred.astype(np.float32)
#     gt   = gt.astype(np.float32)

#     TP = ((pred == 1) & (gt == 1)).sum()
#     TN = ((pred == 0) & (gt == 0)).sum()
#     FP = ((pred == 1) & (gt == 0)).sum()
#     FN = ((pred == 0) & (gt == 1)).sum()

#     eps = 1e-7

#     dice = (2 * TP) / (2 * TP + FP + FN + eps)
#     iou  = TP / (TP + FP + FN + eps)
#     sensitivity = TP / (TP + FN + eps)
#     specificity = TN / (TN + FP + eps)
#     accuracy = (TP + TN) / (TP + TN + FP + FN + eps)

#     return dice, iou, sensitivity, specificity, accuracy


# def main():
#     print("=========> Loading config <=========")
#     args = cfg.parse_args()
#     set_seed(args.seed)

#     device = torch.device(f"cuda:{args.gpu_device}" if args.gpu else "cpu")

#     # =========================
#     # Load model
#     # =========================
#     net = get_network(
#         args,
#         args.net,
#         use_gpu=args.gpu,
#         gpu_device=device,
#         distribution=args.distributed
#     ).to(device)

#     assert args.weights != 0, "❌ Please provide --weights BEST_CHECKPOINT.pth"
#     checkpoint = torch.load(args.weights, map_location=device)

#     print("=> Loading checkpoint from", args.weights)
#     net.load_state_dict(checkpoint["state_dict"], strict=False)
#     net.eval()

#     print(f"✅ Loaded checkpoint: {args.weights}")

#     # =========================
#     # Load TEST dataloader
#     # =========================
#     _, test_loader = get_dataloader(args)

#     # =========================
#     # Meters
#     # =========================
#     dice_list = []
#     iou_list = []
#     spec_list = []
#     acc_list = []
#     sens_list = []

#     # =========================
#     # TEST LOOP
#     # =========================
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Testing", ncols=80):
#             img  = batch["image"].to(device)
#             gt   = batch["label"].to(device)   # [B,1,H,W]

#             # ---- forward ----
#             output = net(
#                 img,
#                 batch["pt"].to(device),
#                 batch["p_label"].to(device)
#             )

#             # final prediction (same logic as SAM validation)
#             pred = torch.sigmoid(output)
#             pred = (pred > 0.5).float()

#             pred_np = pred.cpu().numpy()
#             gt_np   = gt.cpu().numpy()

#             # ---- compute metrics per image ----
#             for b in range(pred_np.shape[0]):
#                 d, i, se, sp, ac = compute_metrics(
#                     pred_np[b, 0],
#                     gt_np[b, 0]
#                 )
#                 dice_list.append(d)
#                 iou_list.append(i)
#                 sens_list.append(se)
#                 spec_list.append(sp)
#                 acc_list.append(ac)

#     # =========================
#     # FINAL RESULTS
#     # =========================
#     print("\n======== TEST RESULTS ========")
#     print(f"Dice        : {np.mean(dice_list):.4f}")
#     print(f"IoU         : {np.mean(iou_list):.4f}")
#     print(f"Sensitivity : {np.mean(sens_list) * 100:.2f}%")
#     print(f"Specificity : {np.mean(spec_list) * 100:.2f}%")
#     print(f"Accuracy    : {np.mean(acc_list) * 100:.2f}%")
#     print("===============================")


# if __name__ == "__main__":
#     main()
