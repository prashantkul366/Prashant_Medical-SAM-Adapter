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
def dice_score(pred, gt, eps=1e-7):
    pred = pred.astype(np.bool_)
    gt   = gt.astype(np.bool_)
    inter = (pred & gt).sum()
    return (2. * inter) / (pred.sum() + gt.sum() + eps)

def iou_score(pred, gt, eps=1e-7):
    pred = pred.astype(np.bool_)
    gt   = gt.astype(np.bool_)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / (union + eps)

def specificity_score(pred, gt, eps=1e-7):
    tn = ((pred == 0) & (gt == 0)).sum()
    fp = ((pred == 1) & (gt == 0)).sum()
    return tn / (tn + fp + eps)

def accuracy_score_bin(pred, gt):
    return (pred == gt).mean()


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

    ckpt = torch.load(args.weights, map_location=device)
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
            pt = pack["pt"]
            pl = pack["p_label"]

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

            dice_list.append(dice_score(pd, gt))
            iou_list.append(iou_score(pd, gt))
            spec_list.append(specificity_score(pd, gt))
            acc_list.append(accuracy_score_bin(pd, gt))

    # -------- Results --------
    print("\n===== TEST RESULTS =====")
    print(f"Dice        : {np.mean(dice_list):.4f}")
    print(f"IoU         : {np.mean(iou_list):.4f}")
    print(f"Specificity : {np.mean(spec_list):.4f}")
    print(f"Accuracy    : {np.mean(acc_list):.4f}")


if __name__ == "__main__":
    main()
