import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from sklearn.model_selection import train_test_split
from torchvision import transforms

# ------------------------
# CONFIG
# ------------------------
SEED       = 42
IMG_SIZE   = (512, 512)
BATCH_SIZE = 2
NUM_EPOCHS = 30
LR         = 1e-4

IMG_DIR    = '/kaggle/input/semantic-drone-dataset/dataset/semantic_drone_dataset/original_images'
MASK_DIR   = '/kaggle/input/semantic-drone-dataset/dataset/semantic_drone_dataset/label_images_semantic'
CLASS_CSV  = '/kaggle/input/semantic-drone-dataset/class_dict_seg.csv'
OUT_BASE   = 'val_outputs'

# ------------------------
# SEED EVERYTHING
# ------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------
# LOAD CLASS PALETTE
# ------------------------
df = pd.read_csv(CLASS_CSV)
df.columns = df.columns.str.strip()
NUM_CLASSES = len(df)
LUT = np.zeros((256**3,), dtype=np.uint8)
for idx, (r, g, b) in enumerate(zip(df['r'], df['g'], df['b'])):
    LUT[(int(r)<<16)|(int(g)<<8)|int(b)] = idx
IDX2COLOR = {
    idx: (int(r), int(g), int(b))
    for idx, (r, g, b) in enumerate(zip(df['r'], df['g'], df['b']))
}

# preprocessing for inference
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])

# ------------------------
# DATASET
# ------------------------
class DroneDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, files, tfm=None):
        self.imgs_dir, self.masks_dir, self.files, self.tfm = imgs_dir, masks_dir, files, tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fn = self.files[i]
        img = cv2.imread(os.path.join(self.imgs_dir, fn))[..., ::-1]
        mask_path = os.path.join(self.masks_dir, fn.replace('.jpg', '.png'))
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # convert mask to index map
        if m.ndim == 2:
            mask = m.astype(np.int64)
        else:
            rgb = m[..., ::-1]
            keys = (
                rgb[...,0].astype(np.int32)<<16 |
                rgb[...,1].astype(np.int32)<<8  |
                rgb[...,2].astype(np.int32)
            )
            mask = LUT[keys]

        # apply augmentations
        if self.tfm:
            aug = self.tfm(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']

        # ensure mask tensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        elif isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            raise TypeError(f"Unexpected mask type: {type(mask)}")

        return img, mask, fn

# ------------------------
# AUGMENTATIONS & LOADERS
# ------------------------
train_tfm = A.Compose([
    A.Resize(*IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ToTensorV2()
])
val_tfm = A.Compose([
    A.Resize(*IMG_SIZE),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ToTensorV2()
])

def get_loaders():
    all_files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg'))
    train_files, val_files = train_test_split(all_files, train_size=0.8, random_state=SEED)
    train_loader = DataLoader(
        DroneDataset(IMG_DIR, MASK_DIR, train_files, train_tfm),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        DroneDataset(IMG_DIR, MASK_DIR, val_files, val_tfm),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader, train_files, val_files

# ------------------------
# COLORIZE UTILITY
# ------------------------
def colorize(idx_map: np.ndarray) -> np.ndarray:
    h, w = idx_map.shape
    cmap = np.zeros((h, w, 3), np.uint8)
    for idx, col in IDX2COLOR.items():
        cmap[idx_map == idx] = col
    return cmap

# ------------------------
# SAVE VALIDATION VISUALS
# ------------------------
def save_val_visuals(model, file_list, epoch, out_base):
    out_dir = os.path.join(out_base, f'ep{epoch:02d}')
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for fn in tqdm(file_list[:5], desc=f"Epoch {epoch} viz"):
        img_bgr = cv2.imread(os.path.join(IMG_DIR, fn))
        img_rgb = img_bgr[..., ::-1]
        inp = preprocess(img_rgb).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            pr_idx = model(inp)['out'][0].argmax(0).cpu().numpy()
        pr_col = colorize(pr_idx)
        pr_resized = cv2.resize(
            pr_col,
            (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        comp = np.concatenate([img_rgb, pr_resized], axis=1)
        cv2.imwrite(
            os.path.join(out_dir, fn.replace('.jpg','_comparison.png')),
            cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
        )
    model.train()

# ------------------------
# LOSSES: Dice + CE
# ------------------------
def dice_loss(pred, target, eps=1e-6):
    pred_soft = torch.softmax(pred, dim=1)
    one_hot = torch.nn.functional.one_hot(target, NUM_CLASSES).permute(0,3,1,2).float()
    inter = torch.sum(pred_soft * one_hot, (0,2,3))
    card  = torch.sum(pred_soft + one_hot, (0,2,3))
    return 1 - ((2*inter + eps) / (card + eps)).mean()

class CombinedLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=weight)
    def forward(self, logits, target):
        return self.ce(logits, target) + dice_loss(logits, target)

# ------------------------
# METRICS
# ------------------------
def compute_metrics(preds, gts, num_classes):
    tp = np.zeros(num_classes, np.int64)
    fp = np.zeros(num_classes, np.int64)
    fn = np.zeros(num_classes, np.int64)
    for c in range(num_classes):
        tp[c] = int(((preds==c)&(gts==c)).sum())
        fp[c] = int(((preds==c)&(gts!=c)).sum())
        fn[c] = int(((preds!=c)&(gts==c)).sum())
    return tp, fp, fn

# ------------------------
# MAIN
# ------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier = DeepLabHead(2048, NUM_CLASSES)
    model.to(device)

    train_loader, val_loader, train_files, val_files = get_loaders()

    # compute class-frequency weights
    counts = np.zeros(NUM_CLASSES, np.int64)
    for imgs, masks, _ in tqdm(train_loader, desc="Counting pixels"):
        arr = masks.numpy()
        for c in range(NUM_CLASSES):
            counts[c] += int((arr==c).sum())
    inv_freq = 1.0/(counts + 1e-6)
    # cast to float32 explicitly
    weight = torch.tensor(
        inv_freq/inv_freq.sum()*NUM_CLASSES,
        dtype=torch.float32,
        device=device
    ).float()

    criterion = CombinedLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_f1 = 0.0
    for epoch in range(1, NUM_EPOCHS+1):
        # — TRAIN —
        model.train()
        train_loss = 0.0
        # initialize separate arrays (fix bug that caused same F1)
        tp_t = np.zeros(NUM_CLASSES, np.int64)
        fp_t = np.zeros(NUM_CLASSES, np.int64)
        fn_t = np.zeros(NUM_CLASSES, np.int64)
        total = 0
        correct = 0

        for imgs, masks, _ in tqdm(train_loader, desc=f"Train Ep{epoch}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)['out']
            loss = criterion(out, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            preds = out.argmax(1).cpu().numpy().ravel()
            gts   = masks.cpu().numpy().ravel()
            correct += int((preds==gts).sum())
            total   += preds.size
            tp, fp, fn = compute_metrics(preds, gts, NUM_CLASSES)
            tp_t += tp; fp_t += fp; fn_t += fn

        acc_train = correct / total if total > 0 else 0.0
        f1_train = np.mean(2*tp_t/(2*tp_t + fp_t + fn_t + 1e-6))

        # — VALIDATE —
        model.eval()
        val_loss = 0.0
        tp_v = np.zeros(NUM_CLASSES, np.int64)
        fp_v = np.zeros(NUM_CLASSES, np.int64)
        fn_v = np.zeros(NUM_CLASSES, np.int64)
        total_v = 0
        correct_v = 0

        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader, desc=f"Val Ep{epoch}", leave=False):
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)['out']
                val_loss += criterion(out, masks).item()

                preds = out.argmax(1).cpu().numpy().ravel()
                gts   = masks.cpu().numpy().ravel()
                correct_v += int((preds==gts).sum())
                total_v   += preds.size
                tp, fp, fn = compute_metrics(preds, gts, NUM_CLASSES)
                tp_v += tp; fp_v += fp; fn_v += fn

        # average losses
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else train_loss
        avg_val_loss   = val_loss / len(val_loader) if len(val_loader) > 0 else val_loss

        acc_val = correct_v / total_v if total_v > 0 else 0.0
        f1_val  = np.mean(2*tp_v/(2*tp_v + fp_v + fn_v + 1e-6))
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch:02d} → "
            f"Train L:{avg_train_loss:.4f}, "
            f"A:{acc_train:.4f}, F1:{f1_train:.4f} | "
            f"Val   L:{avg_val_loss:.4f}, "
            f"A:{acc_val:.4f}, F1:{f1_val:.4f}"
        )

        # save best checkpoint
        if f1_val > best_val_f1:
            best_val_f1 = f1_val
            torch.save(model.state_dict(), 'best_drone_segmentation.pth')
            print(f"→ New best F1: {best_val_f1:.4f} (model saved)")

        # save 5 validation visuals
        save_val_visuals(model, val_files, epoch, OUT_BASE)

    print(f"Training complete. Best Val F1: {best_val_f1:.4f}")

if __name__=='__main__':
    main()