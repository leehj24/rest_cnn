import os
import json
import cv2
import numpy as np
from PIL import Image
import torch

# ── 호환성 패치: torch.version.hip 속성 없을 때 대비 ─────────────────────────────
if not hasattr(torch, 'version'):
    class _v:
        hip = False
    torch.version = _v()
# ────────────────────────────────────────────────────────────────────────────────

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, jaccard_score, r2_score
)

# ─── 1. 경로 & 하이퍼파라미터 ─────────────────────────────────────────────────
IMG_ROOT    = r'E:\Hyunji4579 Dropbox\Hyunji4579의 팀 폴더\도로장애물\Images\CRACK'
ANN_ROOT    = r'E:\Hyunji4579 Dropbox\Hyunji4579의 팀 폴더\도로장애물\Annotations\CRACK'
INPUT_SIZE  = 32
BATCH_SIZE  = 32
NUM_EPOCHS  = 1
NUM_CLASSES = 2
DEVICE      = torch.device('cpu')  # GPU 사용 시 'cuda'

# ─── 2. collate_fn: None 샘플 걸러내기 ─────────────────────────────────────────
def filter_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

# ─── 3. Focal Loss 정의 ───────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ─── 4. Dataset 정의 ─────────────────────────────────────────────────────────
class CrackSegDataset(Dataset):
    def __init__(self, img_root, ann_root, img_tf, mask_tf):
        self.img_tf, self.mask_tf = img_tf, mask_tf
        self.items = []
        exts = ('.png', '.jpg', '.jpeg', '.webp')
        for cls in os.listdir(img_root):
            img_dir = os.path.join(img_root, cls)
            ann_dir = os.path.join(ann_root, cls)
            if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
                continue
            for fn in os.listdir(img_dir):
                if not fn.lower().endswith(exts):
                    continue
                jp = os.path.join(ann_dir, os.path.splitext(fn)[0] + '.json')
                if os.path.isfile(jp):
                    self.items.append((os.path.join(img_dir, fn), jp))
        assert self.items, f"데이터가 없습니다: {img_root}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, json_path = self.items[idx]
        try:
            # 이미지 읽기 (cv2 우선, 실패 시 PIL)
            arr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                pil = Image.open(img_path).convert('RGB')
                arr = np.array(pil)
            else:
                if arr.ndim == 3 and arr.shape[2] == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
                else:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(arr)

            # JSON → mask 생성
            w, h = img.size
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            mask = np.zeros((h, w), dtype=np.uint8)
            for ann in data.get('annotations', []):
                for poly in ann.get('polyline', []):
                    if not poly or any(p is None for p in poly) or len(poly) % 2 != 0:
                        continue
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=1)
            mask = Image.fromarray(mask)

            # Transform 적용
            img_t  = self.img_tf(img)
            mask_t = self.mask_tf(mask).squeeze(0).long()
            return img_t, mask_t

        except Exception as e:
            print(f"[Warning] 스킵: {img_path} → {e}")
            return None

# ─── 5. Transform 정의 ───────────────────────────────────────────────────────
img_tf = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
])
mask_tf = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.NEAREST),
    transforms.PILToTensor(),
])

# ─── 6. 모델 정의 ───────────────────────────────────────────────────────────
class TinySegModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )
    def forward(self, x):
        return self.net(x)

# ─── 7. 시각화 함수 ─────────────────────────────────────────────────────────
def visualize_predictions(model, loader, num_samples=5, output_dir='visualizations'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, masks = batch
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs).argmax(1)
            for i in range(imgs.size(0)):
                if count >= num_samples:
                    return
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(imgs[i].cpu().permute(1,2,0))
                axes[0].set_title('Input');    axes[0].axis('off')
                axes[1].imshow(masks[i].cpu(), cmap='gray')
                axes[1].set_title('GT');       axes[1].axis('off')
                axes[2].imshow(preds[i].cpu(), cmap='gray')
                axes[2].set_title('Pred');     axes[2].axis('off')
                plt.savefig(os.path.join(output_dir, f'pred_{count}.png'))
                plt.close(fig)
                count += 1

# ─── 8. 학습·평가·저장·시각화 루프 ──────────────────────────────────────────
def main():
    ds = CrackSegDataset(IMG_ROOT, ANN_ROOT, img_tf, mask_tf)
    n_train = int(len(ds) * 0.8)
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=filter_none_collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=filter_none_collate)

    model     = TinySegModel(NUM_CLASSES).to(DEVICE)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    best_iou = 0.0
    best_path = r'C:\Users\hyunj\Downloads\deep_test\best_model.pt'

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS} [Train]'):
            if batch is None:
                continue
            imgs, masks = batch
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_masks = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                imgs, masks = batch
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs).argmax(1)
                all_preds.append(preds.cpu().numpy().ravel())
                all_masks.append(masks.cpu().numpy().ravel())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_masks)

        acc   = accuracy_score(y_true, y_pred)
        prec  = precision_score(y_true, y_pred, zero_division=0)
        rec   = recall_score(y_true, y_pred, zero_division=0)
        f1    = f1_score(y_true, y_pred, zero_division=0)
        iou   = jaccard_score(y_true, y_pred, zero_division=0)
        r2    = r2_score(y_true, y_pred)

        print(f'Epoch {epoch} Metrics:')
        print(f'  Acc : {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}, R2: {r2:.4f}')

        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), best_path)
            print(f'  -> New best model saved (IoU={iou:.4f})')

    print('Training complete.')
    print(f'Best IoU: {best_iou:.4f}, saved to {best_path}')

    # 시각화
    best_model = TinySegModel(NUM_CLASSES).to(DEVICE)
    best_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    visualize_predictions(best_model, val_loader, num_samples=5)

if __name__ == '__main__':
    main()
