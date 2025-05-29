import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# ─── 경로 및 설정 ────────────────────────────────────────────────
MODEL_PATH = 'best_model.pt'
IMG_DIR = r'E:\도로장애물·표면 인지 영상(수도권 외)\Validation\Images\2.CRACK\C_Frontback_M01'
SAVE_DIR = './ultra_outputs'
INPUT_SIZE = 32
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Transform 정의 ─────────────────────────────────────────────
img_tf = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
])

# ─── 모델 정의 ──────────────────────────────────────────────────
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

# ─── 모델 로드 ──────────────────────────────────────────────────
model = TinySegModel(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ─── 예측 빠르게 수행 ───────────────────────────────────────────
exts = ('.png', '.jpg', '.jpeg', '.webp')
files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(exts)]

with torch.no_grad():
    for i, fname in enumerate(tqdm(files, desc="Predicting")):
        try:
            img_path = os.path.join(IMG_DIR, fname)
            arr = cv2.imread(img_path)
            if arr is None:
                continue
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(arr)
            img_tensor = img_tf(img).unsqueeze(0).to(DEVICE)

            pred = model(img_tensor).argmax(1).squeeze(0).cpu().numpy()
            cv2.imwrite(os.path.join(SAVE_DIR, f'{i:03}_pred.png'), pred * 255)

        except Exception as e:
            print(f"[ERROR] {fname} → {e}")
