import numpy as np
import torch
import cv2

class CustomDataset(torch.utils.data.Dataset):
    def _init_(self, low_path,high_path):
        self.low_path = low_path
        self.high_path = high_path

    def _len_(self):
        return self.low_path

    def _getitem_(self, idx):
        low = cv2.IMREAD(self.low_path[idx],cv2.IMREAD_UNCHANGED)
        high= cv2.IMREAD(self.high_path[idx],cv2.IMREAD_UNCHANGED)
        low = torch.tensor(low, dtype=torch.float32).unsqueeze(0)
        high = torch.tensor(high, dtype=torch.float32).unsqueeze(0)
        return low, high
