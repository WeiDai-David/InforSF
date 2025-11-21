# dataset/office_home.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class OfficeHomeDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.samples = []
        classes = sorted(os.listdir(root))
        self.class2id = {c:i for i,c in enumerate(classes)}

        for cls in classes:
            cdir = os.path.join(root, cls)
            for img in os.listdir(cdir):
                if img.lower().endswith((".jpg",".png")):
                    self.samples.append((os.path.join(cdir,img), self.class2id[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.tf(img)
        return img, label
