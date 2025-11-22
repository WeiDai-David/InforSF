# dataset/officehome.py

import os
from PIL import Image
from torch.utils.data import Dataset


class OfficeHomeDataset(Dataset):
    """
    Clean Office-Home loader:
    - only loads class folders
    - outputs PIL image (no transforms)
    - label = integer id
    """

    IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, root):
        self.root = root

        # only folders are classes
        classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        self.class2id = {c: i for i, c in enumerate(classes)}
        self.id2class = {i: c for c, i in self.class2id.items()}

        self.samples = []
        for cls in classes:
            cdir = os.path.join(root, cls)
            for fname in os.listdir(cdir):
                if fname.lower().endswith(self.IMG_EXT):
                    fpath = os.path.join(cdir, fname)
                    self.samples.append((fpath, self.class2id[cls]))

        print(f"[OfficeHomeDataset] Loaded {len(self.samples)} samples from {len(classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")  # do NOT apply transforms here
        return img, label
