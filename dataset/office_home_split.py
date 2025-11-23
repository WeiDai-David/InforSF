# dataset/office_home_split.py

import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset

DOMAINS = ["Art", "Clipart", "Product", "Real World"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
SPLIT_FILE = "data/officehome_split.json"


class OfficeHomeSplitDataset(Dataset):
    """
    OfficeHome Split Dataset
    - train_ratio controls training portion
    - test split uses ALL remaining samples
    - domain: None / "Art" / ["Art", "Product"]
    - split created once → saved into JSON
    - no overlap, fully reproducible
    """

    def __init__(self, root, split="train", domain=None, ratio=0.8, seed=42):
        assert split in ["train", "test"]
        self.root = root
        self.split = split
        self.ratio = ratio
        self.seed = seed

        # Ensure dataset root exists
        if not os.path.exists(root):
            raise FileNotFoundError(f"[Error] Dataset root not found: {root}")

        # If split file missing → generate once
        if not os.path.exists(SPLIT_FILE):
            print("[OfficeHomeSplit] No split file → generating...")
            self._generate_split()

        # Load the split
        print(f"[OfficeHomeSplit] Loading split from {SPLIT_FILE}")
        with open(SPLIT_FILE, "r") as f:
            data = json.load(f)

        # class id mappings
        self.class2id = {k: int(v) for k, v in data["class2id"].items()}
        self.id2class = {v: k for k, v in self.class2id.items()}

        # load train/test samples
        samples = data[split]

        # handle domain filter (None / str / list)
        if domain is not None:
            if isinstance(domain, str):
                domain = [domain]

            domain_prefixes = [os.path.join(root, d) for d in domain]

            samples = [
                (p, lab)
                for (p, lab) in samples
                if any(p.startswith(prefix) for prefix in domain_prefixes)
            ]

            print(f"[OfficeHomeSplit] Filtered to domain(s): {domain} → {len(samples)} samples")

        self.samples = samples

        print(f"[OfficeHomeSplit] Loaded {len(self.samples)} samples for split={split}")

    # ------------------------------------------------------------------
    # Generate split JSON (only once)
    # ------------------------------------------------------------------
    def _generate_split(self):
        random.seed(self.seed)

        print("[OfficeHomeSplit] Generating deterministic split...")

        # 1) Determine classes (from any domain)
        first_domain = os.path.join(self.root, DOMAINS[0])
        classes = sorted([
            d for d in os.listdir(first_domain)
            if os.path.isdir(os.path.join(first_domain, d))
        ])

        class2id = {cls: i for i, cls in enumerate(classes)}

        train_samples = []
        test_samples = []

        # 2) For each domain and each class
        for dom in DOMAINS:
            dom_dir = os.path.join(self.root, dom)

            for cls in classes:
                cls_dir = os.path.join(dom_dir, cls)
                if not os.path.exists(cls_dir):
                    continue

                # collect all images
                imgs = [
                    os.path.join(cls_dir, f)
                    for f in os.listdir(cls_dir)
                    if f.lower().endswith(IMG_EXT)
                ]

                random.shuffle(imgs)

                N = len(imgs)
                N_train = int(N * self.ratio)

                # train = first N_train images
                train_imgs = imgs[:N_train]
                # test = all remaining images
                test_imgs = imgs[N_train:]

                label = class2id[cls]

                for p in train_imgs:
                    train_samples.append((p, label))
                for p in test_imgs:
                    test_samples.append((p, label))

        # Save JSON
        split_data = {
            "class2id": class2id,
            "train": train_samples,
            "test": test_samples
        }

        os.makedirs("data", exist_ok=True)
        with open(SPLIT_FILE, "w") as f:
            json.dump(split_data, f, indent=2)

        print(f"[OfficeHomeSplit] Split saved → {SPLIT_FILE}")

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, label
