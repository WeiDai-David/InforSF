# config.py

class Config:
    # ---------- MODEL ----------
    llama_model = "meta-llama/Llama-2-7b-hf"
    vision_model = "openai/clip-vit-large-patch14"
    num_classes = 65

    # ---------- TRAIN ----------
    batch_size = 8
    lr = 1e-5
    epochs = 10

    # ---------- PATH ----------
    office_home_root = "data/office-home/Art"

    # ---------- TRANSFER ----------
    N1 = 10.0
    fisher_samples = 128
    fisher_scale = 0.01   # 最终 1%
