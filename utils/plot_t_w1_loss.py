# utils/plot_t_w1_loss.py

import json
import matplotlib.pyplot as plt
import os

def plot_infoSF_curves(log_path, save_path="output/plots.png"):
    """
    Plot CE, Ratio, t, and w1 curves from JSON log file.

    JSON format:
    {
        "epochs": [1,2,3,...],
        "ce": [0.8,0.6,...],
        "ratio": [...],
        "t": [...],
        "w1": [...]
    }
    """

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with open(log_path, "r") as f:
        log = json.load(f)

    epochs = log["epochs"]
    ce = log["ce"]
    ratio = log["ratio"]
    t = log["t"]
    w1 = log["w1"]

    plt.figure(figsize=(12, 8))

    # -----------------------------
    # Loss subplot
    # -----------------------------
    plt.subplot(2, 2, 1)
    plt.plot(epochs, ce, marker="o")
    plt.title("CE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 2)
    plt.plot(epochs, ratio, marker="o", color="orange")
    plt.title("Ratio Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # -----------------------------
    # Fisher Distance subplot
    # -----------------------------
    plt.subplot(2, 2, 3)
    plt.plot(epochs, t, marker="o", color="green")
    plt.title("Fisher Distance t")
    plt.xlabel("Epoch")

    # -----------------------------
    # w1 dynamic weight subplot
    # -----------------------------
    plt.subplot(2, 2, 4)
    plt.plot(epochs, w1, marker="o", color="red")
    plt.title("Dynamic Weight w1")
    plt.xlabel("Epoch")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[✓] Plot saved to {save_path}")
