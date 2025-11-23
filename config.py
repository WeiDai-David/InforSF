# config.py

import torch
import os

class Config:

    def __init__(self):

        # ---------------------------------------------------------
        # Paths (Windows + Linux compatible)
        # ---------------------------------------------------------
        self.DATA_ROOT = "data/officehome"


        # weights
        self.CLIP_WEIGHTS = "weights/clip"
        self.LLAMA_WEIGHTS = "weights/llama2-7b-hf"

        # output folder
        self.OUTPUT_DIR = "output"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # ---------------------------------------------------------
        # Splitting Hyperparameters
        # ---------------------------------------------------------
        self.DOMAIN = None          # 或 "Art" / ["Art","Product"]
        self.TRAIN_RATIO = 0.8
        self.SEED = 42

        # ---------------------------------------------------------
        # Training Hyperparameters
        # ---------------------------------------------------------
        self.NUM_WORKERS = 0
        self.BATCH_SIZE = 16
        self.LR = 1e-4
        self.EPOCHS = 5

        # ---------------------------------------------------------
        # InfoSF Loss Hyperparameters
        # ---------------------------------------------------------
        self.N1 = 50      # dynamic weight aggressiveness

        # ---------------------------------------------------------
        # Fisher Hyperparameters
        # ---------------------------------------------------------
        self.FISHER_BATCHES = 3   # # of batches to estimate Fisher
        self.FISHER_RATIO = 0.01  # 1% parameter sampling

        # ---------------------------------------------------------
        # Model Hyperparameters
        # ---------------------------------------------------------
        self.NUM_CLASSES = 65
        self.CLIP_FEATURE_DIM = 1024

        # LLaMA backbone freezing
        self.FREEZE_LLAMA = True

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Print summary if needed
        self.print_config()


    def print_config(self):
        print("\n========== InfoSF Configuration ==========")
        print(f"Device             : {self.DEVICE}")
        print(f"Dataset Root       : {self.DATA_ROOT}")
        print(f"CLIP Weights       : {self.CLIP_WEIGHTS}")
        print(f"LLaMA Weights      : {self.LLAMA_WEIGHTS}")
        print(f"Output Dir         : {self.OUTPUT_DIR}")
        print(f"Domain(None is All): {self.DOMAIN}")
        print(f"Train ratio        : {self.TRAIN_RATIO}")
        print(f"Number of workes   : {self.NUM_WORKERS}")
        print(f"Batch Size         : {self.BATCH_SIZE}")
        print(f"Learning Rate      : {self.LR}")
        print(f"Epochs             : {self.EPOCHS}")
        print(f"N1 (Dynamic Weight): {self.N1}")
        print(f"Fisher Batches     : {self.FISHER_BATCHES}")
        print(f"Fisher Ratio       : {self.FISHER_RATIO}")
        print(f"Num Classes        : {self.NUM_CLASSES}")
        print(f"CLIP Feature Dim   : {self.CLIP_FEATURE_DIM}")
        print(f"Freeze LLaMA       : {self.FREEZE_LLAMA}")
        print("===========================================\n")


# global config object
cfg = Config()
