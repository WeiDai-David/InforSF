# train.py

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset.office_home import OfficeHomeDataset
from vision.clip_encoder import CLIPFeatureExtractor
from models.llama_classifier import LLaMAClassifier
from trainer.trainer_dynamic import DynamicTrainer

def main():
    cfg = Config()

    dataset = OfficeHomeDataset(cfg.office_home_root)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    feature_extractor = CLIPFeatureExtractor(cfg.vision_model).cuda()

    model_t = LLaMAClassifier(cfg.llama_model, 1024, cfg.num_classes).cuda()
    model_init = LLaMAClassifier(cfg.llama_model, 1024, cfg.num_classes).cuda()
    model_init.load_state_dict(model_t.state_dict())  # freeze initial

    trainer = DynamicTrainer(model_t, model_init, feature_extractor, loader, cfg)
    trainer.train()

if __name__ == "__main__":
    main()
