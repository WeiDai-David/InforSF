# script/download_weights.py

import os
from modelscope import snapshot_download
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoModel,
    AutoTokenizer
)

def dir_has_files(path):
    return os.path.exists(path) and len(os.listdir(path)) > 0


def main():
    print("=== Downloading model weights to local folders (with existence check) ===")

    os.makedirs("weights", exist_ok=True)

    # ---------------------------------------------------------
    # 1) CLIP
    # ---------------------------------------------------------
    clip_path = "weights/clip"
    if dir_has_files(clip_path):
        print(f"[1/2] CLIP already exists in {clip_path}, skipping download.")
    else:
        print("\n[1/2] Downloading CLIP ViT-L/14 ...")
        os.makedirs(clip_path, exist_ok=True)

        CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=clip_path
        )
        CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=clip_path
        )
        print(f"CLIP saved to: {clip_path}")

    # ---------------------------------------------------------
    # 2) LLaMA-2-7B (ModelScope)
    # ---------------------------------------------------------
    llama_path = "weights/llama2-7b-ms"
    if dir_has_files(llama_path):
        print(f"\n[2/2] LLaMA-2-7B already exists in {llama_path}, skipping download.")
    else:
        print("\n[2/2] Downloading LLaMA-2-7B (ModelScope) ...")
        os.makedirs(llama_path, exist_ok=True)

        llama_path = snapshot_download(
            model_id="modelscope/Llama-2-7b-ms",
            cache_dir=llama_path,
            revision="master"
        )
        print(f"LLaMA-2-7B saved to: {llama_path}")

    # ---------------------------------------------------------
    # 3) Verify (only if llama exists)
    # ---------------------------------------------------------
    print("\nVerifying LLaMA-2 loading with Transformers...")

    tok = AutoTokenizer.from_pretrained("weights/llama2-7b-ms", trust_remote_code=True, local_files_only=True)
    mdl = AutoModel.from_pretrained("weights/llama2-7b-ms", trust_remote_code=True, local_files_only=True)

    print("LLaMA-2-7B verified successfully!")

    print("\n=== All weights are ready locally ===")


if __name__ == "__main__":
    main()
