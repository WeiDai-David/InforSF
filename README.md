# Fisher Information For Source-Free Transfer Learning

Model Mind:CLIP 视觉特征 → 映射到 LLaMA hidden size → LLaMA backbone 作为深度特征变换器 → 分类头

# 📝 Notebook：
### 1
```bash 
pip install -U huggingface_hub
huggingface-cli login 
(if error: export HF_ENDPOINT=https://hf-mirror.com 国内代理)
(url: https://huggingface.co/settings/tokens figure out key)
python script/download_weights.py (Transformers 自动从本地读取token)
```

### 2
```bash 
officehome datasets download：https://www.hemanthdv.org/officeHomeDataset.html
```