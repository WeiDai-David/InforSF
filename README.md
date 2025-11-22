Fisher Information For Source-Free Transfer Learning

## notebook：
### 1
pip install -U huggingface_hub
huggingface-cli login 
(if error: export HF_ENDPOINT=https://hf-mirror.com 国内代理)
(url: https://huggingface.co/settings/tokens figure out key)
python script/download_weights.py (Transformers 自动从本地读取token)
### 2
officehome datasets download：https://www.hemanthdv.org/officeHomeDataset.html