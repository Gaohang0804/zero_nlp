#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen1.5-0.5B-Chat', cache_dir='/home/zee001-w/1TB_DISK/Codes/zero_nlp/internlm-sft/model')