"""
Lightweight Bridge: CLIP/BERT/XLM → PoseMappingLayer → 6D → EG3D Camera
用于对比 VLM (Qwen3-VL 2B) vs 轻量文本编码器的精度差异
"""
import os, sys, math, torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/Qwen3-VL")
from bridge import load_eg3d, pose_6d_to_camera_params

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda-12.8")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6")

_OFFLINE_KWARGS = {"local_files_only": True}

_TEXT_ENCODER_INFO = {
    "clip": {
        "path": "/root/models/clip-vit-base-patch32",
        "hidden_size": 512,
    },
    "bert": {
        "path": "/root/models/bert-base-uncased",
        "hidden_size": 768,
    },
    "xlm": {
        "path": "/root/models/xlm-roberta-base",
        "hidden_size": 768,
    },
}


def load_text_encoder(name="clip"):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    name = name.lower()
    model_path = _TEXT_ENCODER_INFO[name]["path"]

    if name == "clip":
        from transformers import CLIPTextModel, CLIPTokenizer
        model = CLIPTextModel.from_pretrained(model_path, **_OFFLINE_KWARGS)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, **_OFFLINE_KWARGS)
    elif name == "bert":
        from transformers import BertModel, BertTokenizer
        model = BertModel.from_pretrained(model_path, **_OFFLINE_KWARGS)
        tokenizer = BertTokenizer.from_pretrained(model_path, **_OFFLINE_KWARGS)
    elif name == "xlm":
        from transformers import XLMRobertaModel, XLMRobertaTokenizer
        model = XLMRobertaModel.from_pretrained(model_path, **_OFFLINE_KWARGS)
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_path, **_OFFLINE_KWARGS)
    else:
        raise ValueError(f"Unknown encoder: {name}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] loaded: {n_params/1e6:.0f}M params, hidden={_TEXT_ENCODER_INFO[name]['hidden_size']}")
    return model, tokenizer


def get_text_embedding(model, tokenizer, text, name="clip"):
    """
    Get [CLS]/pooler embedding from text encoder.
    Returns: [1, hidden_size] float32
    """
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    if name == "clip":
        emb = outputs.pooler_output  # [1, 512]
    elif name in ("bert", "xlm"):
        emb = outputs.pooler_output  # [1, 768]
    else:
        emb = outputs.last_hidden_state[:, 0, :]  # fallback [CLS]
    
    return emb.float()


class TextProjection(nn.Module):
    """Map encoder hidden → 2048 for PoseMappingLayer"""
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.LayerNorm(2048),
        )
    def forward(self, x):
        return self.net(x)


class LightweightBridge(nn.Module):
    def __init__(self, encoder_name="clip",
                 eg3d_pkl="/root/eg3d/networks/nvidia_ffhqrebalanced512-128.pkl",
                 pose_ckpt=None):
        super().__init__()
        self.encoder_name = encoder_name
        hidden_size = _TEXT_ENCODER_INFO[encoder_name]["hidden_size"]
        
        self.text_model, self.tokenizer = load_text_encoder(encoder_name)
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.text_model.cuda()
        
        self.projection = TextProjection(hidden_size).cuda()
        self.pose = nn.Sequential(
            nn.Linear(2048, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 6),
        )
        self.pose = self.pose.cuda()
        for m in self.pose.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.5)
                nn.init.zeros_(m.bias)
        
        self.eg3d = load_eg3d(eg3d_pkl)
        
        if pose_ckpt is not None:
            ckpt = torch.load(pose_ckpt, map_location="cuda")
            self.projection.load_state_dict(ckpt["projection"])
            self.pose.load_state_dict(ckpt["pose"])
            print(f"[{encoder_name}] loaded checkpoint: {pose_ckpt}")
        
        n_train = sum(p.numel() for p in [*self.projection.parameters(), *self.pose.parameters()])
        print(f"[Bridge {encoder_name}] trainable: {n_train:,} params")
    
    def forward(self, text, truncation_psi=0.7, return_raw_6d=False, w_plus=None):
        emb = get_text_embedding(self.text_model, self.tokenizer, text, self.encoder_name)
        h = self.projection(emb)
        raw_6d = self.pose(h)
        cam = pose_6d_to_camera_params(raw_6d)
        
        if w_plus is not None:
            ws = w_plus.to('cuda')
        else:
            z = torch.randn(1, self.eg3d.z_dim, device='cuda')
            ws = self.eg3d.mapping(z, cam, truncation_psi=truncation_psi)
        
        with torch.no_grad():
            img = self.eg3d.synthesis(ws, cam)['image']
        
        out = (img, cam)
        if return_raw_6d:
            out = out + (raw_6d.detach(),)
        return out


def get_embedding_for_training(bridge, text):
    with torch.no_grad():
        emb = get_text_embedding(
            bridge.text_model, bridge.tokenizer, text, bridge.encoder_name
        )
    return emb
