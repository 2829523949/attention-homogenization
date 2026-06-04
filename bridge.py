"""
Phase 1 Bridge: Qwen3-VL (4bit) → LLM hidden [2048] → PoseMappingLayer → 6D → EG3D Camera
纯文本控制人脸角度, 不涉及 DINOv2/injector/fusion/adapter.
"""
import os, torch, torch.nn as nn
import torch.nn.functional as F
import math

# CUDA 环境 (用于 EG3D 插件编译)
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda-12.8")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6")
import torch.utils.cpp_extension as _cpp_ext
_cpp_ext.CUDA_HOME = "/usr/local/cuda-12.8"


# ===== 1. Qwen3-VL (4bit) =====
_MODEL_PATH = "/root/Qwen3-VL-2B-Instruct"

def load_qwen():
    import os, glob
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    assert os.path.isdir(_MODEL_PATH), f"模型目录不存在: {_MODEL_PATH}"
    configs = glob.glob(f"{_MODEL_PATH}/config*.json")
    safetensors = glob.glob(f"{_MODEL_PATH}/*.safetensors") + glob.glob(f"{_MODEL_PATH}/model*.bin")
    print(f"[Qwen] 目录: {_MODEL_PATH}")
    print(f"[Qwen] config: {len(configs)}个 | 权重: {len(safetensors)}个")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForImageTextToText.from_pretrained(
        _MODEL_PATH, quantization_config=bnb_config,
        device_map="cuda", torch_dtype=torch.float16, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        _MODEL_PATH, local_files_only=True, trust_remote_code=True)
    print(f"[Qwen] loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B (4bit)")
    return model, processor


def get_llm_hidden(qwen, processor, text):
    """纯文本 -> LLM last hidden [2048]
    取最后一个 token (</s>) 的 hidden_state.
    使用 tokenizer 而非 processor, 避免 Qwen3VL 误将文本当图片处理.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor.tokenizer(prompt, return_tensors="pt").to('cuda')
    with torch.no_grad():
        out = qwen(**inputs, output_hidden_states=True)
    return out.hidden_states[-1][0, -1, :2048].float()


# ===== 2. EG3D =====
def load_eg3d(pkl_path):
    import sys
    sys.path.insert(0, "/root/next3d")
    import dnnlib, legacy, torch_utils
    with dnnlib.util.open_url(pkl_path) as f:
        return legacy.load_network_pkl(f)['G_ema'].to('cuda').eval()


# ===== 3. PoseMappingLayer =====
class PoseMappingLayer(nn.Module):
    def __init__(self, hidden_size=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 6),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ===== 4. 6D -> Camera (EG3D兼容版, 从prepare_wplus.py移植) =====
def pose_6d_to_camera_params(raw_6d):
    """
    6D -> EG3D相机参数 [cam2world(16), intrinsics(9)].
    坐标系: y-up, z-forward, x-left
    内参: 归一化坐标 [0,1]
    
    6D定义:
      [0] yaw:      水平角, 0=正面, +右 -左
      [1] pitch:    极角, pi/2=平视, <pi/2=俯视 >pi/2=仰视
      [2] radius:   相机到原点距离
      [3] fov:      视场角(度)
      [4] lx:       lookat x偏移
      [5] lz:       lookat z偏移
    """
    B, device = raw_6d.shape[0], raw_6d.device
    yaw, pitch = raw_6d[:, 0], raw_6d[:, 1]
    radius = raw_6d[:, 2].clamp(min=1.0)
    fov = raw_6d[:, 3].clamp(min=5.0)
    lx, lz = torch.tanh(raw_6d[:, 4]) * 0.3, torch.tanh(raw_6d[:, 5]) * 0.3

    # lookat点
    lookat = torch.stack([lx, torch.zeros(B, device=device), lz], dim=1)

    # 球面 -> 笛卡尔 (与prepare_wplus.py/EG3D一致)
    origin = torch.zeros((B, 3), device=device)
    origin[:, 0] = radius * torch.sin(pitch) * torch.cos(math.pi / 2 - yaw)
    origin[:, 1] = radius * torch.cos(pitch)
    origin[:, 2] = radius * torch.sin(pitch) * torch.sin(math.pi / 2 - yaw)

    # cam2world (与EG3D camera_utils.py一致)
    forward = F.normalize(lookat - origin, dim=1)
    up_world = torch.tensor([[0.0, 1.0, 0.0]], device=device).expand(B, 3)
    right = F.normalize(-torch.linalg.cross(up_world, forward, dim=1), dim=1)
    up = F.normalize(torch.linalg.cross(forward, right, dim=1), dim=1)

    cam2world = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
    cam2world[:, :3, :3] = torch.stack([right, up, forward], dim=-1)
    cam2world[:, :3, 3] = origin

    # 归一化坐标内参 (与EG3D dataset_preprocessing完全一致)
    focal = 1.0 / (torch.tan(fov * math.pi / 360) * 1.414)
    intrinsics = torch.zeros((B, 3, 3), device=device)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5
    intrinsics[:, 2, 2] = 1.0

    return torch.cat([cam2world.reshape(B, 16), intrinsics.reshape(B, 9)], dim=1)


# ===== 5. Bridge =====
class QwenEG3DBridge(nn.Module):
    def __init__(self, eg3d_pkl="/root/eg3d/networks/nvidia_ffhqrebalanced512-128.pkl",
                 pose_ckpt="/root/Qwen3-VL/checkpoints/mix_v2_epoch_040.pt"):
        super().__init__()
        self.qwen, self.processor = load_qwen()
        self.eg3d = load_eg3d(eg3d_pkl)
        self.pose = PoseMappingLayer().to('cuda')
        ckpt = torch.load(pose_ckpt, map_location="cuda")
        self.pose.load_state_dict(ckpt["pose"])
        print(f"[Bridge] Phase1 ready: {sum(p.numel() for p in self.parameters() if p.requires_grad):,} params")

    def forward(self, text, truncation_psi=0.7, return_raw_6d=False, w_plus=None):
        hidden = get_llm_hidden(self.qwen, self.processor, text)
        raw_6d = self.pose(hidden.unsqueeze(0))
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
