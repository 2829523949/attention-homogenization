
"""
Qwen3-VL → 多尺度视觉 + LLM 控制注入 → EG3D 3D Generation Bridge
================================================================
Pipeline (图文模式):
  图片 → ViT(24层) → deepstack[5,11,17] (低/中/高层语义特征)
  文本 → LLM(28层) → hidden state (2048d)
                        ↓
                  LLM 控制注入门控 (预测各层特征权重)
                        ↓
         fused = sum(gate[i] × deepstack[i]) + LLM_hidden
                        ↓
                  PoseMappingLayer → 6D → camera [25] → EG3D

Pipeline (纯文本模式):
  文本 → LLM hidden → PoseMappingLayer → 6D → camera → EG3D

可训练: LLMControlledFeatureFusion (13.1M) + PoseMappingLayer (2.6M) = 15.7M
冻结:   Qwen3-VL (4bit) + EG3D (BF16)
"""
import os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# CUDA 编译修复
# ============================================================
_CUDA = "/root/miniconda3/envs/eg3d"
import torch.utils.cpp_extension as _cpp_ext
_cpp_ext.CUDA_HOME = _CUDA
os.environ["CUDA_HOME"] = _CUDA
os.environ["CUDA_PATH"] = _CUDA
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
os.environ["CUDAHOSTCXX"] = "/usr/bin/gcc-12"
os.environ["CXX"] = "/usr/bin/g++-12"
os.environ["CC"] = "/usr/bin/gcc-12"
os.environ["CPATH"] = _CUDA + "/targets/x86_64-linux/include:" + os.environ.get("CPATH", "")
os.environ["PATH"] = _CUDA + "/nvvm/bin:" + _CUDA + "/bin:" + os.environ.get("PATH", "")
_orig_load = _cpp_ext.load
def _patched_load(name, sources, **kwargs):
    extra_cflags = kwargs.get('extra_cflags', [])
    extra_cflags = list(extra_cflags)
    extra_cflags.append('-I'+ _CUDA + '/targets/x86_64-linux/include')
    kwargs['extra_cflags'] = extra_cflags
    extra_cuda = kwargs.get('extra_cuda_cflags', [])
    extra_cuda = list(extra_cuda)
    extra_cuda.append('-I'+ _CUDA + '/targets/x86_64-linux/include')
    if '-allow-unsupported-compiler' not in extra_cuda:
        extra_cuda.append('-allow-unsupported-compiler')
    kwargs['extra_cuda_cflags'] = extra_cuda
    return _orig_load(name, sources, **kwargs)
_cpp_ext.load = _patched_load


# ============================================================
# 第1部分：加载 Qwen3-VL (4bit)
# ============================================================
def load_qwen():
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", quantization_config=bnb_config, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("/root/Qwen3-VL-2B-Instruct")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(f"[Qwen] Loaded | VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    return model, processor


# ============================================================
# 第1.5部分：加载 DINOv2 (替换 ViT deepstack, 204M)
# ============================================================
DINO_DEEPSTACK_LAYERS = [5, 11, 17]

def load_dinov2():
    """加载 DINOv2-large (不含分类头)"""
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from transformers import Dinov2Model
    dino = Dinov2Model.from_pretrained("facebook/dinov2-large", local_files_only=True).to('cuda').eval()
    for p in dino.parameters():
        p.requires_grad = False
    print(f"[DINOv2] Loaded | layers={DINO_DEEPSTACK_LAYERS} | VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    return dino


# ============================================================
# 第2部分：特征提取（LLM hidden + 多尺度视觉 DeepStack）
# ============================================================
def get_llm_hidden(qwen, processor, text):
    """文本 → Qwen LLM 最后一层 hidden state (2048d)"""
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], return_tensors="pt").to(qwen.device)
    with torch.no_grad():
        outputs = qwen(**inputs, output_hidden_states=True, return_dict=True)
    return outputs.hidden_states[-1][0, -1, :].float()


def get_multimodal_features(qwen, processor, text, image_path, dino=None, dino_processor=None):
    """
    图文输入 → DINOv2 多尺度特征 (deepstack) + LLM hidden state
    返回: (deepstack_features [3], llm_hidden [2048])
    deepstack 来自 DINOv2[5,11,17], 每层 [N_patches, 1024]
    """
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": text},
    ]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image_path], return_tensors="pt", padding=True).to(qwen.device)

    with torch.no_grad():
        # Qwen: LLM hidden (用于文本理解)
        outputs = qwen(**inputs, output_hidden_states=True, return_dict=True)
        llm_hidden = outputs.hidden_states[-1][0, -1, :].float()

        # DINOv2: 多尺度视觉特征 (独立处理图像)
        if dino is not None and dino_processor is not None:
            from PIL import Image
            pil_img = Image.open(image_path).convert('RGB')
            dino_inputs = dino_processor(images=pil_img, return_tensors="pt").to(qwen.device)
            pv = dino_inputs['pixel_values']
            dino_out = dino(pv, output_hidden_states=True)
            hs = dino_out.hidden_states
            deepstack_features = [hs[i][0, 1:, :].float() for i in DINO_DEEPSTACK_LAYERS]
            # 每层 [N_patches, 1024]
        else:
            deepstack_features = None

    return deepstack_features, llm_hidden


# ============================================================
# 第3部分：LLM 控制的多尺度视觉注入门控
# ============================================================
class LLMControlledFeatureFusion(nn.Module):
    """
    LLM hidden state 预测门控权重，控制各层 ViT 特征的融合。
    
    这就是"LLM 控制注入"的核心:
      text = "一张正面的人脸"  → LLM → gates=[0.3, 0.4, 0.3] (均衡)
      text = "一张侧面纹理"   → LLM → gates=[0.7, 0.2, 0.1] (更重低层纹理)
      text = "戴墨镜的人"     → LLM → gates=[0.1, 0.3, 0.6] (更重高层语义)
    """
    def __init__(self, hidden_size=2048, num_deep_layers=3):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, num_deep_layers),
        )
        # 纯文本模式时，LLM hidden → 虚拟视觉特征
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        # 全层拼接投影: 确保所有三层信息都被利用
        self.fusion_proj = nn.Linear(hidden_size * num_deep_layers, hidden_size)

    def forward(self, llm_hidden, deepstack_features=None):
        """
        Args:
            llm_hidden: [2048] LLM 语义编码
            deepstack_features: [3 × N_v, 2048] 或 None (纯文本模式)
        Returns:
            fused: [2048] 融合后的特征
            gates: [3] 门控权重
        """
        # LLM 预测门控
        gates = torch.softmax(self.gate_net(llm_hidden), dim=0)  # [3]

        if deepstack_features is not None:
            # 图文模式: 用 LLM 门控加权多尺度视觉特征
            pooled = torch.stack([f.mean(dim=0).float() for f in deepstack_features])  # [3, 2048]
            gated_fused = (gates.view(3, 1) * pooled).sum(dim=0)  # [2048]
            # 全层拼接投影: 强制使用所有三层信息
            all_concat = torch.cat([f.mean(dim=0).float() for f in deepstack_features])  # [3*2048]
            all_fused = self.fusion_proj(all_concat)  # [2048]
            fused = (gated_fused + all_fused).float()  # 门控选择 + 全层融合
        else:
            # 纯文本模式: 生成 3 个不同的"伪视觉特征"，让门控有意义
            virtual = self.text_proj(llm_hidden).unsqueeze(0).expand(3, -1)  # [3, 2048]
            # 添加位置编码区分三层
            positions = torch.tensor([[0.0], [1.0], [2.0]], device=llm_hidden.device)
            virtual = virtual + positions * 0.1  # 轻微区分
            fused = (gates.view(3, 1) * virtual).sum(dim=0)  # [2048]

        return llm_hidden + fused, gates  # 残差连接


# ============================================================
# 第3.5部分：Gated Residual Adapter (多样化文本 → 统一隐空间)
# ============================================================
class GatedResidualAdapter(nn.Module):
    """
    门控残差适配器: 将不同词序的 LLM hidden 映射到统一的隐空间,
    保持输入维度不变, 可插拔地叠加在 PoseMappingLayer 前。
    
    架构:
        Input [2048] → LayerNorm → Down(2048→256) × Gate(2048→256, sigmoid)
        → GELU → Up(256→2048) + Residual
    """
    def __init__(self, dim=2048, bottleneck=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.gate = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        
        # 初始化: 接近恒等映射 (残差小→初始≈原行为)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        gate = torch.sigmoid(self.gate(x))  # 门控: 哪些维度需要调整
        x = F.gelu(self.down(x))            # 瓶颈映射
        x = x * gate                         # 门控筛选
        x = self.up(x)                       # 还原维度
        return x + residual                   # 残差连接 (初始≈恒等)


# ============================================================
# 第4部分：融合特征 → 6D pose
# ============================================================
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
        return self.net(x)  # [B, 6]


# ============================================================
# 第5部分：6D → EG3D camera params [25]
# ============================================================
def pose_6d_to_camera_params(raw_6d, device='cuda'):
    B = raw_6d.shape[0]
    device = raw_6d.device  # 使用输入张量的设备
    yaw, pitch = raw_6d[:, 0], raw_6d[:, 1]
    radius = raw_6d[:, 2].clamp(min=1.0)  # 直接使用物理半径值 (不含 sigmoid)
    fov = raw_6d[:, 3].clamp(min=5.0)     # 直接使用物理FOV角度值 (不含 sigmoid)
    lx, lz = torch.tanh(raw_6d[:, 4]) * 0.3, torch.tanh(raw_6d[:, 5]) * 0.3

    lookat = torch.stack([lx, torch.zeros(B, device=device), lz], dim=1)
    origin = torch.zeros((B, 3), device=device)
    origin[:, 0] = radius * torch.sin(pitch) * torch.cos(math.pi / 2 - yaw)
    origin[:, 1] = radius * torch.cos(pitch)
    origin[:, 2] = radius * torch.sin(pitch) * torch.sin(math.pi / 2 - yaw)

    # 关键修复: forward 应指向 lookat (EG3D 官方约定)
    forward = F.normalize(lookat - origin, dim=1)
    up = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(B, 3)
    right = F.normalize(-torch.cross(up, forward, dim=1), dim=1)
    up = F.normalize(torch.cross(forward, right, dim=1), dim=1)  # 也加 normalize

    rot = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
    rot[:, :3, :3] = torch.stack([right, up, forward], dim=-1)
    trans = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
    trans[:, :3, 3] = origin
    cam2world = trans @ rot

    focal = 1.0 / (torch.tan(fov * math.pi / 360) * 1.414)
    intrinsics = torch.zeros((B, 3, 3), device=device)
    intrinsics[:, 0, 0] = focal; intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = 0.5; intrinsics[:, 1, 2] = 0.5; intrinsics[:, 2, 2] = 1.0
    return torch.cat([cam2world.reshape(B, 16), intrinsics.reshape(B, 9)], dim=1)


# ============================================================
# 第6部分：加载 EG3D
# ============================================================
def load_eg3d(pkl_path="/root/eg3d/networks/nvidia_ffhqrebalanced512-128.pkl"):
    if "/root/eg3d/eg3d" not in sys.path:
        sys.path.insert(0, "/root/eg3d/eg3d")
    import legacy, dnnlib
    with dnnlib.util.open_url(pkl_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to('cuda')
    G.eval()
    for param in G.parameters():
        param.requires_grad = False
    print(f"[EG3D] Loaded | rendering_res={G.neural_rendering_resolution}")
    return G


# ============================================================
# 第6.5部分：DeepStack 注入器 (空间特征注入 EG3D Synthesis)
# ============================================================
# DINOv2 patch grid = 16×16 (224/14), 3层 → 注入3组分辨率
DS_SPATIAL_SIZE = 16  # DINOv2 的 patch grid 尺寸

class DeepStackInjector(nn.Module):
    def __init__(self, eg3d, dino_dim=1024):
        super().__init__()
        self.eg3d = eg3d
        synth = eg3d.backbone.synthesis
        self.block_resolutions = synth.block_resolutions
        
        # 每个 block: 1×1 Conv2d 把 DINOv2 空间特征投影到 block 通道数
        self.inject_convs = nn.ModuleDict()
        self.res_to_ds = {}  # res → deepstack_idx
        for res in self.block_resolutions:
            block = getattr(synth, f'b{res}')
            ch = block.conv1.out_channels
            conv = nn.Conv2d(dino_dim, ch, kernel_size=1)
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            self.inject_convs[str(res)] = conv
            self.add_module(f'inj_conv_{res}', conv)
            # 映射: 分辨率 → deepstack 层
            if res <= 16:
                self.res_to_ds[res] = 0
            elif res <= 64:
                self.res_to_ds[res] = 1
            else:
                self.res_to_ds[res] = 2

    def inject_planes(self, ws, deepstack_spatial, gates):
        """
        Args:
            deepstack_spatial: list[3] of [N_patches, C=1024] per DINOv2 layer
                             N_patches = 256 (16×16)
            gates: [3] 门控权重
        Returns:
            img: [1, 96, 256, 256] tri-plane
        """
        # 每层 deepstack → 空间 grid: [1, C, H, W]
        ds_maps = []
        for f in deepstack_spatial:
            n = f.shape[0]
            gs = int(n ** 0.5)  # 16
            ds_maps.append(f.permute(1, 0).reshape(1, -1, gs, gs))  # [1, 1024, 16, 16]

        synth = self.eg3d.backbone.synthesis
        x = img = None
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(synth, f'b{res}')
            ws_block = ws[:, w_idx:w_idx + block.num_conv + block.num_torgb]
            w_idx += block.num_conv
            x, img = block(x, img, ws_block)

            # 空间注入
            ds_idx = self.res_to_ds[res]
            feat = ds_maps[ds_idx]  # [1, 1024, 16, 16]
            # resize 到当前 block 分辨率
            if feat.shape[2] != x.shape[2] or feat.shape[3] != x.shape[3]:
                feat = F.interpolate(feat, size=(x.shape[2], x.shape[3]),
                                     mode='bilinear', align_corners=False)
            injection = self.inject_convs[str(res)](feat)  # [1, ch, H, W]
            x = x + gates[ds_idx] * injection

        return img  # [1, 96, 256, 256] tri-plane


# ============================================================
# 第7部分：Qwen3-VL ↔ EG3D Bridge
# ============================================================
class QwenEG3DBridge(nn.Module):
    """
    完整 Bridge: 文本/图文 → 多尺度ViT + LLM控制注入 → 6D → EG3D
    """
    def __init__(self, eg3d_pkl="/root/eg3d/networks/nvidia_ffhqrebalanced512-128.pkl"):
        super().__init__()
        self.qwen, self.processor = load_qwen()
        self.eg3d = load_eg3d(eg3d_pkl)
        self.dino = load_dinov2()
        from transformers import AutoImageProcessor
        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large", local_files_only=True)
        self.fusion = LLMControlledFeatureFusion().to('cuda')
        self.pose = PoseMappingLayer().to('cuda')
        self.adapter = GatedResidualAdapter().to('cuda')
        # DINOv2(1024) → Qwen(2048) 投影
        self.dino_proj = nn.Linear(1024, 2048).to('cuda')
        nn.init.zeros_(self.dino_proj.weight)
        nn.init.zeros_(self.dino_proj.bias)
        self.injector = DeepStackInjector(self.eg3d, dino_dim=1024).to('cuda')
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Bridge] 可训练参数: {n:,}")

    def forward(self, text, image_path=None, return_gates=False, truncation_psi=0.7, return_raw_6d=False, w_plus=None):
        """
        Args:
            text: 文本描述
            image_path: (可选) 图片路径 → 开启多尺度视觉 + LLM 控制注入
            return_gates: (可选) 返回门控权重用于分析
            truncation_psi: EG3D 截断系数 (0.3~1.0, 默认0.7)
            return_raw_6d: (可选) 返回原始6D位姿(用于调试/打印)
            w_plus: (可选) [1, 14, 512] w+ 编码 → 直接使用, 跳过 mapping
        """
        if image_path:
            # 图文模式: DINOv2 多尺度视觉 + LLM 控制
            raw_ds, llm_hidden = get_multimodal_features(
                self.qwen, self.processor, text, image_path,
                dino=self.dino, dino_processor=self.dino_processor
            )
            if raw_ds is not None:
                deep_features = [self.dino_proj(f) for f in raw_ds]  # [N_v, 1024]→[N_v, 2048] 用于 gate
            else:
                deep_features = None
            fused, gates = self.fusion(llm_hidden, deep_features)
        else:
            # 纯文本模式: LLM hidden → GatedAdapter(新) → PoseMappingLayer(预训练保留)
            llm_hidden = get_llm_hidden(self.qwen, self.processor, text)
            fused = self.adapter(llm_hidden)  # 适配器统一不同词序的隐空间分布
            gates = torch.zeros(3, device='cuda')

        raw_6d = self.pose(fused.unsqueeze(0))  # [1, 6]: yaw,pitch,radius_f, fov, lx, lz
        cam = pose_6d_to_camera_params(raw_6d)  # [1, 25]: cam2world(16)+intrinsics(9)

        if w_plus is not None:
            ws = w_plus.to('cuda')  # 直接使用 w+, 跳过 mapping, 身份完全不变
        else:
            z = torch.randn(1, self.eg3d.z_dim, device='cuda')
            ws = self.eg3d.mapping(z, cam, truncation_psi=truncation_psi)
        with torch.no_grad():
            img = self.eg3d.synthesis(ws, cam)['image']

        out = (img, cam)
        if return_gates:
            out = out + (gates,)
        if return_raw_6d:
            out = out + (raw_6d.detach(),)
        return out  # (img, cam) 或 (img, cam, gates) 或 (img, cam, gates, raw_6d)

    def infer_text_only(self, text, truncation_psi=0.7):
        """纯文本模式（无图片，经过 Adapter → PoseMappingLayer）"""
        hidden = get_llm_hidden(self.qwen, self.processor, text)
        hidden = self.adapter(hidden)  # 适配器统一分布
        raw_6d = self.pose(hidden.unsqueeze(0))
        cam = pose_6d_to_camera_params(raw_6d)
        with torch.no_grad():
            ws = self.eg3d.mapping(torch.randn(1, self.eg3d.z_dim, device='cuda'), cam, truncation_psi=truncation_psi)
            img = self.eg3d.synthesis(ws, cam)['image']
        return img, cam


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3-VL + 多尺度视觉 + LLM控制注入 → EG3D")
    print("=" * 60)
    bridge = QwenEG3DBridge()

    print("\n纯文本模式测试:")
    for t in ["正面平视", "左侧45度平视", "右侧90度俯视", "背面"]:
        img, cam = bridge(t)
        print(f"  '{t}' → img={img.shape} cam={cam.shape}")

    print(f"\n总显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB / 7.96 GB")
    print("✅ Bridge 运行成功!")
