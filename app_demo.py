"""
Qwen3-VL + LLM 控制注入 → EG3D 3D人脸生成
Gradio 交互式演示
===========================================
运行: python app_demo.py
然后在浏览器打开 http://localhost:7860
"""
import os, sys, math
import torch
import gradio as gr
import numpy as np
from PIL import Image as PILImage
import torchvision.transforms as T

# 确保在 bridge.py 所在目录
sys.path.insert(0, "/root/Qwen3-VL")

from bridge import QwenEG3DBridge

# ============================================================
# 加载 Bridge
# ============================================================
print("加载 Bridge ...")
bridge = QwenEG3DBridge()

# 加载最新 checkpoint (数值排序)
ckpt_dir = "/root/Qwen3-VL/checkpoints"
ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('mm_') and f.endswith('.pt')],
               key=lambda x: int(x.split('_')[-1].split('.')[0]))
if ckpts:
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    ckpt = torch.load(ckpt_path, map_location='cuda')
    bridge.fusion.load_state_dict(ckpt['fusion'])
    bridge.pose.load_state_dict(ckpt['pose'])
    print(f"加载 checkpoint: {ckpt_path}")
print("Bridge 就绪!\n")


# ============================================================
# 生成函数
# ============================================================
def generate_3d_face(text_prompt, seed, truncation, show_gates):
    """
    文本 → 3D人脸生成
    """
    try:
        # 设置随机种子
        torch.manual_seed(int(seed))

        # 调用 Bridge (纯文本模式, 无图片)
        with torch.no_grad():
            img_tensor, cam_tensor = bridge(text_prompt, truncation_psi=truncation)
            yaw, pitch, radius, fov = cam_tensor[0, :4].data.cpu().numpy()

        # EG3D 输出: [1, 3, 512, 512], [-1, 1] → [0, 255]
        img = img_tensor[0].data.cpu().numpy()           # [3, 512, 512]
        img = np.transpose(img, (1, 2, 0))               # [512, 512, 3]
        img = ((img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

        # 门控信息
        if show_gates:
            # 纯文本模式: 不经过 fusion，门控为 0
            gates_str = "门控: low=0.000  mid=0.000  high=0.000 (纯文本模式)"
        else:
            gates_str = ""

        info = (
            f"yaw={yaw:+.2f}  pitch={pitch:+.2f}  "
            f"radius={radius:.2f}  fov={fov:.1f}°\n"
            f"{gates_str}"
        )

        return PILImage.fromarray(img), info

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"错误: {str(e)}"


def generate_with_image(text_prompt, image_input, seed):
    """
    图文模式: 参考图片 + 文本描述 → 3D人脸
    图片提供初始身份特征，文本控制视角
    """
    try:
        import tempfile
        torch.manual_seed(int(seed))

        # 保存上传的图片为临时文件
        tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        image_input.save(tmp_img.name)

        with torch.no_grad():
            img_tensor, cam_tensor, gates = bridge(
                text_prompt, image_path=tmp_img.name, return_gates=True
            )

        os.unlink(tmp_img.name)

        yaw, pitch, radius, fov = cam_tensor[0, :4].data.cpu().numpy()
        img = img_tensor[0].data.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

        info = (
            f"yaw={yaw:+.2f}  pitch={pitch:+.2f}  "
            f"radius={radius:.2f}  fov={fov:.1f}°\n"
            f"门控: low={gates[0]:.3f}  mid={gates[1]:.3f}  high={gates[2]:.3f}"
        )

        return PILImage.fromarray(img), info

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"错误: {str(e)}"


# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks(title="Qwen3-VL + EG3D 3D人脸生成", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Qwen3-VL + EG3D 3D人脸生成

    **文本描述 → LLM控制多尺度视觉注入 → 6D位姿 → EG3D 3D渲染**

    输入文字描述视角，即可生成对应角度的3D人脸。
    LLM会根据语义自动选择最合适的视觉特征层次（low/mid/high）。
    """)

    with gr.Tab("纯文本模式"):
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="视角描述",
                    value="正面平视",
                    placeholder="输入如: 左侧45度俯视, 右侧90度平视, 背面仰视...",
                )
                seed_slider = gr.Slider(
                    label="随机种子 (控制身份)", minimum=0, maximum=999, value=42, step=1
                )
                trunc_slider = gr.Slider(
                    label="截断系数 (truncation_psi)", minimum=0.3, maximum=1.0, value=0.7, step=0.05
                )
                show_gates_cb = gr.Checkbox(label="显示门控权重", value=True)
                submit_btn = gr.Button("🎯 生成 3D 人脸", variant="primary")

            with gr.Column(scale=2):
                image_output = gr.Image(label="生成的人脸", type="pil", height=512)
                info_output = gr.Textbox(label="相机参数", lines=3)

        submit_btn.click(
            fn=generate_3d_face,
            inputs=[text_input, seed_slider, trunc_slider, show_gates_cb],
            outputs=[image_output, info_output],
        )

        gr.Examples(
            examples=[
                ["正面平视", 42, 0.7],
                ["正面俯视", 42, 0.7],
                ["正面仰视", 42, 0.7],
                ["左侧45度俯视", 42, 0.7],
                ["左侧45度仰视", 42, 0.7],
                ["右侧45度平视", 42, 0.7],
                ["左侧90度侧面", 42, 0.7],
                ["右侧90度侧面", 42, 0.7],
                ["背面", 42, 0.7],
            ],
            inputs=[text_input, seed_slider, trunc_slider],
        )

    with gr.Tab("图文模式 (参考图片+文本)"):
        gr.Markdown("""
        上传一张参考人脸照片，输入文本描述来控制视角。
        LLM 会从参考图片提取多尺度视觉特征，并通过门控自动加权融合。
        """)
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(label="参考图片", type="pil", height=256)
                text_input2 = gr.Textbox(
                    label="视角描述",
                    value="右侧45度平视",
                )
                seed_slider2 = gr.Slider(
                    label="随机种子", minimum=0, maximum=999, value=42, step=1
                )
                submit_img_btn = gr.Button("🎯 图文生成 3D 人脸", variant="primary")

            with gr.Column(scale=2):
                img_output2 = gr.Image(label="生成的人脸", type="pil", height=512)
                info_output2 = gr.Textbox(label="相机参数与门控", lines=3)

        submit_img_btn.click(
            fn=generate_with_image,
            inputs=[text_input2, img_input, seed_slider2],
            outputs=[img_output2, info_output2],
        )

    if os.path.exists("/root/output/all_views_grid.png"):
        with gr.Tab("效果展示"):
            gr.Markdown("### 全视角生成效果 (21种视角)")
            gr.Image("/root/output/all_views_grid.png", height=600)

    gr.Markdown("""
    ---
    **架构**: Qwen3-VL (4bit) → 多尺度ViT特征(Layer 5/11/17) + LLM hidden → 
    LLMControlledFeatureFusion (门控加权) → PoseMappingLayer → 6D位姿 → EG3D (BF16)
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
