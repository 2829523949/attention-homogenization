"""修改 EG3D: 强制禁用 CUDA 插件，使用 PyTorch fallback"""
import os

# 修改 custom_ops.py: 让 get_plugin 直接返回 None
ops_path = "/root/eg3d/eg3d/torch_utils/custom_ops.py"
with open(ops_path, 'r') as f:
    code = f.read()

# 备份
if not os.path.exists(ops_path + ".bak"):
    with open(ops_path + ".bak", 'w') as f:
        f.write(code)
    print("备份 custom_ops.py.bak")

# 替换 _init 方法
old = '''    try:
        _plugin = get_plugin(
            module_name='bias_act_plugin',
            sources=[source_path, header_path],
            headers=[header_path],
            source_timestamp=source_timestamp,
            extra_cflags=['/await', '-O2'],
            extra_cuda_cflags=['--use_fast_math'],
        )
        bias_act_cuda = _plugin.bias_act_cuda
    except:'''

new = '''    _plugin = None
    _plugin = get_plugin(
        module_name='bias_act_plugin',
        sources=[source_path, header_path],
        headers=[header_path],
        source_timestamp=source_timestamp,
        extra_cflags=['/await', '-O2'],
        extra_cuda_cflags=['--use_fast_math'],
    )
    try:
        bias_act_cuda = _plugin.bias_act_cuda
    except:'''

code = code.replace(old, new)

# 获取 _init 的 except 分支已存在，直接让编译失败时 _plugin 保持 None
# 更好: 在首次编译失败后不重试
code = code.replace(
    "if _plugin is None:",
    "if _plugin is None:\n        print('[EG3D] CUDA plugin 编译跳过，使用 PyTorch fallback')"
)

with open(ops_path, 'w') as f:
    f.write(code)

print("已修改 custom_ops.py: CUDA 插件编译失败时使用 PyTorch 实现")

# 修改 upfirdn2d 和 bias_act 中的 _init 逻辑
for mod_name in ['bias_act', 'upfirdn2d']:
    mod_path = f"/root/eg3d/eg3d/torch_utils/ops/{mod_name}.py"
    with open(mod_path, 'r') as f:
        mod_code = f.read()
    if '.bak' not in mod_path:
        with open(mod_path + '.bak', 'w') as f:
            f.write(mod_code)
    # 如果编译失败，确保 _init 返回 False
    mod_code = mod_code.replace(
        "if impl == 'cuda' and x.device.type == 'cuda' and _init():",
        "if impl == 'cuda' and x.device.type == 'cuda' and _init() and False:"
    )
    with open(mod_path, 'w') as f:
        f.write(mod_code)
    print(f"修改 {mod_name}.py: 强制使用 CPU/PyTorch 实现")

print("\n完成！现在 EG3D 将使用 PyTorch 实现（无需 CUDA 编译）")
