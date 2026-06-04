"""精确禁用 EG3D CUDA 插件"""
import re

for fname in ['bias_act', 'upfirdn2d']:
    path = f'/root/eg3d/eg3d/torch_utils/ops/{fname}.py'
    with open(path) as f:
        code = f.read()

    with open(path + '.bak2', 'w') as f:
        f.write(code)

    code = re.sub(r'def _init\(\):.*?return True',
                   'def _init():\n        return False',
                   code, flags=re.DOTALL)
    code = re.sub(r'def _init\(\):.*?return any',
                   'def _init():\n        return False',
                   code, flags=re.DOTALL)

    with open(path, 'w') as f:
        f.write(code)
    print(f'Patched {fname}.py ✓')

# custom_ops.py: 跳过编译
path = '/root/eg3d/eg3d/torch_utils/custom_ops.py'
with open(path) as f:
    code = f.read()
with open(path + '.bak2', 'w') as f:
    f.write(code)

code = code.replace('def get_plugin(',
                    'def get_plugin(*a,**kw):\n    return None\ndef _get_plugin_impl(')
code = code.replace('warnings.warn(',
                    'warnings.warn(\'[EG3D] CUDA disabled, using CPU\' ')
with open(path, 'w') as f:
    f.write(code)
print('Patched custom_ops.py ✓')
print('\n完成！EG3D 将使用纯 PyTorch 实现')
