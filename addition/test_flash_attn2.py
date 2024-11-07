'''
You said:
import torch
print("FlashAttention test start...")
from flash_attn import flash_attn_func
print('imported')

# 创建随机输入
batch_size, seq_len, hidden_dim = 2, 1024, 64
q = torch.randn(batch_size, seq_len, hidden_dim).cuda().half()
k = v = q

# 运行 FlashAttention
output = flash_attn_func(q, k, v, causal=True)

print("FlashAttention output shape:", output.shape)
print("FlashAttention test passed!")
FlashAttention test start...
imported
Traceback (most recent call last):
  File "/nesa_data/remote_shome/zch/workspace/LongLLaVA/addition/test_flash_attn2.py", line 12, in <module>
    output = flash_attn_func(q, k, v, causal=True)
  File "/nesa_data/remote_shome/xianfeng/anaconda3/envs/LongLLaVa/lib/python3.10/site-packages/flash_attn-2.6.3-py3.10-linux-x86_64.egg/flash_attn/flash_attn_interface.py", line 1163, in flash_attn_func
    return FlashAttnFunc.apply(
  File "/nesa_data/remote_shome/xianfeng/anaconda3/envs/LongLLaVa/lib/python3.10/site-packages/torch/autograd/function.py", line 574, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/nesa_data/remote_shome/xianfeng/anaconda3/envs/LongLLaVa/lib/python3.10/site-packages/flash_attn-2.6.3-py3.10-linux-x86_64.egg/flash_attn/flash_attn_interface.py", line 805, in forward
    head_size_og = q.size(3)
IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)
为什么?

这个错误信息表明，`q.size(3)` 试图访问张量 `q` 的第4维，但是张量 `q` 实际上只有3个维度，因此抛出了 `IndexError`。错误出现在 `flash_attn_func` 的实现中，具体是在 `flash_attn_interface.py` 文件的第805行。

在 PyTorch 中，张量的维度是从0开始的，因此对于张量 `q` 形状为 `(batch_size, seq_len, hidden_dim)`，它只有3个维度。你使用的 `q.size(3)` 意图访问第四个维度（索引为3），但是 `q` 只有三个维度，因此会导致错误。

### 解决方法：
你需要检查 `flash_attn_func` 所期望的输入张量形状。一般来说，FlashAttention 期望输入的 `q`, `k`, `v` 张量形状为 `(batch_size, num_heads, seq_len, head_dim)`。如果你的 `q` 张量形状是 `(batch_size, seq_len, hidden_dim)`，你需要进行维度调整，使得张量包含 `num_heads` 和 `head_dim` 这两个维度。

你可以尝试像下面这样重新调整张量的形状：

```python
batch_size, seq_len, hidden_dim = 2, 1024, 64
num_heads = 8  # 假设你有8个头
head_dim = hidden_dim // num_heads  # 每个头的维度

# 调整 q, k, v 的形状
q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda().half()
k = v = q

# 运行 FlashAttention
output = flash_attn_func(q, k, v, causal=True)

print("FlashAttention output shape:", output.shape)
print("FlashAttention test passed!")
```

这样，你就为每个注意力头分配了合适的维度，并且确保 `q.size(3)` 能够正常工作，因为此时 `q` 的形状是 `(batch_size, num_heads, seq_len, head_dim)`，第四维是 `head_dim`。

如果你没有 `num_heads` 和 `head_dim` 的具体值，你需要根据模型的要求调整这些参数。


'''
import torch
print("FlashAttention test start...")
from flash_attn import flash_attn_func
print('imported')
batch_size, seq_len, hidden_dim = 2, 1024, 64
num_heads = 8  # 假设你有8个头
head_dim = hidden_dim // num_heads  # 每个头的维度

# 调整 q, k, v 的形状
q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda().half()
k = v = q

# 运行 FlashAttention
output = flash_attn_func(q, k, v, causal=True)

print("FlashAttention output shape:", output.shape)
print("FlashAttention test passed!")

# import torch
# import time
# from flash_attn import flash_attn_func

# def standard_attention(q, k, v):
#     scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
#     attn = torch.softmax(scores, dim=-1)
#     return torch.matmul(attn, v)

# # 设置大小
# batch_size, seq_len, hidden_dim = 2, 4096, 64
# q = torch.randn(batch_size, seq_len, hidden_dim).cuda().half()
# k = v = q

# # 测试标准注意力
# torch.cuda.synchronize()
# start = time.time()
# _ = standard_attention(q, k, v)
# torch.cuda.synchronize()
# standard_time = time.time() - start

# # 测试 FlashAttention
# torch.cuda.synchronize()
# start = time.time()
# _ = flash_attn_func(q, k, v)
# torch.cuda.synchronize()
# flash_time = time.time() - start

# print(f"Standard Attention Time: {standard_time:.4f}s")
# print(f"FlashAttention Time: {flash_time:.4f}s")
# print(f"Speedup: {standard_time / flash_time:.2f}x")


import torch
print("CUDA version:", torch.version.cuda)


import torch
from flash_attn import flash_attn_func

def measure_memory(func, *args):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = func(*args)
    return torch.cuda.max_memory_allocated() / 1e6  # Convert to MB

# 设置大小
batch_size, seq_len, hidden_dim = 2, 8192, 64
num_heads = 8  # 设定注意力头的数量
head_dim = hidden_dim // num_heads  # 计算每个头的维度

# 创建 q, k, v 张量，形状为 (batch_size, num_heads, seq_len, head_dim)
q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda().half()
k = v = q

# 运行 FlashAttention，测量内存使用情况
flash_memory = measure_memory(flash_attn_func, q, k, v)
print(f"FlashAttention Memory Usage: {flash_memory:.2f} MB")


if q.is_cuda:
    print("q is on GPU")
else:
    print("q is on CPU")
