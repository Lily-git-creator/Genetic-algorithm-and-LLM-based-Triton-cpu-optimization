import torch
import triton
import triton.language as tl
import time
import math

# Triton RMSNorm 内核 - 针对CPU优化
@triton.jit
def rms_norm_kernel(
    # 输入/输出指针
    input_ptr, weight_ptr, output_ptr,
    # 张量维度信息
    n_elements,  # 每个样本的特征数 (标准化维度)
    stride_batch, stride_feature,  # 输入张量的内存步长
    # 参数
    eps: tl.constexpr,
    # 平铺参数
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm 内核: output = (input / sqrt(mean(input^2) + eps)) * weight
    
    计算过程:
    1. 计算输入张量的平方的均值 (RMS)
    2. 使用 RMS 对输入进行标准化
    3. 应用可学习的权重参数
    """
    # 当前程序处理的批次索引
    batch_idx = tl.program_id(axis=0)
    
    # 计算当前批次在内存中的起始偏移
    input_batch_start = batch_idx * stride_batch
    output_batch_start = batch_idx * stride_batch
    
    # ====== 步骤1: 计算 RMS (均方根) ======
    mean_square = tl.zeros((1,), dtype=tl.float32)
    
    # 分块计算平方和 (减少内存压力)
    for offset in range(0, n_elements, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_elements
        
        # 加载当前块的数据
        input_vals = tl.load(
            input_ptr + input_batch_start + col_idx * stride_feature,
            mask=mask,
            other=0.0,
        )
        
        # 累加平方值
        mean_square += tl.sum(input_vals * input_vals, axis=0)
    
    # 计算均值并加上 epsilon
    rms = tl.sqrt(mean_square / n_elements + eps)
    
    # ====== 步骤2: 应用标准化和权重 ======
    for offset in range(0, n_elements, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_elements
        
        # 重新加载输入数据
        input_vals = tl.load(
            input_ptr + input_batch_start + col_idx * stride_feature,
            mask=mask,
            other=0.0,
        )
        
        # 加载权重 (广播到整个批次)
        weight_vals = tl.load(
            weight_ptr + col_idx * stride_feature,
            mask=mask,
            other=1.0,  # 默认权重为1
        )
        
        # RMSNorm 计算: (input / rms) * weight
        normalized = (input_vals / rms) * weight_vals
        
        # 存储结果
        tl.store(
            output_ptr + output_batch_start + col_idx * stride_feature,
            normalized,
            mask=mask,
        )

def triton_rms_norm(input_tensor, weight, eps=1e-5):
    """
    Triton RMSNorm 实现
    
    参数:
        input_tensor: [batch_size, feature_dim] 输入张量
        weight: [feature_dim] 可学习的缩放权重
        eps: 防止除零的小常数
    
    返回:
        normalized: [batch_size, feature_dim] 标准化后的张量
    """
    # 验证输入维度
    assert input_tensor.dim() == 2, "输入必须是二维张量 [batch_size, feature_dim]"
    assert weight.dim() == 1, "权重必须是一维张量 [feature_dim]"
    assert input_tensor.shape[1] == weight.shape[0], "特征维度必须匹配"
    
    batch_size, feature_dim = input_tensor.shape
    
    # 分配输出张量
    output = torch.empty_like(input_tensor)
    
    # 配置内核参数
    BLOCK_SIZE = 128  # 可根据CPU缓存调整
    
    # 定义一维网格 (每个批次一个线程)
    grid = (batch_size,)
    
    # 启动内核
    rms_norm_kernel[grid](
        input_tensor, weight, output,
        feature_dim,
        input_tensor.stride(0), input_tensor.stride(1),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# ====== 参考实现 (PyTorch) ======
def pytorch_rms_norm(x, weight, eps=1e-5):
    """PyTorch 参考实现"""
    # 计算 RMS
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    # 标准化并应用权重
    return (x / rms) * weight

def test_rms_norm():
    """测试 RMSNorm 算子的正确性和性能"""
    print("🧪 开始 RMSNorm 算子测试")
    print("=" * 60)
    
    # 测试配置
    test_configs = [
        # (batch_size, feature_dim, 描述)
        (32, 512, "小批量 - 小型特征"),
        (128, 4096, "中等批量 - LLM典型隐藏层"),
        (512, 1024, "大批量 - 中型特征"),
        (16, 16384, "小批量 - 超宽特征"),
    ]
    
    # 精度容差
    rtol, atol = 1e-4, 1e-5
    
    for batch_size, feature_dim, desc in test_configs:
        print(f"\n📊 测试配置: {desc}")
        print(f"   批次大小: {batch_size}, 特征维度: {feature_dim}")
        
        # 生成随机测试数据
        torch.manual_seed(42)
        x = torch.randn(batch_size, feature_dim, device='cpu', dtype=torch.float32)
        weight = torch.randn(feature_dim, device='cpu', dtype=torch.float32)
        
        # 预热 (避免首次运行开销)
        if batch_size == test_configs[0][0]:
            print("   预热运行...")
            _ = triton_rms_norm(x[:2], weight)
            _ = pytorch_rms_norm(x[:2], weight)
        
        # ====== Triton 实现 ======
        torch.cuda.synchronize() if x.is_cuda else None
        start_time = time.perf_counter()
        triton_result = triton_rms_norm(x, weight)
        torch.cuda.synchronize() if x.is_cuda else None
        triton_time = time.perf_counter() - start_time
        
        # ====== PyTorch 参考实现 ======
        torch.cuda.synchronize() if x.is_cuda else None
        start_time = time.perf_counter()
        pytorch_result = pytorch_rms_norm(x, weight)
        torch.cuda.synchronize() if x.is_cuda else None
        pytorch_time = time.perf_counter() - start_time
        
        # ====== 正确性验证 ======
        # 计算最大绝对误差和相对误差
        abs_diff = torch.abs(triton_result - pytorch_result)
        max_abs_error = torch.max(abs_diff).item()
        
        # 相对误差 (避免除零)
        rel_diff = abs_diff / (torch.abs(pytorch_result) + 1e-8)
        max_rel_error = torch.max(rel_diff).item()
        
        is_correct = torch.allclose(
            triton_result, pytorch_result, 
            rtol=rtol, atol=atol
        )
        
        # ====== 性能分析 ======
        # 计算浮点运算次数 (近似)
        # 每个元素: 平方(1), 加法(1), 开方(1), 除法(1), 乘法(2) ≈ 6 FLOPs
        flops_per_element = 6
        total_flops = batch_size * feature_dim * flops_per_element
        
        triton_gflops = (total_flops / triton_time) / 1e9
        pytorch_gflops = (total_flops / pytorch_time) / 1e9
        speedup = pytorch_time / triton_time
        
        # ====== 打印结果 ======
        print(f"   ✅ 正确性: {'PASS' if is_correct else 'FAIL'}")
        if not is_correct:
            print(f"      最大绝对误差: {max_abs_error:.2e}")
            print(f"      最大相对误差: {max_rel_error:.2e}")
        
        print(f"   ⚡ 性能对比:")
        print(f"     - Triton:  {triton_time*1000:6.2f} ms, {triton_gflops:5.2f} GFLOP/s")
        print(f"     - PyTorch: {pytorch_time*1000:6.2f} ms, {pytorch_gflops:5.2f} GFLOP/s")
        print(f"     加速比: {speedup:.2f}x {'(Triton更快)' if speedup > 1.0 else '(PyTorch更快)'}")
        
        # ====== 额外验证: RMS 计算正确性 ======
        if batch_size <= 4:  # 只对小批次打印详细验证
            print(f"\n   🔍 详细验证 (前2个样本):")
            for i in range(min(2, batch_size)):
                # 计算 Triton 的 RMS
                rms_triton = torch.sqrt(torch.mean(triton_result[i]**2))
                # 计算 PyTorch 的 RMS
                rms_pytorch = torch.sqrt(torch.mean(pytorch_result[i]**2))
                print(f"      样本 {i}: Triton RMS={rms_triton:.4f}, PyTorch RMS={rms_pytorch:.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    
    # 返回最后一个测试的结果用于进一步分析
    return triton_result, pytorch_result

# ====== 主函数 ======
if __name__ == "__main__":
    print("🚀 Triton-CPU RMSNorm 算子测试套件")
    print("=" * 60)
    
    # 运行主测试
    triton_result, pytorch_result = test_rms_norm()

'''
python ./rmsnorm.py
🚀 Triton-CPU RMSNorm 算子测试套件
============================================================
🧪 开始 RMSNorm 算子测试
============================================================

📊 测试配置: 小批量 - 小型特征
   批次大小: 32, 特征维度: 512
   预热运行...
   ✅ 正确性: PASS
   ⚡ 性能对比:
     - Triton:    0.27 ms,  0.37 GFLOP/s
     - PyTorch:   0.13 ms,  0.75 GFLOP/s
     加速比: 0.49x (PyTorch更快)

📊 测试配置: 中等批量 - LLM典型隐藏层
   批次大小: 128, 特征维度: 4096
   ✅ 正确性: PASS
   ⚡ 性能对比:
     - Triton:    3.64 ms,  0.86 GFLOP/s
     - PyTorch:   2.97 ms,  1.06 GFLOP/s
     加速比: 0.81x (PyTorch更快)

📊 测试配置: 大批量 - 中型特征
   批次大小: 512, 特征维度: 1024
   ✅ 正确性: PASS
   ⚡ 性能对比:
     - Triton:    3.74 ms,  0.84 GFLOP/s
     - PyTorch:   0.92 ms,  3.41 GFLOP/s
     加速比: 0.25x (PyTorch更快)

📊 测试配置: 小批量 - 超宽特征
   批次大小: 16, 特征维度: 16384
   ✅ 正确性: PASS
   ⚡ 性能对比:
     - Triton:    1.87 ms,  0.84 GFLOP/s
     - PyTorch:   8.91 ms,  0.18 GFLOP/s
     加速比: 4.76x (Triton更快)

============================================================
测试完成！
'''