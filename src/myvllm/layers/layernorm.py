import torch  # 导入 PyTorch 主库
import time  # 导入时间模块用于基准测试计时

# LayerNorm：RMSNorm 归一化层（省去均值中心化，仅做均方根缩放）
class LayerNorm(torch.nn.Module):
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):  # gamma 为初始缩放权重，eps 为数值稳定项
        super().__init__()  # 调用父类 nn.Module 的初始化
        # Use nn.Parameter to make gamma learnable and loadable from checkpoints
        self.weight = torch.nn.Parameter(gamma.detach().clone())  # 用 nn.Parameter 包装 gamma，使其可训练并可从 checkpoint 加载
        self.eps = eps  # 保存数值稳定项，防止除零
    @property
    def gamma(self):  # gamma 属性：提供 weight 的别名以向后兼容
        """Backward compatibility: gamma alias for weight"""
        return self.weight  # 返回可学习权重

    @torch.compile  # 使用 torch.compile 编译以加速 RMSNorm 计算
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:  # 标准 RMSNorm 前向：x 形状 (..., hidden)
        # RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ

        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps  # 沿最后一维计算均方值并加 eps（保持维度以便广播）
        sqrt_variance = variance.sqrt()  # 开方得到均方根
        x_norm = (x / sqrt_variance * self.weight)  # 归一化后乘以可学习缩放权重 γ

        return x_norm  # 返回归一化后的张量

    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:  # 融合残差相加的 RMSNorm 前向
        x = x + residual  # 先将输入与残差相加（attention/MLP 输出叠加回主干）
        return self.rms_forward(x), x  # 返回（归一化结果, 相加后的值）；相加后的值作为下一层的 residual 传出

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:  # 前向入口：可选传入 residual
        if residual is not None:  # 若提供了残差张量
            return self.residual_rms_forward(x, residual)  # 走"残差相加 + RMSNorm"分支
        else:  # 未提供残差
            return self.rms_forward(x)  # 走标准 RMSNorm 分支

if __name__ == "__main__":  # 作为脚本直接运行时执行基准测试
    # Example usage
    x = torch.randn(8,4000,8000).cuda()  # 生成随机输入张量 (batch, seq, hidden) 并移到 GPU
    gamma = torch.full((8000,), 0.5, device="cuda", dtype=x.dtype)  # 创建全 0.5 的缩放权重 γ
    layer = LayerNorm(gamma=gamma).cuda()  # 实例化 LayerNorm 并移到 GPU
    residual = torch.full_like(x,fill_value=1)  # 创建与 x 同形状、全 1 的残差张量

    for _ in range(10): # Warm-up iterations  # 预热 10 次，触发编译并消除首跑开销
        _ = layer(x)  # 执行前向计算（预热）
    
    # Without residuals
    times = []  # 记录无残差路径的耗时
    for _ in range(100): # Timing iterations  # 正式计时 100 次迭代
        torch.cuda.synchronize()  # 同步 CUDA 流，确保计时准确
        start_time = time.time()  # 记录开始时间
        _ = layer(x)  # 执行无残差的前向计算
        torch.cuda.synchronize()  # 等待 GPU 计算完成
        end_time = time.time()  # 记录结束时间
        times.append(end_time - start_time)  # 累加本次耗时
    avg_time = sum(times) / len(times)  # 计算平均耗时
    print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")  # 打印无残差路径平均耗时

    # With residuals
    times.clear()  # 清空耗时列表，准备测量有残差路径
    for _ in range(100): # Timing iterations  # 正式计时 100 次迭代
        torch.cuda.synchronize()  # 同步 CUDA 流
        start_time = time.time()  # 记录开始时间
        _ = layer(x,residual)  # 执行带残差的前向计算
        torch.cuda.synchronize()  # 等待 GPU 计算完成
        end_time = time.time()  # 记录结束时间
        times.append(end_time - start_time)  # 累加本次耗时
    avg_time = sum(times) / len(times)  # 计算平均耗时
    print(f"[With residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")  # 打印有残差路径平均耗时
