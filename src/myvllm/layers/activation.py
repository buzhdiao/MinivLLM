import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块基类
import torch.nn.functional as F  # 导入函数式接口（含 silu 激活函数）
import time  # 导入时间模块用于基准测试计时

# SiluAndMul：SwiGLU 风格激活层，对输入后半部分做 SiLU 激活后与前半部分逐元素相乘
class SiluAndMul(nn.Module):
    """
    A custom activation layer that applies the SiLU (Sigmoid Linear Unit) activation
    function followed by element-wise multiplication with the input tensor.
    """

    def __init__(self):  # 构造函数，无可学习参数
        super().__init__()  # 调用父类 nn.Module 的初始化

    @torch.compile  # 使用 torch.compile 将 forward 编译为融合 kernel 以提升性能
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向计算：x 形状 (..., 2*d)，输出形状 (..., d)
        x, y = x.chunk(2, -1)  # 沿最后一维将输入均分为两半：x 为门控分支，y 为数值分支
        return F.silu(x) * y  # 对 x 施加 SiLU(x)=x*sigmoid(x) 后与 y 逐元素相乘（SwiGLU 激活）

if __name__ == "__main__":  # 作为脚本直接运行时执行基准测试
    # Example usage
    layer = SiluAndMul().cuda()  # 实例化激活层并移动到 GPU
    input_tensor = torch.randn(8, 4000, 8000).cuda()  # Example input tensor with shape (8, 4000, 8000)  # 生成随机输入张量并移到 GPU

    for _ in range(10):  # Warm-up iterations  # 预热 10 次，触发编译并消除首跑开销
        _ = layer(input_tensor)  # 执行前向计算（预热，不计入时间）

    times = []  # 记录每次运行的耗时
    for _ in range(100):  # Timing iterations  # 正式计时 100 次迭代
        torch.cuda.synchronize()  # 同步 CUDA 流，确保计时准确
        start_time = time.time()  # 记录开始时间
        output_tensor = layer(input_tensor)  # 执行一次前向计算
        torch.cuda.synchronize()  # 等待 GPU 计算完成后再读时间
        end_time = time.time()  # 记录结束时间
        times.append(end_time - start_time)  # 累加本次耗时
    avg_time = sum(times) / len(times)  # 计算 100 次的平均耗时
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")  # 打印平均推理耗时（毫秒）
