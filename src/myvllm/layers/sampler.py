import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块基类


# SamplerLayer：从最终 logits 中按温度缩放后采样出下一个 token 的 id
class SamplerLayer(nn.Module):
    """
    A custom sampler layer that selects elements from the input tensor
    based on provided indices.
    """

    def __init__(self):  # 构造函数，无可学习参数
        super().__init__()  # 调用父类 nn.Module 的初始化

    @torch.compile  # 使用 torch.compile 编译前向计算以提升性能
    def forward(self, logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:  # logits 形状 (batch, vocab)，temperature 形状 (batch,)
        logits/= temperature.unsqueeze(-1)  # 对每个样本用其温度缩放 logits（temperature 越大分布越平滑）
        probs = torch.softmax(logits, dim=-1)  # 在词表维上做 softmax，得到概率分布
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)  # Gumbel 技巧：p/Exp(1) 等价于按概率分布采样，取 argmax 得到采样出的 token id
        return sample_tokens  # 返回采样得到的 token id，形状 (batch,)
