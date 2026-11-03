import torch.nn as nn  # 导入神经网络模块基类
import torch  # 导入 PyTorch 主库

# apply_rotary_pos_emb：对 Q/K 的最后 rotary_embedding 维施加 RoPE 旋转（将 (x1,x2) 视为复数对旋转 cos+isin）
def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:  # x 为 Q 或 K，cos/sin 为按位置查表得到的三角函数值
    # Handle both 3D varlen (total_tokens, num_heads, head_dim) and 4D batched (B, seq_len, num_heads, head_dim)
    if x.dim() == 3:  # 3D 变长（varlen）模式：整个 batch 的 token 拼成一维
        # Varlen mode: (total_tokens, num_heads, head_dim)
        total_tokens, num_heads, head_dim = x.shape  # 解包形状：总 token 数、注意力头数、每头维度
        # cos, sin shape: (total_tokens, head_dim/2)
        # Expand to (total_tokens, 1, head_dim/2) for broadcasting
        cos = cos.unsqueeze(1)  # 在头维度插入大小为 1 的轴，便于对 num_heads 广播
        sin = sin.unsqueeze(1)  # 同上，对 sin 做广播扩展

        # Split x into two halves along the head dimension
        x1, x2 = x.chunk(2, dim=-1)  # 沿最后一维切成两半，构成待旋转的复数对 (x1, x2)

        # Apply rotary embedding
        # x1, x2 shape: (total_tokens, num_heads, head_dim/2)
        # cos, sin shape: (total_tokens, 1, head_dim/2)
        out1 = x1 * cos - x2 * sin  # 旋转公式的实部：x1·cosθ - x2·sinθ
        out2 = x1 * sin + x2 * cos  # 旋转公式的虚部：x1·sinθ + x2·cosθ

        return torch.cat([out1, out2], dim=-1)  # 拼回完整 head_dim 并返回
    else:  # 4D 批处理模式：每个 batch 有独立的 seq_len
        # Batched mode: (B, seq_len, num_heads, head_dim)
        B = x.size(0)  # 批大小
        seq_len = x.size(1)  # 序列长度
        num_heads = x.size(2)  # 注意力头数
        head_dim = x.size(-1)  # 每头维度

        # Expand cos and sin to match the batch and head dimensions
        # cos, sin shape: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(2)  # 依次插入 batch 轴和头轴为 1，便于广播
        sin = sin.unsqueeze(0).unsqueeze(2)  # 同上，对 sin 做广播扩展

        # Split x into two halves along the head dimension
        x1, x2 = x.chunk(2, dim=-1)  # 沿最后一维切成两半，构成复数对 (x1, x2)

        # Apply rotary embedding with proper broadcasting
        # x1, x2 shape: (B, seq_len, num_heads, head_dim/2)
        # cos, sin shape: (1, seq_len, 1, head_dim/2)
        out1 = x1 * cos - x2 * sin  # 旋转实部：x1·cosθ - x2·sinθ
        out2 = x1 * sin + x2 * cos  # 旋转虚部：x1·sinθ + x2·cosθ

        return torch.cat([out1, out2], dim=-1)  # 拼回完整 head_dim 并返回


# RotaryEmbedding：RoPE 旋转位置编码模块，预计算 cos/sin 查找表并按位置索引对 Q/K 施加旋转
class RotaryEmbedding(nn.Module):
    def __init__(  # 构造函数
        self, 
        base:int,  # RoPE 基数 θ（如 10000），决定频率分布
        rotary_embedding: int,  # 施加旋转的维度数（通常是 head_dim，可为部分维度）
        max_position: int = 2048,  # 需要预计算的最大位置数
        is_llama3: bool = False,  # 是否使用 Llama3 的 RoPE 缩放策略
        # the following params are only used in llama3.2
        llama3_rope_factor: float = 32.0,  # Llama3 长上下文外推的缩放因子
        llama3_rope_high_freq_factor: float = 4.0,  # 高频因子
        llama3_rope_low_freq_factor: float = 1.0,  # 低频因子
        llama3_rope_original_max_position_embeddings: int = 8192,  # 原始预训练最大位置数
    ):
        super().__init__()  # 调用父类 nn.Module 的初始化
        self.base = base  # 保存 RoPE 基数
        # how many dimensions to apply rotary embedding
        self.rotary_embedding = rotary_embedding  # 保存施加旋转的维度数
        # max position that the long context can reach
        self.max_position = max_position  # 保存最大位置数
        self.inv_freq = 1/(base ** (torch.arange(0, self.rotary_embedding, 2)/self.rotary_embedding))  # 计算 inverse frequency：θ_i = 1/base^(2i/d)，低维高频、高维低频

        if is_llama3:  # Llama3.2 的频率平滑缩放：衰减低频分量以支持长上下文外推
            # specifically for llama3.2
            import math  # 仅此处需要 math（用到 2π）
            inv_freq = self.inv_freq  # 取出原始 inverse frequency
            # no smooth if low_freq_factor == high_freq_factor
            wave_len = 2 * math.pi / inv_freq  # 把频率换算为波长 λ = 2π/f
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:  # 两个因子相等时使用硬阈值（无平滑过渡）
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings / llama3_rope_high_freq_factor,  # 波长小于阈值（高频）保持不变
                    inv_freq,
                    inv_freq / llama3_rope_factor,  # 低频分量按 factor 缩放以拉伸波长
                )
            else:  # 因子不等时使用平滑插值
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor  # 高低频因子之差
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta  # 按波长计算 [0,1] 的平滑插值系数
                smooth = torch.clamp(smooth, 0, 1)  # 将平滑系数截断到 [0,1]
                factor = (1 - smooth) / llama3_rope_factor + smooth  # 对缩放因子做 (1-s)/factor + s 的平滑混合
                inv_freq = factor * inv_freq  # 应用平滑后的缩放因子
            self.inv_freq = inv_freq  # 用（可能缩放过的）频率覆盖原始值

        positions = torch.arange(self.max_position).float()  # 生成所有位置索引 [0, max_position)
        # (max_position, rotary_embedding/2)
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)  # 外积：每个位置与每个频率相乘，得到相位角 θ·pos

        cos = torch.cos(freqs)  # 预计算各位置各频率的余弦值
        sin = torch.sin(freqs)  # 预计算各位置各频率的正弦值

        # (max_position, rotary_embedding)
        cos_sin_cache = torch.cat([cos, sin], dim=-1)  # 将 cos 与 sin 拼接为查找表，每行 [cos|sin]
        self.register_buffer("cos_sin_cache", cos_sin_cache)  # 注册为 buffer：随模型保存/加载、随设备迁移，但不参与训练

    @torch.compile  # 使用 torch.compile 编译前向计算
    # tell the position index of the token
    # apply rotary embedding to query and key
    def forward(self, positions, query, key):  # positions 为每个 token 的绝对位置索引，query/key 为待旋转的 Q/K
        cos_sin = self.cos_sin_cache[positions]  # (seq_len, rotary_embedding)  # 按位置索引从预计算表中查 cos/sin
        cos, sin = cos_sin.chunk(2, dim=-1)  # 将查找表最后一维切开，得到 cos 与 sin 两部分
        return (
            apply_rotary_pos_emb(query, cos, sin),  # 对 query 施加 RoPE 旋转
            apply_rotary_pos_emb(key, cos, sin)  # 对 key 施加 RoPE 旋转
        )


if __name__ == "__main__":  # 作为脚本直接运行时演示 inv_freq 与频率表的计算过程
    base = 5  # 演示用的 RoPE 基数
    # how many dimensions to apply rotary embedding
    rotary_dim = 16  # 演示用的旋转维度
    # maximum position that the long context can reach
    max_position = 100  # 演示用的最大位置数
    print(torch.arange(0, rotary_dim, 2))  # 打印偶数维索引序列 [0,2,4,...]
    print(base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))  # 打印 base^(2i/d)，即频率的分母
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))  # 计算 inverse frequency
    print(inv_freq)  # 打印频率值

    t = torch.arange(max_position).float()  # 生成位置索引序列

    freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 位置与频率的外积，得到相位角表

    print(freqs.size())  # 打印频率表形状 (max_position, rotary_dim/2)

    print(freqs[2])  # 打印位置 2 处各频率的相位角
