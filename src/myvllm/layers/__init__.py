from .activation import SiluAndMul  # 导入 SiLU 激活 + 逐元素相乘的自定义激活层
from .attention import Attention  # 导入支持 PagedAttention/SDPA 的注意力层
from .embedding_head import ParallelLMHead, VocabParallelEmbedding  # 导入词表并行嵌入层与语言模型输出头
from .layernorm import LayerNorm  # 导入 RMSNorm 归一化层
from .linear import *  # 导入线性层模块的全部公开成员（ColumnParallel/RowParallel 等）
from .rotary_embedding import RotaryEmbedding  # 导入 RoPE 旋转位置编码层
from .sampler import SamplerLayer  # 导入采样层（logits -> token id）
