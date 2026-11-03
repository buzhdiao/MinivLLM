from dataclasses import dataclass  # 装饰器：自动生成 __init__/__repr__ 等样板代码
import torch  # PyTorch：上下文中的张量字段类型


# Context：一次调度步（一次 forward batch）的推理上下文
# 职责：携带 prefill/decode 标志、varlen 的 cu_seqlens、slot_mapping（KV cache 物理槽位）、
#       context_lens（各序列当前长度）与 block_tables（分页 KV cache 的逻辑块→物理块映射）
@dataclass
class Context:
    is_prefill: bool = False  # 当前批次是否为 prefill 阶段（False 表示 decode 阶段）
    cu_seqlens_q: torch.Tensor | None = None  # 批量 prefill 时各序列 Q 的累积长度边界（varlen 布局）
    cu_seqlens_k: torch.Tensor | None = None  # 批量 prefill 时各序列 K 的累积长度边界
    max_seqlen_q: int = 0  # 批内 Q 侧最长序列长度（varlen kernel 的优化提示）
    max_seqlen_k: int = 0  # 批内 K 侧最长序列长度
    slot_mapping: torch.Tensor | None = None  # 每个 token 在分页 KV cache 中的物理槽位编号
    context_lens: torch.Tensor | None = None  # 每条序列当前的上下文长度（decode 时兼作位置索引）
    block_tables: torch.Tensor | None = None  # 每条序列的逻辑块号 → 物理块号映射表

_context = Context()  # 模块级全局单例：当前调度步的上下文（默认空值）

# get_context：读取当前全局上下文（模型 forward 期间由此获取批次信息）
def get_context() -> Context:
    return _context  # 返回全局 Context 单例

# reset_context：把全局上下文重置为空（用于清理状态/测试）
def reset_context():
    global _context  # 声明修改模块级变量
    _context = Context()  # 用默认空 Context 覆盖

# set_context：由 scheduler 在每次 forward 前设置本批次的上下文信息
def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _context  # 声明修改模块级变量
    _context = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)  # 用本批次字段构造新的 Context 并替换全局单例
