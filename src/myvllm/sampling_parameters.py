from dataclasses import dataclass  # 装饰器：自动生成 __init__ 等样板代码


# SamplingParams：单条请求的采样参数配置
# 职责：控制解码阶段的采样行为（温度、最大生成长度、是否忽略 EOS 等）
@dataclass
class SamplingParams:
    temperature: float = 1.0  # 采样温度：越低分布越尖锐（越确定），越高越随机；不支持 0（贪婪解码）
    max_tokens: int = 64  # Maximum number of tokens to generate (completion tokens only)  # 最大生成 token 数（仅计补全部分，不含 prompt）
    ignore_eos: bool = False  # 是否忽略 EOS token（True 时即使生成结束符也继续解码）
    max_model_length: int | None = None  # Maximum total sequence length (prompt + completion)  # 序列总长上限（prompt + 生成部分），None 表示不限制

    # __post_init__：dataclass 构造后的校验钩子
    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"  # 校验温度为正：温度趋近 0 的贪婪解码在本引擎中不允许
