import sys, os  # 导入系统模块 sys（用于修改 Python 路径）和操作系统模块 os（用于处理路径、环境变量）
from pathlib import Path  # 导入 Path 类，用于以面向对象方式处理文件路径
import torch.distributed as dist  # 导入 PyTorch 分布式通信模块（本脚本未直接使用，保持与分布式引擎的兼容）

from transformers import AutoTokenizer, AutoModelForCausalLM  # 从 transformers 导入自动分词器与因果语言模型类（此处主要用于加载分词器）
import torch  # 导入 PyTorch 主库

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))  # 将脚本所在目录下的 src 目录插入 sys.path 首位，使 myvllm 包可以被导入

from myvllm.engine.llm_engine import LLMEngine as LLM  # 导入自实现的 LLM 推理引擎，并简写为 LLM（对标 vLLM 的 LLM 入口类）
from myvllm.sampling_parameters import SamplingParams  # 导入采样参数类，用于控制生成时的温度、最大生成长度等

config = {  # 定义引擎与模型的全局配置字典（针对 Llama-3.2-1B-Instruct）
    'max_num_sequences': 16,  # 调度器同时处理的最大序列数
    'max_num_batched_tokens': 1024,  # 每次批量调度中允许的最大 token 数
    'max_cached_blocks': 1024,  # KV 缓存可使用的最大块数量

    'max_num_batch_tokens': 4096,  # 单次前向计算允许批量处理的最大 token 数
    'max_model_length': 128,  # 模型支持的最大总长度（提示词 + 生成内容）
    'gpu_memory_utilization': 0.9,  # GPU 显存的目标利用率（0.9 表示预留 10% 余量）

    'block_size': 256,  # KV 缓存中每个块包含的 token 数（块大小）
    'world_size': 1,  # 分布式并行进程数（1 表示单卡运行）

    'enforce_eager': True,  # 强制以 eager 模式运行（不使用 CUDA Graph）

    # model params  # 以下为模型结构参数
    'model_name_or_path': 'meta-llama/Llama-3.2-1B-Instruct',  # 模型在 HuggingFace 上的名称或本地路径
    'vocab_size': 128256,  # 词表大小
    'hidden_size': 2048,  # 隐藏层维度
    'head_dim': 64,  # 每个注意力头的维度
    'num_qo_heads': 32,  # 注意力的查询/输出头数量
    'num_kv_heads': 8,  # 注意力的键值头数量（GQA：少于查询头数量）
    'has_attn_bias': False,  # 注意力投影是否使用偏置项
    'rms_norm_epsilon': 1e-5,  # RMSNorm 归一化的数值稳定常数 epsilon
    'rope_base': 500000,  # RoPE 旋转位置编码的基数（rope_theta）
    'max_position_embeddings': 32768, # 旋转位置编码允许的最大位置索引，应不小于最大模型长度 # should be >= max_model_length, max position index allowed in rotary embedding
    'intermediate_size': 8192,  # FFN 前馈层的中间维度
    'ffn_bias': False,  # FFN 层是否使用偏置项
    'num_layers': 16,  # Transformer 层数
    'tie_word_embeddings': True,  # 输入词嵌入与输出 lm_head 共享权重
    'eos': 128009,  # 结束符（EOS）的 token id，用于判断生成结束  # should match tokenizer.eos_token_id. But strangely, HF says eos is [128001, 128008, 128009].
}

def main():
    # 主函数：加载分词器与推理引擎，构造提示词并执行批量文本生成
    model_name = config.get('model_name_or_path')  # 从配置中读取模型名称或路径
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 加载与模型配套的分词器
    llm = LLM(config=config)  # 用配置字典实例化自实现的 LLM 推理引擎
    
    # max_tokens is the max number of generated tokens
    # max_model_length is the max total length including prompt
    # both should be set in SamplingParams and help to determine when to stop generation
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256, max_model_length=128)  # 设置采样参数：温度 0.6、最多生成 256 个 token、总长度上限 128
    prompts = [  # 定义原始用户提示词列表
        "introduce yourself",# * 15,
        "list all prime numbers within 100",# * 15,
        "give me your opinion on the impact of artificial intelligence on society",# * 15,
    ] #* 30
    prompts = [  # 用分词器的对话模板把每条原始提示包装成模型可读的聊天格式（不进行 tokenize，只得到文本）
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],  # 构造单轮 user 消息
            tokenize=False,  # 只返回模板文本，不做分词
            add_generation_prompt=True,  # 在末尾追加 assistant 开始标记，提示模型开始生成回答
        )
        for prompt in prompts  # 遍历原始提示词列表
    ]
    outputs = llm.generate(prompts, sampling_params)  # 调用推理引擎对全部提示词批量生成结果

    # outputs is a dict with 'text' and 'token_ids' keys
    generated_texts = outputs['text']  # 从结果字典中取出生成的文本列表

    for prompt, output in zip(prompts, generated_texts):  # 逐条遍历（提示词, 生成文本）配对
        print("\n")  # 打印空行分隔各条结果
        print(f"Prompt: {prompt}")  # 打印模板化后的提示词
        print(f"Completion: {output}")  # 打印模型生成的补全内容


if __name__ == "__main__":  # 仅当脚本被直接运行（而非被导入）时才执行主函数
    main()  # 调用主函数启动推理流程
