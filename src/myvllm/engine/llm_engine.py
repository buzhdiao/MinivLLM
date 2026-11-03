import atexit  # 导入 atexit，注册进程退出时的清理回调
import torch.distributed as dist  # 导入分布式模块，用于多卡（tensor parallel）初始化判断
import time  # 导入 time，用于统计每步吞吐
import torch.multiprocessing as mp  # 导入 torch 的多进程封装，spawn 子进程运行 worker

from myvllm.engine.sequence import Sequence  # 导入序列类，add_prompt 时把文本转为 Sequence
from myvllm.engine.scheduler import Scheduler  # 导入调度器，负责 continuous batching 的批次调度
from myvllm.engine.model_runner import ModelRunner  # 导入模型执行器，封装模型前向与 CUDA graph
from myvllm.sampling_parameters import SamplingParams  # 导入采样参数类
from transformers import AutoTokenizer  # 导入 HF 分词器，负责 prompt 编码与输出解码


# worker_process：子进程入口。每个非 0 rank 启动一个 ModelRunner 并进入事件循环，
# 等待主进程（rank 0）经共享内存下发的方法调用
def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code  # 在其他任何代码之前先把 stdout 重配好，保证日志输出
    import sys  # 导入 sys 以拿到 stdout/stderr 的文件描述符
    import os  # 导入 os 以按 fd 重建文件对象
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering  # stdout 设为行缓冲，子进程日志实时可见
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)  # stderr 同样设为行缓冲

    model_runner = ModelRunner(config, rank, event)  # 初始化本 rank 的模型执行器（加载模型、分配 KV cache、捕获 CUDA graph）
    model_runner.loop()  # 进入事件循环：阻塞等待 rank 0 经共享内存下发的调用并执行


# LLMEngine：对外的高层引擎入口。组合 ModelRunner（执行）、Scheduler（调度）与
# Tokenizer（编解码），对上提供 add_prompt / generate 接口
class LLMEngine:
    # 初始化：启动 worker 子进程、创建 rank 0 的 ModelRunner、分词器与调度器
    def __init__(self, config: dict):
        self.config = config  # 保存全局配置字典
        world_size = config.get("world_size", 1)  # 卡数（rank 总数），默认单卡
        ctx = mp.get_context("spawn")  # 使用 spawn 启动方式（CUDA 要求子进程全新初始化）
        self.processes = []  # 保存 worker 子进程列表
        self.events = []  # 保存与各 worker 通信的进程事件（同步共享内存读写）
        for i in range(1, world_size):  # rank 0 在本进程内运行，只为 rank 1..N-1 创建子进程
            event = ctx.Event()  # 每个 worker 一个事件，用于共享内存的就绪通知
            process = ctx.Process(target=worker_process, args=(config, i, event))  # 创建 worker 进程
            self.events.append(event)  # 登记事件
            self.processes.append(process)  # 登记进程
            process.start()  # 启动子进程
        # start the engine only on the master thread with rank = 0  # rank 0（主进程）内直接构建 ModelRunner；event 列表用于向各 worker 广播
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))  # 加载与模型配套的分词器
        
        # scheduler needs to init after model_runner: when world_size > 1,
        # ModelRunner.__init__ calls dist.init_process_group() which is a
        # collective barrier — rank-0 blocks until all worker ranks have joined.
        # The scheduler should only be created after that rendezvous completes.
        # When world_size == 1 there is no barrier and no real dependency.  # 调度器必须在 ModelRunner 之后创建：多卡时 init_process_group 是集合通信屏障，rank 0 会阻塞直到所有 worker 加入，调度器要等这次会合完成后再建（单卡无屏障，无真实依赖）
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),  # 每步批次最大序列数
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),  # 每步 token budget
            max_cached_blocks=config.get("max_cached_blocks", 1024),  # KV cache 可用块数（多卡时已被 ModelRunner 同步为各 rank 最小值）
            block_size=config.get("block_size", 256),  # 每块 token 数
            eos=config.get("eos", 50256)  # EOS token id，停止条件之一
        )

        atexit.register(self.exit)  # 注册退出钩子，进程结束时清理子进程与共享资源


    # 退出清理：通知各 worker 退出、等待子进程结束
    def exit(self):
        self.model_runner.call("exit")  # 经 call 通道广播 exit：rank 0 自身清理，worker 收到后释放共享内存并退出
        del self.model_runner  # 释放 rank 0 的 ModelRunner（删除 CUDA graph 等资源）
        for process in self.processes:  # 逐个等待 worker 子进程退出
            process.join()

    # call scheduler to schedule the next batch  # 引擎单步：让调度器组出下一步批次
    # return scheduled sequences and whether it is for prefilling  # 
    # call model_runner.run() to run the model  # 执行一次模型前向（prefill 或 decode）
    # call postprocessor to process the outputs and update sequences and update block manager  # 采样结果写回序列并按停止条件收尾
    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        scheduled_sequences, is_prefill = self.scheduler.schedule()  # 调度：拿到本步批次与批次类型
        num_processed_tokens = 0  # 本步处理的 token 数（用于吞吐统计）
        if not scheduled_sequences:  # 空批次（例如刚发生 preemption）：直接返回
            return [], num_processed_tokens, is_prefill
        # run the model  # 执行模型前向；多卡时 rank 0 经共享内存把调用转发给各 worker
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        # Move outputs to CPU and convert them to a list  # 采样得到的 token id 张量搬到 CPU 并转为列表
        if outputs is not None:
            outputs = outputs.cpu().tolist()
        # postprocess the outputs  # 后处理：把 token 写回序列、判断停止条件并释放已完成序列的块
        self.scheduler.postprocess(scheduled_sequences, outputs)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]  # 只收集本步已完成序列的 (seq_id, 补全 token 列表)
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)  # prefill 按各序列总 token 数计，decode 每序列计 1 个

        return outputs, num_processed_tokens, is_prefill


    # add prompt string to the waiting queue by first transforming it to Sequence object  # 对外接口：把 prompt 文本编码为 Sequence 后加入调度队列
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), block_size=self.config['block_size'],sampling_params=sampling_params))  # 分词 -> 构建 Sequence（携带采样参数与块大小）-> 入队

    # given a list of prompts  # 对外主接口：批量生成
    # add_prompt for each prompt  # 逐条入队
    # call step until all sequences are finished  # 循环执行引擎单步直到所有序列完成
    # return the generated texts  # 返回各 prompt 的生成文本与 token id
    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        for prompt in prompts:  # 逐条把 prompt 加入 waiting 队列
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}  # seq_id -> 生成的 token 列表
        while not self.scheduler.is_finished():  # 直到 waiting 与 running 均为空
            start_t = time.time()  # 记录单步开始时间
            outputs, num_processed_tokens, is_prefill = self.step()  # 执行一步调度+前向+采样+后处理
            end_t = time.time()  # 记录单步结束时间
            running_time = end_t - start_t + 1e-10  # 步耗时（加小量防除零）
            if is_prefill:  # prefill 步：打印 prefill 吞吐
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:  # decode 步：打印 decode 吞吐
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})  # 合并本步完成序列的结果

        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]  # 按 seq_id 排序，保证输出顺序与输入 prompts 一致
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}  # 解码 token id 为文本，同时保留原始 id
        return output
