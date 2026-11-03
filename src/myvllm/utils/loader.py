import torch  # PyTorch：张量拼接（qkv/gate_up 合并）等操作
from torch import nn  # nn.Module：权重加载的目标模型类型
import os  # 文件与目录操作（本地路径检查、列目录）
from safetensors import safe_open  # safetensors 读取接口：安全、零拷贝地加载权重分片文件
from transformers import AutoConfig  # HuggingFace 配置加载（当前仅导入备用）
import re  # 正则表达式：从权重名中解析层号（layers.N）


# default_weight_loader：默认权重加载函数
# 职责：校验形状后把 HF 权重逐元素拷贝进模型参数
def default_weight_loader(param, weight):
    """Default weight loader that copies weight data to parameter."""
    if param.shape != weight.shape:  # 形状不一致时直接报错（防止静默错载）
        raise ValueError(f"Shape mismatch: param {param.shape} vs weight {weight.shape}")
    param.data.copy_(weight)  # 将权重数据拷贝进参数（不替换参数对象，保持叶子节点身份）


# load_weights_from_checkpoint：从 HF checkpoint 加载权重到自定义模型
# 职责：解析本地路径或从 HF Hub 下载 → 读取所有 safetensors 分片 →
#       按名称映射加载；其中 q/k/v 三路权重拼接合并进 qkv_projection，
#       gate/up 两路权重拼接合并进 gate_up（对应合并后的列并行大矩阵）
def load_weights_from_checkpoint(model: nn.Module, model_name_or_path: str):
    """
    Load weights from a Hugging Face model checkpoint into the custom model.
    Handles QKV and gate_up weight merging for optimized layers.

    Args:
        model: The target model to load weights into
        model_name_or_path: Path to local checkpoint or Hugging Face model name
    """
    from huggingface_hub import snapshot_download  # 函数内导入：按需下载 HF 仓库快照

    # Try to resolve the path - could be local or from HF cache  # 先尝试解析路径——可能是本地路径或 HF 缓存
    checkpoint_path = None  # 解析出的 checkpoint 目录，初始为 None

    # First, try local paths  # 第一步：尝试本地路径
    if model_name_or_path.startswith('~'):  # 以 ~ 开头的用户目录路径
        checkpoint_path = os.path.expanduser(model_name_or_path)  # 展开为绝对路径（如 /home/user/...）
    elif os.path.isdir(model_name_or_path):  # 已存在的本地目录
        checkpoint_path = model_name_or_path  # 直接作为 checkpoint 路径

    # If not a local path, try to download from HuggingFace  # 若不是本地路径，则尝试从 HuggingFace 下载
    if checkpoint_path is None or not os.path.exists(checkpoint_path):  # 本地解析失败时
        try:
            checkpoint_path = snapshot_download(  # 下载仓库快照到 HF 缓存目录，返回本地路径
                repo_id=model_name_or_path,  # HF 模型仓库名
                allow_patterns=["*.safetensors", "*.json"],  # 只下载权重分片与配置 JSON
                ignore_patterns=["*.msgpack", "*.h5", "*.bin"]  # 跳过非 safetensors 的权重格式
            )
        except Exception as e:  # 下载失败（仓库名错误、网络问题等）
            raise ValueError(  # 抛出带上下文信息的错误
                f"Could not find or download model '{model_name_or_path}'. "
                f"Error: {e}\n"
                f"Please ensure the model name is correct or provide a valid local path."
            )

    if not os.path.exists(checkpoint_path):  # 最终路径仍不存在则报错
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    # Load all safetensors files in the checkpoint directory  # 收集 checkpoint 目录下所有 safetensors 分片文件
    safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]  # 大模型权重通常被切分为多个分片（shard）

    if not safetensor_files:  # 目录中没有任何 safetensors 文件则报错
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    # Collect all weights from HF model  # 把所有分片中的权重汇总到一个字典
    hf_weights = {}  # 键为 HF 权重名，值为 CPU 上的张量
    for file in sorted(safetensor_files):  # 按文件名排序，保证分片读取顺序确定
        file_path = os.path.join(checkpoint_path, file)  # 拼出分片完整路径
        with safe_open(file_path, framework='pt', device='cpu') as f:  # 打开单个 safetensors 分片（PyTorch 框架、CPU 设备）
            for weight_name in f.keys():  # 遍历该分片内的所有权重名
                hf_weights[weight_name] = f.get_tensor(weight_name)  # 惰性读取并缓存该张量

    # Now map and load weights into custom model  # 开始按名称映射并加载进自定义模型
    loaded_params = set()  # 已成功加载（或已合并）的参数名集合
    skipped_params = []  # 被跳过的 (参数名, 原因) 列表

    # Process each HF weight  # 逐条处理每个 HF 权重
    for hf_name, hf_weight in hf_weights.items():
        try:
            # 1. Handle QKV merge (q_proj + k_proj + v_proj → qkv_projection)  # 情形1：把 q/k/v 三路权重拼接合并进 qkv_projection
            if '.self_attn.q_proj.weight' in hf_name:  # 以 q_proj 权重为入口（k/v 随后一并处理）
                layer_match = re.search(r'layers\.(\d+)', hf_name)  # 从权重名解析层号 N
                if layer_match:  # 能解析出层号才继续
                    layer_idx = layer_match.group(1)  # 取层号字符串
                    k_name = hf_name.replace('q_proj', 'k_proj')  # 推导同层 K 权重名
                    v_name = hf_name.replace('q_proj', 'v_proj')  # 推导同层 V 权重名

                    if k_name in hf_weights and v_name in hf_weights:  # K/V 权重都存在时才合并
                        q_weight = hf_weight  # 当前权重即 Q
                        k_weight = hf_weights[k_name]  # 取出 K
                        v_weight = hf_weights[v_name]  # 取出 V

                        # Concatenate q, k, v along output dimension  # 沿输出维（dim=0）拼接 Q/K/V
                        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)  # 得到与融合 QKV 大矩阵一致的布局

                        custom_name = f"model.layers.{layer_idx}.self_attn.qkv_projection.weight"  # 目标合并参数名
                        try:
                            param = model.get_parameter(custom_name)  # 在模型中查找该合并参数
                            param.data.copy_(qkv_weight)  # 拷贝拼接后的权重
                            loaded_params.add(custom_name)  # 记录合并参数已加载
                            loaded_params.add(hf_name)  # 记录 Q 来源已消费
                            loaded_params.add(k_name)  # 记录 K 来源已消费
                            loaded_params.add(v_name)  # 记录 V 来源已消费
                        except AttributeError:  # 模型中不存在该参数（结构不匹配）
                            skipped_params.append((custom_name, "Parameter not found"))

            # 2. Handle gate_up merge (gate_proj + up_proj → gate_up)  # 情形2：把 gate/up 两路权重拼接合并进 gate_up
            elif '.mlp.gate_proj.weight' in hf_name:  # 以 gate_proj 权重为入口
                layer_match = re.search(r'layers\.(\d+)', hf_name)  # 解析层号
                if layer_match:  # 能解析出层号才继续
                    layer_idx = layer_match.group(1)  # 取层号字符串
                    up_name = hf_name.replace('gate_proj', 'up_proj')  # 推导同层 up_proj 权重名

                    if up_name in hf_weights:  # up 权重存在时才合并
                        gate_weight = hf_weight  # 当前权重即 gate
                        up_weight = hf_weights[up_name]  # 取出 up

                        # Concatenate gate and up along output dimension  # 沿输出维（dim=0）拼接 gate 与 up
                        gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)  # 得到与 MergedColumnParallelLinear 一致的布局

                        custom_name = f"model.layers.{layer_idx}.mlp.gate_up.weight"  # 目标合并参数名
                        try:
                            param = model.get_parameter(custom_name)  # 查找该合并参数
                            param.data.copy_(gate_up_weight)  # 拷贝拼接后的权重
                            loaded_params.add(custom_name)  # 记录合并参数已加载
                            loaded_params.add(hf_name)  # 记录 gate 来源已消费
                            loaded_params.add(up_name)  # 记录 up 来源已消费
                        except AttributeError:  # 模型中不存在该参数
                            skipped_params.append((custom_name, "Parameter not found"))

            # 3. Handle gate_up merge for bias (if present)  # 情形3：若存在 bias，同样把 gate/up 的 bias 拼接合并
            elif '.mlp.gate_proj.bias' in hf_name:  # 以 gate_proj.bias 为入口
                layer_match = re.search(r'layers\.(\d+)', hf_name)  # 解析层号
                if layer_match:  # 能解析出层号才继续
                    layer_idx = layer_match.group(1)  # 取层号字符串
                    up_bias_name = hf_name.replace('gate_proj', 'up_proj')  # 推导同层 up_proj.bias 名

                    if up_bias_name in hf_weights:  # up bias 存在时才合并
                        gate_bias = hf_weight  # 当前即 gate bias
                        up_bias = hf_weights[up_bias_name]  # 取出 up bias
                        gate_up_bias = torch.cat([gate_bias, up_bias], dim=0)  # 沿输出维拼接两路 bias

                        custom_name = f"model.layers.{layer_idx}.mlp.gate_up.bias"  # 目标合并参数名
                        try:
                            param = model.get_parameter(custom_name)  # 查找该合并参数
                            param.data.copy_(gate_up_bias)  # 拷贝拼接后的 bias
                            loaded_params.add(custom_name)  # 记录合并参数已加载
                            loaded_params.add(hf_name)  # 记录 gate bias 来源已消费
                            loaded_params.add(up_bias_name)  # 记录 up bias 来源已消费
                        except AttributeError:  # 模型中不存在该参数
                            skipped_params.append((custom_name, "Parameter not found"))

            # 4. Skip k_proj, v_proj, up_proj (already merged)  # 情形4：k/v/up 三类权重已在合并时被消费，直接跳过
            elif any(x in hf_name for x in ['.k_proj.', '.v_proj.', '.up_proj.']):  # 命中任一已合并的权重名
                if hf_name not in loaded_params:  # 若它没被前面的合并步骤登记过
                    skipped_params.append((hf_name, "Merged into qkv_projection or gate_up"))  # 记录为"已并入其他权重"而跳过

            # 5. All other parameters: load directly (names match HF)  # 情形5：其余权重（embed/norm/lm_head 等）按原名直接加载
            else:
                try:
                    param = model.get_parameter(hf_name)  # 在模型中查找同名参数
                    if param.shape != hf_weight.shape:  # 形状不一致（常见于词表 padding 差异）
                        # Handle vocab size mismatch for embeddings/lm_head  # 处理嵌入/lm_head 的词表大小不匹配
                        if len(param.shape) > 0 and len(hf_weight.shape) > 0:  # 双方都是非标量张量
                            min_size = min(param.shape[0], hf_weight.shape[0])  # 取词表维的较小者
                            param.data[:min_size].copy_(hf_weight[:min_size])  # 只拷贝重叠部分（多余行为随机初始化）
                        else:
                            param.data.copy_(hf_weight)  # 标量情形直接拷贝
                    else:
                        param.data.copy_(hf_weight)  # 形状一致时整块拷贝
                    loaded_params.add(hf_name)  # 记录该参数已加载
                except AttributeError:  # 模型中不存在该参数
                    skipped_params.append((hf_name, "Parameter not found"))

        except Exception as e:  # 兜底：任何意外异常都不中断整体加载
            skipped_params.append((hf_name, f"Error: {str(e)}"))

    # Check for model parameters that weren't loaded  # 反向检查：模型中未被任何权重覆盖的参数
    unloaded_params = []  # 收集未加载的参数名
    for name, param in model.named_parameters():  # 遍历模型全部参数
        if name not in loaded_params:  # 不在已加载集合中
            unloaded_params.append(name)  # 记录为未加载

    print(f"\n{'='*80}")  # 打印分隔线
    print(f"Weight Loading Summary:")  # 打印汇总标题
    print(f"{'='*80}")  # 打印分隔线
    print(f"Successfully loaded: {len([p for p in loaded_params if not any(x in p for x in ['.k_proj.', '.v_proj.', '.up_proj.'])])} parameter groups")  # 统计成功加载的参数组数（排除已被合并的 k/v/up 来源名）

    if unloaded_params:  # 存在未加载参数时给出警告
        print(f"\n⚠️  WARNING: {len(unloaded_params)} model parameters NOT loaded from checkpoint:")  # 打印未加载参数数量
        for name in unloaded_params[:15]:  # 最多展示前 15 个
            param = dict(model.named_parameters())[name]  # 取出该参数
            print(f"  - {name} (shape: {param.shape}, mean: {param.data.mean():.6f})")  # 打印参数名、形状与均值（帮助定位）
        if len(unloaded_params) > 15:  # 超过 15 个时提示省略数量
            print(f"  ... and {len(unloaded_params) - 15} more")

    if skipped_params:  # 存在被跳过的权重时按原因分组打印
        # Group skipped by reason  # 按跳过原因分组
        merged_skips = [s for s in skipped_params if "Merged" in s[1]]  # 已合并进其他权重的
        not_found_skips = [s for s in skipped_params if "not found" in s[1]]  # 模型中找不到对应参数的
        no_mapping_skips = [s for s in skipped_params if "No mapping" in s[1]]  # 无映射规则的

        if merged_skips:  # 打印已合并类数量
            print(f"Skipped (merged into other weights): {len(merged_skips)}")
        if not_found_skips:  # 打印未找到类详情
            print(f"Skipped (not found in model): {len(not_found_skips)}")
            for name, reason in not_found_skips[:5]:  # 最多展示 5 个
                print(f"  - {name}")
            if len(not_found_skips) > 5:  # 超出时提示省略数量
                print(f"  ... and {len(not_found_skips) - 5} more")
        if no_mapping_skips:  # 打印无映射类详情
            print(f"Skipped (no mapping rule): {len(no_mapping_skips)}")
            for name, reason in no_mapping_skips[:5]:  # 最多展示 5 个
                print(f"  - {name}")
            if len(no_mapping_skips) > 5:  # 超出时提示省略数量
                print(f"  ... and {len(no_mapping_skips) - 5} more")

    print(f"{'='*80}")  # 打印结尾分隔线
    return loaded_params  # 返回已加载参数名集合
