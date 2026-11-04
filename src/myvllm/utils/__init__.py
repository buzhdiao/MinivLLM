# 通用工具子包的入口。
# re-export context 模块的三个函数：set_context / get_context / reset_context
# 构成全局推理上下文（当前 batch 的位置信息等）的存取接口，
# 使 attention 等算子无需层层传参即可拿到本轮 forward 的上下文。
