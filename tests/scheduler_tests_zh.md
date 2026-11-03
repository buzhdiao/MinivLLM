# 调度器测试说明

> 中文版说明文档，对应原文：[scheduler_tests.md](scheduler_tests.md)（结构与章节保持一致，代码、命令不做翻译）

## 环境准备

```bash
pip install pytest xxhash
```

## 运行方式

运行全部测试：
```bash
python3 -m pytest tests/test_scheduler.py -v
```

运行指定测试类：
```bash
python3 -m pytest tests/test_scheduler.py::TestBug2TokenLimitBreak -v
python3 -m pytest tests/test_scheduler.py::TestBug1CanAppendFailure -v
python3 -m pytest tests/test_scheduler.py::TestSchedulerHappyPath -v
```

## 测试类说明

### TestBug2TokenLimitBreak

防止在循环中途 token 预算（或序列数上限）耗尽时，序列被静默丢弃。

测试项：
- `test_seq_count_is_correct` — 2-token 预算只容纳 2 个序列；`seq_c` 必须仍保留在 `running` 中
- `test_seq_count_limit_variant` — 同一 bug 的另一触发路径：由 `max_num_sequences` 而非 token 预算触发
- `test_no_sequence_is_lost` — 序列总数守恒：每个序列必须位于 `running`、`waiting` 或 `scheduled` 之一

### TestBug1CanAppendFailure

防止在 `block_manager.can_append` 返回 `False` 时序列丢失。

测试项：
- `test_seq_a_not_lost` — 调用之后 `seq_a` 必须出现在 `running`、`waiting` 或 `scheduled` 中
- `test_total_conservation` — `seq_a` 与 `seq_b` 均不允许消失

### TestSchedulerHappyPath

正常条件下调度器的基本正确性。

测试项：
- `test_prefill_scheduled_first` — 新加入的序列被调度为 prefill 并移入 `running`
- `test_all_running_seqs_scheduled_when_budget_allows` — 当 token 预算足够大时，所有 running 序列都被调度并保留在 `running` 中
- `test_preempt_only_seq_when_cant_append_and_running_empty` — 当唯一的 running 序列无法追加时，它被抢占到 `waiting` 且状态置为 `WAITING`
