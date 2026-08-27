# Base-E16 / Reference

Selected best-validation model from the shared base run. It corresponds to checkpoint index 16 and validation epoch 17 (loss 0.3426, policy accuracy 0.8709). The included plain state dictionary has the same 126 tensor payloads as the archived full `checkpoint/16.pkl` and was used as the frozen reference opponent.

`base_training.log` is the original RTX 5090 log. `base_training_curve.csv` is a direct extraction of its 18 validation records. This is the base stage inherited by the final system, not the missing outcome-weighted fine-tuning log.

## 中文

这是同一条基础训练中根据验证集选择的最佳模型，对应 checkpoint 16 和第 17 次验证：验证损失为 0.3426，策略准确率为 0.8709。目录中的纯模型状态字典包含 126 个张量，其内容与历史完整文件 `checkpoint/16.pkl` 中的模型张量完全一致。该模型也是已有评测使用的冻结参考对手。

`base_training.log` 是 RTX 5090 上的原始基础训练日志，`base_training_curve.csv` 是从日志的 18 次验证记录中直接提取的曲线数据。这条曲线属于最终系统继承的基础监督训练阶段，不能冒充已经丢失的 outcome-weighted 后续微调日志。
