# Final-Hybrid

Final 130-tensor checkpoint trained by initializing from Base-E17 and continuing for 12 epochs on outcome-weighted all-player data, with new action-value and opponent-severity heads initialized from scratch. The competition system combines this neural checkpoint with the frozen deterministic auxiliary controller in `__main__.py`.

Before the post-competition calibration extension, the seven submission source files were verified against the archived competition package after newline normalization. The current paper copy is no longer byte-identical because it contains the disabled-by-default calibration hook; with calibration disabled, its competition inference path is unchanged. The evaluator is not part of the submitted bot.

Post-competition code adds optional Platt calibration for the class-weighted risk logits. `calibrate_risk.py` fits parameters on matches 90%--95% and reports final metrics only on matches 95%--100%; `evaluate.py --risk-calibration NAME=JSON` enables it for a named challenger. The completed 1000-wall adoption test reduced deal-in rate from 16.78% to 16.13% but reduced paired average score by 0.261 (95% CI $[-0.509,-0.010]$). Calibration therefore remains disabled by default, and the selected controller continues to use the original cost-sensitive risk scores.

## 中文

这是最终的 130 张量模型。它从 Base-E17 初始化，在结果加权的四家样本上继续训练 12 个 epoch；新增的动作价值头和逐对手损失严重度头从头初始化。真正参加比赛的是混合系统：该神经网络 checkpoint 与 `__main__.py` 中冻结的确定性辅助控制器共同决定动作。因此，第九名成绩属于整个 Final-Hybrid 系统，不能只归因于 checkpoint。

加入赛后校准扩展之前，七个提交源文件已在统一换行符后与归档比赛包核验一致。当前论文副本因包含默认关闭的校准接口而不再是字节级相同文件；关闭校准时，其比赛推理路径不变。`evaluate.py` 和 `calibrate_risk.py` 不是比赛提交 Bot 的组成部分。

赛后代码增加了可选的 Platt calibration，用于修正类别加权风险 logits 的过度自信。`calibrate_risk.py` 只用 90%--95% 的比赛拟合参数，并只在 95%--100% 上报告最终指标；评测时必须显式传入 `evaluate.py --risk-calibration NAME=JSON` 才会启用。1000 个牌墙的采用测试显示，校准使点炮率从 16.78% 降至 16.13%，但配对平均分下降 0.261，95% CI 为 $[-0.509,-0.010]$。因此该功能继续默认关闭，最终控制器保留原始代价敏感风险分数。
