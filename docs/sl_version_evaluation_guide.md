# 中国标准麻将 SL Bot：版本演进与评测记录

本文档记录 `Chinese-Standard-Mahjong-Lab` 中监督学习 Bot 的版本变化、训练方式、评测工具、实验数据、已知问题与决赛配置。所有最终结论均优先依据固定牌墙、四座轮换和配对置信区间，而不是单一验证集动作准确率。

最后更新：2026-07-06。

## 1. 决赛结论

最终推荐模型为第 18 版 `value_off`：使用 outcome-weighted 全玩家轨迹重新训练得到的 `best18.pkl`，启用第 16/17 版已有 risk head，但关闭尚未证明有效的新 action-value、expected-loss 和 fan-route 推理。

第三轮固定牌墙配对测试中，`value_off` 相对旧 `e17`：

```text
平均分差：+1.0391
95% bootstrap CI：[0.1531, 1.9297]
胜率：24.53% -> 25.72%
点炮率：18.13% -> 14.94%
每局点炮番损失：2.256 -> 1.853
平均排名：2.5003 -> 2.4411
```

置信区间完全高于 0，因此第 18 版已具备统计显著的平均分优势。

最终推理配置：

```python
USE_AUX_RANK = False
USE_RISK_HEAD = True
POSTPROCESS_MODE = 'light'
TENPAI_LEARNED_WEIGHT = 1.0
NORMALIZE_DISCARD_LOGITS = False
FAN_WAIT_WEIGHT = 0.0
ACTION_VALUE_WEIGHT = 0.0
FAN_ROUTE_WEIGHT = 0.0
USE_EXPECTED_LOSS_HEAD = False
```

默认模型路径：

```text
/data/best18.pkl
```

## 2. 历史版本总览

| 版本 | 模型/输入 | 主要变化 | 结论 |
|---|---|---|---|
| `1_0604_0.731_baseline` | `test.pkl`，代码未完整保存 | 最早 baseline，目录名记录约 0.731 | 仅作历史参考 |
| `2_0605` | 60 通道，6-block ResNet，单一 235 动作头 | 基础可见信息、危险度与弃牌位置 | 第一代完整 ResNet |
| `3_0607` | 60 通道，分层动作头 | 动作类型与子动作分解 | 最佳验证准确率约 0.87617 |
| `5` | 与 v3 相同 | 增加“牌河现物”奖励 | 错把任一对手现物视为全局安全，后续废弃 |
| `6` | 与 v3 相同 | 听牌时增强进攻、压低 Pass | 缓解过度防守 |
| `7` | 与 v3 相同 | 风险惩罚乘向听因子 | 远手防守、近手进攻 |
| `8_0608` | 66 通道 | 新增向听、有效进张、七对、清一色距离 | 显式牌型结构，验证准确率约 0.87485 |
| `9_0612` | 70 通道，多任务 ResNet | 分数上下文、胜负/番数/向听/弃牌排序、高番加权、花色增强 | 决赛前主要架构基线，验证准确率约 0.87022 |
| `10` | 与 v9 相同 | 弱化后处理，仅重排接近首选的候选 | 减少规则覆盖模型 |
| `11` | 与 v9 相同 | 更轻风险重排，增加 `none/light` | 建立纯模型与后处理对照 |
| `12` | 与 v9 相同 | 候选弃牌显式计算向听变化和有效进张 | 增加做牌推进价值 |
| `13` | 与 v9 相同 | 连续风险惩罚、危险门控 | 加强弃和控制 |
| `14` | 与 v9 相同 | 局面阶段、对手副露、push level | 动态攻守切换 |
| `15` | 仍为 v9 模型 | 尝试时序与风险推理 | 尚无真正 GRU/risk head，序列重建存在问题 |
| `16_0704` | 70 通道 ResNet + 80 步 GRU，126 tensors | 真实全局弃牌顺序、逐对手风险/损失/听牌、合法八番等待、RTX 5090 训练适配 | 新完整架构，最佳策略准确率约 0.87092 |
| `17` | 与 v16 同架构 | checkpoint 对比、risk/light 消融、合法动作兜底、风险默认启用 | `e17-risk` 成为旧决赛候选 |
| `18` | 130 tensors | outcome-weighted 策略、新 action-value、逐对手 loss、fan-route、24 座次 league evaluator | 主策略显著提升；新 heads 推理暂不启用 |

历史目录并非每一个都是独立训练模型：`5–7` 主要是 v3 后处理；`10–15` 主要是 v9 后处理。旧 checkpoint 必须匹配其历史 `feature.py/model.py/__main__.py`，不能用部分随机加载做正式比较。

## 3. 第 16/17 版核心代码修改

### 3.1 防守与时序

- 修复“任意对手打过即全局安全”的错误，按三个对手分别计算安全性。
- `FeatureAgent.discardEvents` 保存真实桌面弃牌交错顺序。
- GRU 输入最近 80 次弃牌及相对玩家。
- 同筋/形状证据加入时间衰减；较新的弃牌证据更强。
- 增加逐对手 `risk_opp[3,34]`、`tenpai_opp[3]` 和点炮损失标签。
- 对手同花色副露、字牌副露、多副露和后段局面加入威胁判断。

### 3.2 做牌与胡牌

- 候选弃牌计算向听数、有效进张和剩余枚数。
- 调用国标番数计算器检查真实合法的八番等待。
- 七对接近时保护对子。
- nominal tenpai 但无法达到八番时扣分。
- 加入清一色、七对、碰碰和及字牌路线的基础标签。

### 3.3 训练与硬件

- RTX 5090 使用 PyTorch 2.8.0 + CUDA 12.8、BF16、TF32、fused AdamW。
- 支持 `--batch-size`、workers、prefetch、缓存、AMP、compile、resume。
- 新增 CUDA 环境依赖文件 `requirements-cuda.txt`。
- 保存 epoch checkpoint、`best.pkl` 和训练性能日志。

## 4. 专用评测工具

### 4.1 `evaluate.py`

功能：

- 固定随机 seed 和牌墙。
- challenger 轮换四个座位，对手保持冻结。
- 同一进程测试多个 checkpoint。
- 支持 raw/light/risk、逐 challenger 开启 risk。
- 输出逐局 CSV、汇总 CSV、配对 bootstrap JSON 和 SHA-256 manifest。
- 严格检查 checkpoint 架构；v18 仅允许 v17 缺失两个新增 head。
- 记录平均分、排名、胜率、自摸、点炮、大牌点炮、流局听牌、副露、延迟和无效局。
- 最终动作增加合法性防火墙。

输出：

```text
*_games.csv
*_summary.csv
*_paired.json
*_manifest.json
```

当前实现全部对局完成后才写文件；中途 `Ctrl+C` 会丢失内存中的部分结果。后续应增加增量写入与 resume。

### 4.2 `evaluate_league.py`

第 18 版新增四 Bot 复式评测：每个固定牌墙运行四个不同 Bot 的全部 24 种座次排列。用于模拟决赛复式赛，减少“一名 challenger 对三个相同旧模型”的针对性偏差。

## 5. 第一次 checkpoint 筛选

配置：三个 `best-light` 对手；checkpoint 12/13/14/16/17/best；每个模型 500 墙 × 4 座，共 2,000 局。

| 模型 | 平均分 | 平均排名 | 胜率 | 点炮率 | 流局听牌率 | 无效率 |
|---|---:|---:|---:|---:|---:|---:|
| `e12` | -0.3550 | 2.5308 | 22.90% | 18.10% | 62.89% | 0.55% |
| `e13` | -0.3915 | 2.5155 | 22.90% | 16.90% | 65.52% | 0.80% |
| `e14` | -0.2965 | 2.5173 | 23.35% | 17.10% | 68.92% | 0.40% |
| `e16` | 0.0000 | 2.5000 | 23.80% | 17.30% | 61.36% | 0.40% |
| `e17` | **0.1545** | **2.4995** | **24.40%** | 17.25% | 67.50% | **0.35%** |
| `best` | 0.0000 | 2.5000 | 23.80% | 17.30% | 61.36% | 0.40% |

结论：

- `best.pkl` 与 `e16` 在全部 2,000 局关键结果完全一致，视为同一策略。
- `e17-light` 点估计最好，但相对 e16 的 95% CI 跨过 0。
- e14 更稳健，但平均分较低。
- e13 点炮率低，却因进攻损失导致平均分最低。

## 6. Light 与 Risk 候选测试

配置：每个候选 2,000 墙 × 4 座，共 8,000 局；对手固定 `best-light`。

| 配置 | 平均分 | 平均排名 | 胜率 | 点炮率 | 平均点炮番 | 大牌点炮率 |
|---|---:|---:|---:|---:|---:|---:|
| `e16-light` | 0.0000 | 2.5000 | 23.49% | 17.04% | 12.83 | 0.125% |
| `e17-light` | 0.2770 | 2.4928 | 24.04% | **16.95%** | 12.68 | 0.1125% |
| `e14-risk` | 0.5175 | 2.4942 | 24.49% | 17.50% | 12.54 | 0.125% |
| `e16-risk` | 0.7093 | 2.4914 | 24.94% | 18.16% | 12.80 | **0.0875%** |
| `e17-risk` | **0.7573** | 2.4916 | **25.00%** | 18.08% | 12.60 | 0.100% |

配对结果：

```text
e16-risk - e16-light = +0.7093，95% CI [0.3775, 1.0444]
e17-risk - e17-light = +0.4803，近似 95% CI [0.1520, 0.8085]
e17-risk - e16-risk = +0.0480，CI 跨 0
```

结论：risk 显著提高平均分，但没有降低普通点炮率；它增加胡牌与普通点炮，同时略微减少极端大牌点炮。旧决赛候选因此选择 `e17-risk`。

## 7. Tenpai 与 Fan 参数实验

配置：e17-risk；`TENPAI_LEARNED_WEIGHT={0,0.6,1}` × `FAN_WAIT_WEIGHT={0,0.12}`；每组 2,000 局。

| Tenpai | Fan | 平均分 | 胜率 | 点炮率 |
|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.897 | 25.25% | 17.95% |
| 0.0 | 0.12 | 0.897 | 25.25% | 17.95% |
| 0.6 | 0.0 | 0.916 | 25.25% | 17.90% |
| 0.6 | 0.12 | 0.916 | 25.25% | 17.90% |
| 1.0 | 0.0 | **0.934** | 25.25% | 17.95% |
| 1.0 | 0.12 | **0.934** | 25.25% | 17.95% |

逐局比较：

- Fan 权重 0.12 在六组中改变结果的牌局数均为 0。
- Tenpai 1.0 -> 0.6 只改变 10/2,000 局。
- Tenpai 1.0 -> 0.0 改变 24/2,000 局。
- 所有差异 CI 均跨 0，点估计以完全使用模型 tenpai 的 1.0 最好。

最终保留 `TENPAI_LEARNED_WEIGHT=1.0`，关闭 fan 奖励。

## 8. 第 18 版训练修改

### 8.1 Outcome-weighted policy

预处理新增：

```bash
--policy-weighting winner
--policy-weighting outcome
--policy-weighting all
```

最终使用 `outcome`：保留所有玩家轨迹，按终局收益加权，并强烈下调点炮玩家策略权重。与旧 winner-only 模仿相比，它使模型学到安全弃和，而不是只模仿赢家的进攻路线。

### 8.2 新 heads

- `action_value[235]`：预测每个动作的归一化终局收益；当前只监督真实执行动作。
- `risk_loss_opp[3,34]`：分别预测三个对手的点炮损失。
- 补全 `fan_route` 的均衡、七对、清一色/混一色、碰碰和、字牌五类。
- 模型由 126 tensors 增至 130 tensors。

训练命令：

```bash
python preprocess.py --workers 16 --policy-weighting outcome

python -u supervised.py \
  --device cuda \
  --amp bf16 \
  --batch-size 2048 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --cache-size 512 \
  --epochs 12 \
  --resume model/rtx5090/checkpoint/17.pkl \
  --action-value-loss-weight 0.20 \
  --risk-severity-opp-loss-weight 0.10 \
  --fan-route-loss-weight 0.05 \
  --logdir model/value_v1
```

旧 v17 加载时显示 `126/130` 属正常现象；两个新增层随机初始化。新 v18 应完整加载 `130/130`。

## 9. 第 18 版六组消融

配置：每组 500 墙 × 4 座，共 2,000 局。

| 配置 | Action value | Expected loss | Fan route | 平均分 | 胜率 | 点炮率 |
|---|---:|---:|---:|---:|---:|---:|
| `old_e17` | 关 | 关 | 关 | 0.9340 | 25.25% | 17.95% |
| `value_off` | 关 | 关 | 关 | 1.7715 | **26.35%** | 15.70% |
| `value_only` | 0.20 | 关 | 关 | 1.7500 | 26.30% | 15.60% |
| `expected_loss_only` | 关 | 开 | 关 | 1.7715 | **26.35%** | 15.70% |
| `value_loss` | 0.20 | 开 | 关 | 1.7500 | 26.30% | 15.60% |
| `all_heads` | 0.20 | 开 | 0.10 | **1.7955** | **26.35%** | **15.50%** |

逐局消融：

```text
value_off - old_e17：+0.8375，95% CI [-0.2928, 1.9678]
value_only - value_off：-0.0215，仅改变 11 局
expected_loss_only - value_off：0，改变 0 局
all_heads - value_off：+0.0240，仅改变 18 局
```

结论：主要收益来自 outcome-weighted 后的共享主策略；新 heads 尚无可靠推理收益。因此决赛采用 `value_off`。

## 10. 第三轮最终配对测试

配置：800 墙 × 4 座；每模型 3,200 局；两个模型合计 6,400 局；10,000 次 bootstrap。

| 模型 | 平均分 | 平均排名 | 胜率 | 点炮率 | 平均点炮番 | 大牌点炮率 | 无效率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `old_e17` | 0.4941 | 2.5003 | 24.53% | 18.13% | 12.45 | 0.0938% | 0.875% |
| `value_off` | **1.5331** | **2.4411** | **25.72%** | **14.94%** | 12.41 | 0.0938% | **0.781%** |

配对统计：

```text
平均分差：+1.0391
95% CI：[0.1531, 1.9297]
```

座位分项：

```text
座位0：2.580 -> 2.111，差 -0.469
座位1：0.020 -> 1.421，差 +1.401
座位2：0.374 -> 2.354，差 +1.980，单座 CI 高于 0
座位3：-0.998 -> 0.246，差 +1.244
```

流局听牌：

```text
old_e17：141 次流局，103 次听牌
value_off：142 次流局，59 次听牌
```

新模型更愿意在危险且难以获胜的局面弃和，因此流局听牌下降；但胜率更高、点炮更低、最终得分显著提高，说明总收益取舍有效。

## 11. 开发与评测过程中修复的问题

### 11.1 环境与依赖

- Windows 安装 `PyMahjongGB` 需要 Microsoft C++ Build Tools；AutoDL Linux 可安装编译工具后构建。
- Conda 非交互 shell 需执行 `source "$(conda info --base)/etc/profile.d/conda.sh"`。
- `data/count.json` 缺失意味着运行目录或预处理数据未上传。
- RTX 5090 要求支持 Blackwell 的 CUDA 12.8 PyTorch 构建。

### 11.2 SL/RL 接口冲突

`RL/env.py` 需要 `observation_space/action_space`，SL agent 缺失导致初始化失败。已在 SL 基类添加兼容属性。

### 11.3 双 NumPy 模块冲突

AutoDL 中 PyTorch、NumPy 和编译版 Mahjong 扩展可能形成两个 `numpy.ndarray` 类型身份，出现：

```text
expected np.ndarray (got numpy.ndarray)
__array_wrap__ argument must be numpy.ndarray, not numpy.ndarray
Unable to configure default ndarray.__repr__
```

修复：

- 输入通过 Python buffer 构建 Torch tensor。
- aux 输出使用 `.tolist()`。
- logits 不使用 `tensor.numpy()`，先 `.tolist()` 再由当前 NumPy 创建数组。
- Dataset 动作标签显式转换为 Python `int`。

### 11.4 评测中断

旧 evaluator 仅结束时写文件；中断后部分局数不会保存。后续应实现周期 flush 和 `--resume`。

### 11.5 非法动作

推理与 evaluator 均加入最终合法动作防火墙；若后处理输出非法动作，退回原模型最佳合法动作。无效率仍未完全归零，剩余问题可能来自环境同时响应和结算逻辑。

## 12. 决赛提交清单

必须使用当前 `src/SL` 最新文件：

```text
agent.py
feature.py
model.py
__main__.py
best18.pkl
```

不要使用 `models and logs/SL/18` 中较早保存的旧 `model.py/__main__.py`，它们可能仍是 126 tensors 或保留 `tensor.numpy()` 调用。

提交前检查：

1. `best18.pkl` 位于 `/data/best18.pkl`。
2. 启动时 checkpoint 必须完整匹配 130 tensors。
3. 决赛配置与第 1 节一致。
4. 冒烟测试能完成至少 20 个固定牌墙。
5. stderr 无 checkpoint mismatch、NumPy 或非法动作异常。
6. 单次平均决策约 2.4 ms，满足时间要求。

## 13. 赛后继续优化方向

1. 为 action-value 生成反事实标签，而不是只监督日志中实际动作。
2. 将 expected-loss 直接纳入统一期望得分公式，避免当前与旧 risk 取 `max` 后不产生影响。
3. 用固定小型 league 的平均分选择 checkpoint，不再只按 policy accuracy 保存 `best.pkl`。
4. 增加激进、高番、门清和强防守等多风格对手。
5. 为 evaluator 增加增量保存、断点续测、24 座次配对置信区间。
6. 论文中分别消融 outcome weighting、GRU 时序、逐对手风险、合法八番等待与规则后处理。

