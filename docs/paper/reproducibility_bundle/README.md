# 麻将论文可复现实验包 / Mahjong paper reproducibility bundle

## 中文说明

这是与论文方法、表格和复现实验一一对应的公开精简包。目录采用论文中的名称，而不是开发阶段容易混淆的历史编号。本地完整副本包含匹配的模型文件；Git 版本只跟踪可运行代码、版本清单、训练曲线以及论文实际使用的 CSV/JSON/PDF 证据。原始牌谱、生成的 `.npz`、模型权重、临时日志、PID、smoke test 和无关历史版本由 `.gitignore` 排除。允许公开的 checkpoint 将通过带哈希的论文附件或版本化 release 单独提供。

### 训练谱系与论文名称

```text
基础监督训练（18 次验证）
  epoch 12 -> epoch 14 -> epoch 16 / 选定参考模型 -> epoch 17
                                                    |
                                                    +-- 初始化 12 epoch 结果加权微调
                                                            -> 最终混合比赛系统
```

| 目录                                 | 论文名称               | 用途                                |
| ---------------------------------- | ------------------ | --------------------------------- |
| `01_base_epoch12`                  | Base-E12           | 早期检查点，用于检查点选择曲线和本地对手敏感性实验。        |
| `02_base_epoch14`                  | Base-E14           | 中期检查点，用于检查点选择曲线和本地对手。             |
| `03_base_epoch16_reference`        | Base-E16/Reference | 基础训练中验证集表现最好的检查点，也是已有评测使用的冻结参考对手。 |
| `04_base_epoch17`                  | Base-E17           | 基础训练最后一个检查点，也是最终模型继续训练前的基线。       |
| `05_final_outcome_weighted_hybrid` | Final-Hybrid       | 结果加权继续训练后的最终神经网络，以及比赛使用的冻结辅助控制器。  |

历史名称 `best16` 和 `best17` 容易造成误解。`best16.pkl` 是一个包含 126 个张量的纯模型状态字典，其张量内容与完整的 `checkpoint/16.pkl` 完全一致；`best17.pkl` 与下一 epoch 的 `checkpoint/17.pkl` 文件完全一致。因此它们是同一次基础训练中相邻的两个检查点，不是独立随机种子。为了避免保存两份内容相同的 Base-E16 权重，本包只保留精简的参考模型。

RTX 5090 基础训练日志及从中提取的 CSV 位于 `03_base_epoch16_reference`。它们记录了最终系统继承的基础监督训练阶段，而不是结果加权微调阶段的完整曲线。最终模型从 Base-E17 初始化，在结果加权数据上继续训练 12 个 epoch，并从头初始化两个新增预测头；该阶段日志没有保存，论文中必须如实说明。

### 仅为可移植性所做的代码调整

- 每个 `__main__.py` 默认读取同目录的 `model.pkl`，仍可通过 `MODEL_PATH` 覆盖。
- 每个版本目录都包含评测所需的本地 `env.py` 快照，因此不再依赖仓库中是否存在 `src/RL`；`MAHJONG_RL_DIR` 仍保留为兼容性后备设置。
- 最终 evaluator 额外支持 `--raw-challenger`、严格的无效牌墙排除和按牌墙聚类的 bootstrap 置信区间。
- 最终系统额外提供默认关闭的赛后 Platt calibration；校准器按 challenger 绑定并校验 checkpoint SHA-256，避免跨模型误用。

默认配置不改变特征、网络权重、比赛策略逻辑或控制器系数。只有显式传入校准文件时，推理阶段才会将风险 logit 映射为校准概率。该赛后变体已经完成单独消融：虽然概率校准显著改善 ECE/Brier 并降低点炮率，但配对平均分下降 `0.261`，95% CI `[-0.509, -0.010]`，因此没有进入最终默认控制器。

### 为什么不包含其他历史版本

版本 1--11 没有支撑论文中的定量结论，其中一些也缺少完整可靠的代码—checkpoint 绑定。版本 12、14、16、17 是唯一一条具有完整训练日志的基础训练轨迹中的代表性节点。版本 13 不会提供超出原始密集 checkpoint sweep 的论文证据，因此不重复收录。版本 18 和 19 合并表示为 `05_final_outcome_weighted_hybrid`：版本 19 提供已核验的比赛源代码，版本 18 提供匹配的最终权重。

A--F 完成结果、剩余论文实验和可直接在 AutoDL 执行的复现命令见 `EXPERIMENT_PLAN.md`。如果审稿人要求补做结果加权的因果消融，完整的数据传输、预处理、训练、评测、打包和下载流程见 `OWBC_CONTROLLED_RETRAINING.md`。

## English

This public, paper-facing bundle uses manuscript labels rather than ambiguous historical folder numbers. The complete local copy contains each matching checkpoint, whereas Git tracks runnable source, manifests, the base training curve, and the CSV/JSON/PDF evidence used by the paper. Raw records, generated arrays, checkpoint binaries, transient logs, PID files, smoke tests, and unrelated historical versions are excluded. Checkpoints that can be redistributed should be supplied separately through a hash-bound article supplement or versioned release.

## Training lineage and paper labels

```text
base supervised run (18 validations)
  epoch 12 -> epoch 14 -> epoch 16 / selected reference -> epoch 17
                                               |
                                               +-- initialize 12-epoch outcome-weighted fine-tuning
                                                       -> final hybrid competition system
```

| Directory | Paper label | Purpose |
|---|---|---|
| `01_base_epoch12` | Base-E12 | Early checkpoint for the checkpoint-selection curve and local opponent pool. |
| `02_base_epoch14` | Base-E14 | Intermediate checkpoint for the checkpoint-selection curve and local opponent pool. |
| `03_base_epoch16_reference` | Base-E16/Reference | Best validation checkpoint of the base run and frozen opponent used by archived evaluations. |
| `04_base_epoch17` | Base-E17 | Last base-run checkpoint and baseline for comparison with the final system. |
| `05_final_outcome_weighted_hybrid` | Final-Hybrid | Outcome-weighted continuation, final neural checkpoint, and frozen auxiliary controller used in competition. |

The historical names `best16` and `best17` are misleading. `best16.pkl` is a plain 126-tensor state dictionary whose tensor payloads exactly match `checkpoint/16.pkl`; `best17.pkl` is the complete next-epoch checkpoint and exactly matches `checkpoint/17.pkl`. They are two adjacent selections from the same base training run, not independent random seeds. To avoid a duplicate 10 MiB/31 MiB serialization of identical Base-E16 weights, this bundle retains only the plain reference model.

The archived RTX 5090 log and its derived CSV are stored with `03_base_epoch16_reference`. They document the shared base supervised stage inherited by the final system. The final checkpoint was initialized from Base-E17 and then trained for 12 epochs on outcome-weighted data with two new heads initialized from scratch. Its fine-tuning log was not preserved, so the base curve must not be presented as the complete final-stage curve.

## Portability-only source changes

- Each `__main__.py` defaults to the local `model.pkl`; `MODEL_PATH` still overrides it.
- Each version includes a local `env.py` snapshot required by evaluation, so it no longer depends on the repository containing `src/RL`; `MAHJONG_RL_DIR` remains as a compatibility fallback.
- The final evaluator additionally supports `--raw-challenger`, strict invalid-wall filtering, and wall-clustered bootstrap intervals.
- The final system also exposes post-competition Platt calibration, disabled by default and bound to a named challenger with checkpoint SHA-256 validation.

The default configuration does not change feature construction, network weights, competition policy logic, or controller coefficients. Risk logits are calibrated only when a calibration file is explicitly supplied. The completed adoption ablation improved calibration and reduced deal-ins but lowered paired average score by 0.261 (95% CI `[-0.509, -0.010]`), so the post-competition variant is not adopted for control.

## Why other historical versions are absent

Versions 1--11 are not used by a quantitative claim in the paper and several lack a complete code--checkpoint binding. Versions 12, 14, 16, and 17 are representative points from the one fully logged base run. Version 13 is omitted because it adds no paper-specific comparison beyond the denser checkpoint sweep already preserved in the original archive. Versions 18/19 are represented once as `05_final_outcome_weighted_hybrid`: version 19 supplies the verified submission source and version 18 supplies the matching final weights.

All originally planned A--F experiments are complete. See `EXPERIMENT_PLAN.md` for results and retained AutoDL commands. If a reviewer requests a controlled outcome-weighting ablation, follow `OWBC_CONTROLLED_RETRAINING.md` from upload through local checksum verification.
