# Chinese Standard Mahjong Lab

本项目实现了一个面向 Botzone 国标麻将平台与 IJCAI 麻将人工智能比赛的国标麻将 AI Bot，核心目标是在四人、不完全信息、8 番起胡的复杂规则下，训练一个兼顾胡牌能力、做番能力和防守稳定性的智能体。基础版本曾在 16 支队伍中排名第 12；经过监督目标、弃牌时序和风险推理等改进后，最终提交 Bot 在决赛中取得第 9 名。

## 项目背景

国标麻将具有动作空间大、隐藏信息多、奖励稀疏、四人博弈非平稳等特点。相比围棋、五子棋等完全信息游戏，麻将 Bot 不仅要判断自己能否更快胡牌，还要根据弃牌、副露、牌墙进度和分数形势估计点炮风险。本项目主要探索两条路线：

- 使用监督学习（SL）从牌谱中学习强 Bot 的决策，并通过特征工程和推理后处理提升实战表现。
- 使用强化学习（RL）在 SL 初始化基础上进行 PPO 自博弈微调，探索长期收益优化。

当前具有完整训练日志和固定牌墙评测记录的是 SL 主线。仓库虽然保留了 PPO 自博弈代码和部分历史 checkpoint，但缺少足以进行等算力比较的 RL 训练日志、环境步数和多随机种子实验，因此本项目暂不声称 RL 弱于 SL；RL 被保留为后续在强 SL checkpoint 基础上的策略微调方向。

## 核心算法

### 监督学习 Bot

SL 部分位于 `src/SL`，当前版本包含以下设计：

- `feature.py`：将麻将局面编码为 `70 x 4 x 9` 张量，其中 66 个通道在当前数据与比赛运行时实际启用。特征包括手牌、副露、弃牌、剩余牌估计、牌墙进度、对手危险度、弃牌新近度、向听数、有效牌和番型路线。预留的 4 个公开积分上下文通道保持为零，当前系统没有调用 `set_public_scores()`，也不在论文中声称使用了该信息。
- `model.py`：使用六个残差块提取牌面特征，并以 GRU 编码最近 80 次全局弃牌的真实顺序和相对玩家。动作空间被分解为 8 类动作类型和具体动作子头，同时包含胜率、番数、向听、弃牌排序、逐对手点炮风险、损失严重度、对手听牌、番型路线和动作价值等辅助头。
- `preprocess.py`：把原始牌谱转换为 `.npz` 训练样本，支持 `winner`、`all` 和 `outcome` 三种策略监督方式。`outcome` 会保留四家轨迹，根据终局得分连续调整样本权重，并额外降低终局点炮者轨迹的权重。
- `dataset.py`：实现懒加载 LRU 缓存和花色置换数据增强，降低数据读取开销。
- `supervised.py`：使用 AdamW、余弦学习率退火、加权交叉熵、动作类型损失、条件子动作损失和多任务辅助损失训练模型。
- `__main__.py`：Botzone 交互入口。推理阶段在模型输出上加入轻量后处理，包括胡牌优先、听牌进攻、现物奖励、危险牌惩罚、吃碰杠惩罚、七对子保对子和后期防守门控。

### 强化学习 Bot

RL 部分位于 `src/RL`，实现 PPO 自博弈框架：

- `actor.py`：并行采样对局，支持与历史模型池中的对手对战。
- `learner.py`：使用 PPO clipped objective、GAE、价值函数、熵正则和 SL teacher KL 约束更新策略。
- `model_pool.py`：维护历史模型，增加自博弈对手多样性。
- `env.py`：封装国标麻将环境，并加入胡牌得分、流局听牌奖励和未听牌惩罚。
- `train.py`：训练入口，支持 SL checkpoint warm start、断点续训、NPU/CUDA/CPU 自动选择和 TensorBoard 日志。

由于麻将奖励稀疏且四人自博弈样本效率较低，RL 需要更完整的训练预算和实验记录。后续计划从冻结的强 SL checkpoint 初始化，在统一的环境、对手池和评测牌墙下，按相同 GPU 时长或环境步数比较 PPO 微调与纯 SL。

## 项目结构

```text
.
├── src/
│   ├── SL/                              # 监督学习主线
│   │   ├── __main__.py                  # Botzone 推理入口与轻量后处理
│   │   ├── feature.py                   # 70 通道容器（66 个启用）与局面特征
│   │   ├── model.py                     # ResNet + GRU 多头策略网络
│   │   ├── preprocess.py                # 牌谱预处理与结果加权
│   │   ├── dataset.py                   # 数据加载、缓存与花色增强
│   │   ├── supervised.py                # 监督训练入口
│   │   ├── evaluate.py                  # 固定牌墙配对评测
│   │   ├── evaluate_league.py           # 多模型联赛评测
│   │   └── data/                        # 数据格式说明与示例；完整牌谱不入库
│   └── RL/                              # PPO 自博弈强化学习实验
│       ├── train.py                     # RL 训练入口
│       ├── actor.py / learner.py        # 并行采样与策略更新
│       ├── env.py                       # 国标麻将环境封装
│       └── model_pool.py                # 历史对手模型池
├── docs/
│   ├── final report/                     # 课程结课报告
│   │   ├── mahjong_sl_report.tex
│   │   └── mahjong_sl_report.pdf
│   ├── reference/                        # 文献调研与本地参考论文
│   │   ├── deep-research-report.md
│   │   └── algorithms-18-00738.pdf
│   ├── sl_version_evaluation_guide.md    # SL 版本、训练与评测记录
│   └── Presentation_craft_final_bilingual.md  # IJCAI 中英文讲稿
├── deliverables/
│   ├── IJCAI_Mahjong_Final_Presentation.pdf  # 最终无 Backup 演示稿
│   ├── IJCAI_Mahjong_Final_Presentation.tex  # 最终演示稿入口
│   ├── IJCAI_Mahjong_Full_With_Backup.pdf    # 带 Backup 的完整版本
│   ├── IJCAI_Mahjong_Full_With_Backup.tex    # Beamer 共享主源码
│   └── data/                            # PPT 图表使用的精简 CSV
├── models and logs/
│   ├── SL/                              # 历史 SL 代码、训练记录与评测结果
│   └── RL/                              # 历史 RL 代码与实验记录
├── requirements.txt                     # 通用依赖
├── requirements-cuda.txt                # NVIDIA CUDA 环境依赖
├── requirements-npu.txt                 # Ascend NPU 环境依赖
├── .gitignore                           # 数据、权重与构建产物忽略规则
├── LICENSE
├── example.jpg
└── README.md
```

## 环境依赖

通用环境建议使用 Python 3.10 或 3.11；RTX 5090 配置按 `requirements-cuda.txt` 使用 Python 3.11、PyTorch 2.7+ 与 CUDA 12.8 构建。核心依赖包括：

- `torch`
- `numpy`
- `MahjongGB`
- `tensorboard`（可选，用于 RL 日志）
- `torch_npu`（可选，仅 Ascend NPU 环境需要）

安装依赖：

```bash
pip install -r requirements.txt
```

如果在 Ascend NPU 环境训练，需要额外安装与机器 CANN 版本匹配的 `torch_npu`。

## 运行指南

### 1. 预处理 SL 数据

将原始牌谱放在 `src/SL/data/data.txt`，然后运行：

```bash
cd src/SL
python preprocess.py --workers 4
```

当前论文方法推荐生成结果加权的四家样本：

```bash
python preprocess.py --workers 4 --policy-weighting outcome
```

用于消融实验时，可将 `outcome` 替换为 `winner` 或 `all`。`--all-players` 仍可使用，但只是 `--policy-weighting all` 的兼容别名。

### 2. 训练 SL 模型

```bash
cd src/SL
python -u supervised.py --num-workers 0 --epochs 16 --batch-size 1024
```

在部分 Ascend/NPU 环境中，`num-workers > 0` 可能因为多进程与算子编译交互而卡住。若无输出或长时间停在编译 warning 后，建议先使用 `--num-workers 0` 保证稳定，再逐步尝试 1 或 2。

实时查看日志可以使用：

```bash
python -u supervised.py --num-workers 0 --epochs 16 --batch-size 1024 > training.log 2>&1
tail -f training.log
```

### 3. 运行 Botzone SL Bot

`src/SL/__main__.py` 默认读取 `/data/best18.pkl`，也可以通过环境变量指定模型路径：

```bash
cd src/SL
MODEL_PATH=./model/checkpoint/best.pkl python __main__.py
```

Windows PowerShell 中可写为：

```powershell
$env:MODEL_PATH="./model/checkpoint/best.pkl"
python __main__.py
```

当前推理配置位于 `src/SL/__main__.py` 文件开头，并非环境变量：

- `POSTPROCESS_MODE = 'none'`：关闭规则后处理，直接使用模型输出。
- `USE_AUX_RANK = True`：使用弃牌排序辅助头参与打牌评分。
- `USE_RISK_HEAD = True`：使用逐对手风险头参与点炮风险估计。
- `ACTION_VALUE_WEIGHT`、`FAN_WAIT_WEIGHT`：控制实验性动作价值与合法番数等待调整；决赛配置中均为 `0.0`。

### 4. 训练 RL 模型

```bash
cd src/RL
python train.py --sl_checkpoint ../SL/model/checkpoint/best.pkl --num_actors 8 --episodes_per_actor 200
```

RL 训练耗时远高于 SL。若只是复现实战提交，优先使用 SL checkpoint；RL 更适合作为后续长时间微调实验。

## 主要优化总结

- 使用 70 通道兼容张量，其中 66 个通道实际启用；未调用公开积分上下文接口，避免误把未使用的预留通道写成贡献。
- 使用 ResNet + GRU 融合结构，同时建模静态牌面和最近 80 次全局弃牌的真实时间顺序。
- 将 235 维动作空间拆成动作类型与子动作，降低训练难度。
- 引入结果加权行为克隆：保留四家决策，根据终局得分调整监督强度，并降低终局点炮者轨迹权重。
- 引入逐对手点炮风险、损失严重度、对手听牌、番型路线、向听、胜率、弃牌排序和动作价值辅助任务。
- 推理阶段加入轻量攻守平衡后处理。现有开发集数据显示它提高了平均分，但没有降低总体点炮率，因此应同时报告点炮频率、损失严重度和平均得分。
- 实现 PPO 自博弈框架，为 SL 到 RL 的长期迁移预留接口。

## 文档导航

- [SL 版本与评测指南](docs/sl_version_evaluation_guide.md)：记录各版本代码、训练配置、对战结果和已知限制。
- [文献调研](docs/reference/deep-research-report.md)：整理国标麻将监督学习、强化学习和公开系统。
- [课程报告](docs/final%20report/mahjong_sl_report.pdf)：介绍项目早期实现与课程实验。

## Bot实操展示
![](https://ik.imagekit.io/Amyxue/Chinese_Standard_Mahjong/example.jpg)
## 参考资料

- Suphx: Mastering Mahjong with Deep Reinforcement Learning.
- IJCAI Mahjong AI Competition / Botzone Chinese Standard Mahjong.
- `docs/reference/deep-research-report.md` 中整理的公开麻将 Bot 文献与实现思路。
- MahjongGB 国标麻将规则与计番库。

## AI 工具声明

本项目开发过程中使用了 ChatGPT 5.5 Codex 辅助进行代码审查、特征设计讨论、训练问题排查、参数调优建议和文档润色。核心代码结构、训练实验、比赛提交和最终策略取舍均由作者结合实验结果人工确认与修改。
