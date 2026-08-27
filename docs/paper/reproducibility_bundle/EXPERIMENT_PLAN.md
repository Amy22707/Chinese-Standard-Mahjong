# 已完成实验记录与后续计划 / Completed experiments and remaining plan

## 中文说明

该计划不要求重新进行多随机种子训练，也不要求获取第三方 Bot。论文必须明确说明这两项限制。所有主要对局比较仍使用相同牌墙、四个受控座位、冻结对手、严格的无效牌墙排除，以及 10,000 次按牌墙聚类的 bootstrap 采样。

以下所有命令均为 Linux Bash 语法，应在 AutoDL 的 Bash 终端中运行，不要在 Windows PowerShell 中运行。复制多行命令时，行末反斜杠 `\` 后不能再有空格或其他字符。

当前克隆实例的 SSH 入口为 `connect.westd.seetacloud.com:45040`。以下登录命令在本地 Bash 终端执行：

```bash
ssh -p 45040 root@connect.westd.seetacloud.com
```

端口只用于 SSH/SCP 连接，不应写进远程 Python、tmux 或训练命令。实例克隆后仍使用 `/root/autodl-tmp`，因此尚未完成的 G--J 已按该远程路径编写。如果实际仓库位置不同，可先运行 `find /root/autodl-tmp -type d -name Chinese-Standard-Mahjong-Lab -print` 查找后再修改 `ROOT`。

若克隆发生在本次代码更新之前，先在本地 Git Bash 中只同步 G--I 所需的新脚本和本计划；这不会上传或覆盖 `model.pkl` 和既有 `results/`：

```bash
LOCAL_BUNDLE=/d/PKU/CODE/26spring-ai/Homework/Chinese-Standard-Mahjong-Lab/docs/paper/reproducibility_bundle
REMOTE_BUNDLE=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab/docs/paper/reproducibility_bundle
REMOTE=root@connect.westd.seetacloud.com

scp -P 45040 \
  "$LOCAL_BUNDLE/EXPERIMENT_PLAN.md" \
  "$REMOTE:$REMOTE_BUNDLE/EXPERIMENT_PLAN.md"

scp -P 45040 \
  "$LOCAL_BUNDLE/05_final_outcome_weighted_hybrid/__main__.py" \
  "$LOCAL_BUNDLE/05_final_outcome_weighted_hybrid/evaluate.py" \
  "$LOCAL_BUNDLE/05_final_outcome_weighted_hybrid/supervised.py" \
  "$REMOTE:$REMOTE_BUNDLE/05_final_outcome_weighted_hybrid/"
```

如果远程目录不存在，应先确认克隆位置，而不是盲目创建一个可能错误的空目录：

```bash
ssh -p 45040 root@connect.westd.seetacloud.com \
  "find /root/autodl-tmp -type d -name reproducibility_bundle -print"
```

在 AutoDL 上进入最终系统目录后运行：

```bash
cd /root/autodl-tmp/Chinese-Standard-Mahjong-Lab/docs/paper/reproducibility_bundle/05_final_outcome_weighted_hybrid
```

如果实验包上传到了其他位置，请修改仓库路径前缀。各命令块由中英文说明共用，不需要重复执行。

其余长时间实验均使用 `tmux` 后台运行。首次使用时执行：

```bash
apt-get update && apt-get install -y tmux
mkdir -p results
```

每条 `tmux new-session` 命令执行一次即可。至少启动一个会话后，可执行 `tmux set-option -g mouse on` 启用 tmux 鼠标滚动。实验启动后可以安全关闭 SSH；按 `Ctrl+b`，再按 `d` 可主动退出当前会话而不中止实验。常用检查命令如下：

```bash
tmux ls
tmux attach -t SESSION_NAME
tail -f results/EXPERIMENT_NAME.log
```

### 查看实验进程与资源 / Inspect processes and resources

下面的命令均在已登录的 AutoDL Bash 中执行，不会停止实验。

查看现有 tmux 会话以及每个 pane 的 shell PID：

```bash
tmux ls 2>/dev/null || echo "No tmux session is currently running."
tmux list-panes -a \
  -F 'session=#{session_name} pane=#{pane_index} pid=#{pane_pid} command=#{pane_current_command}' \
  2>/dev/null || true
```

查看本计划启动的 Python 进程、运行时间、CPU 和内存占用：

```bash
pgrep -af 'python.*(preprocess|supervised|evaluate|calibrate_risk)\.py' || \
  echo "No matching Mahjong experiment process was found."

PIDS=$(pgrep -d, -f 'python.*(preprocess|supervised|evaluate|calibrate_risk)\.py' || true)
if [ -n "$PIDS" ]; then
  ps -p "$PIDS" -o pid,ppid,stat,etime,%cpu,%mem,args --width 240
fi
```

查看 GPU 占用；第二条会每两秒刷新，按 `Ctrl+C` 退出监控，不会终止训练：

```bash
nvidia-smi
watch -n 2 nvidia-smi
```

查看磁盘、内存以及实验目录大小：

```bash
df -h /root/autodl-tmp
free -h
du -sh /root/autodl-tmp/Chinese-Standard-Mahjong-Lab/docs/paper/ablation_runs 2>/dev/null || true
```

查看尚未完成实验的实时日志，按 `Ctrl+C` 只会退出 `tail`：

```bash
ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
RUNS="$ROOT/docs/paper/ablation_runs/outcome_weighting"
FINAL="$ROOT/docs/paper/reproducibility_bundle/05_final_outcome_weighted_hybrid"

# G：根据当前阶段选择一条。
tail -F "$RUNS/all/preprocess.log"
tail -F "$RUNS/outcome/preprocess.log"
tail -F "$RUNS/all/training.log"
tail -F "$RUNS/outcome/training.log"
tail -F "$FINAL/results/outcome_weighting_ablation.log"

# H、I：分别查看风险范围消融与时序输入压力测试。
tail -F "$FINAL/results/risk_scope_ablation.log"
tail -F "$FINAL/results/sequence_input_stress_test.log"
```

如只想看最近输出而不持续占用终端，把 `tail -F` 改成 `tail -n 40`。如果 tmux 会话已经消失且 `pgrep` 也没有找到对应进程，说明任务已经退出；此时检查日志最后 60 行中的正常完成标记或 traceback：

```bash
tail -n 60 /absolute/path/to/experiment.log
grep -nE 'Traceback|Error|Killed|CUDA out of memory' /absolute/path/to/experiment.log || true
```

进入 tmux 后，按 `Ctrl+b` 再按 `[` 可进入历史浏览模式，使用鼠标滚轮或 PageUp/PageDown 查看，按 `q` 退出浏览模式。如果 `tmux ls` 中对应会话消失，通常表示命令已经结束；此时应检查日志末尾及结果文件，而不是直接重跑。若提示 `duplicate session`，说明同名实验仍在运行或会话仍存在。

## English overview

The plan deliberately does not require multiple training seeds or third-party opponents. Those limitations must remain explicit in the paper. All primary play comparisons still use identical walls, all four controlled seats, a frozen opponent, strict invalid-wall exclusion, and 10,000 wall-cluster bootstrap samples. Run the shared commands below from the final-system directory shown above, adjusting the repository prefix if the bundle is uploaded elsewhere.

The retained commands use detached `tmux` sessions and write a separate log under `results/`. Disconnecting SSH therefore does not terminate them. Completed commands are preserved for provenance and should not be rerun unless the underlying model, controller, data, or wall seeds change.

G--I require the evaluator/trainer revision dated 27 August 2026. After uploading the latest bundle, verify the new switches before launching long jobs:

```bash
python evaluate.py -h | grep -E 'aggregate-risk-challenger|zero-sequence-challenger'
python supervised.py -h | grep -- '--seed'
```

## 已完成结果摘要 / Completed result summary

截至 2026-08-27，A--F 均已完成，日志中没有 Python traceback。最终选择仍为未校准的 `Final-Hybrid + risk-aware controller`；Platt 校准保留用于概率解释，但不进入默认控制器。

As of 27 August 2026, experiments A--F are complete. The selected deployment remains Final-Hybrid with the original uncalibrated risk-aware controller. Platt scaling is retained for probabilistic interpretation but is not enabled by default.

| Experiment | Main result |
|---|---|
| A. Controller ablation | Risk-aware minus raw: `+0.168`, wall-bootstrap 95% CI `[-0.220, 0.550]`; risk-aware has the best score point estimate (`1.458`). |
| B. Final vs Base-E17 | `+0.677`, 95% CI `[-0.139, 1.483]`; deal-in `18.18% -> 15.78%`. |
| C. Opponent sensitivity | Final-Hybrid average score remains positive against Base-E12/E14/E16/E17: `2.138/1.386/1.788/0.918`. |
| D. Risk calibration | Aggregate-risk ECE `0.2600 -> 0.00033`; opponent-risk ECE `0.1438 -> 0.00009`; tenpai ECE `0.1869 -> 0.00407`. |
| E. Checkpoint sweep | Base-E16 minus Base-E12: `+1.093`, 95% CI `[0.230, 1.963]`; Base-E17 is not supported as better than Base-E16. |
| F. Calibrated controller | Calibrated minus uncalibrated: `-0.261`, 95% CI `[-0.509, -0.010]`; calibration is therefore not adopted for control. |

## A. 最终控制器消融（已完成） / Final controller ablation — completed

这是目前因果关系最清楚的比较，因为三种配置使用完全相同的最终神经网络权重，只改变推理控制方式。论文应比较 raw argmax、light controller 和 risk-aware controller。

This is the cleanest causal comparison because all three rows use exactly the same learned checkpoint.

以下命令保留用于记录，不需要重复执行。

The command is retained for provenance and does not need to be rerun.

```bash
python -u evaluate.py \
  --opponent ../03_base_epoch16_reference/model.pkl \
  --challenger final_raw=model.pkl \
  --challenger final_light=model.pkl \
  --challenger final_risk=model.pkl \
  --raw-challenger final_raw \
  --risk-challenger final_risk \
  --postprocess light \
  --opponent-postprocess light \
  --walls 1000 \
  --seed 2026082500 \
  --bootstrap-samples 10000 \
  --output results/final_controller_ablation
```

计算量：3 种策略 × 1000 个牌墙 × 4 个座位 = 12,000 局。

Workload: 3 policies × 1000 walls × 4 seats = 12,000 games.

## B. 新牌墙最终系统比较（已完成） / Fresh-wall final-system comparison — completed

使用与开发阶段不重叠的新牌墙重新比较 Final-Hybrid 和 Base-E17。两者都采用同一个 risk-aware controller，因此该实验主要比较基础检查点与最终继续训练阶段形成的完整模型差异，但仍不能把差异单独归因于 outcome weighting。

This repeats the archived Final-Hybrid versus Base-E17 result on a new wall range. Both use risk-aware control, so the main difference is the learned checkpoint/training stage.

```bash
mkdir -p results
tmux new-session -d -s mahjong_b_fresh \
  "python -u evaluate.py \
    --opponent ../03_base_epoch16_reference/model.pkl \
    --challenger base_e17=../04_base_epoch17/model.pkl \
    --challenger final_hybrid=model.pkl \
    --risk-challenger base_e17 \
    --risk-challenger final_hybrid \
    --postprocess light \
    --opponent-postprocess light \
    --walls 1000 \
    --seed 2026084000 \
    --bootstrap-samples 10000 \
    --output results/fresh_final_vs_base_e17 \
    > results/fresh_final_vs_base_e17.log 2>&1"
```

会话 / session: `mahjong_b_fresh`；日志 / log: `results/fresh_final_vs_base_e17.log`。

计算量：2 种策略 × 1000 个牌墙 × 4 个座位 = 8,000 局。

Workload: 2 policies × 1000 walls × 4 seats = 8,000 games.

## C. 本地对手敏感性（已完成） / Local-opponent sensitivity — completed

分别使用 Base-E12、E14、E16、E17 作为冻结对手，检查主要结论是否依赖某一个本地 checkpoint。由于这些对手共享数据、架构和训练谱系，本实验只能称为“本地 checkpoint 敏感性”，不能称为不同风格或异构强对手泛化测试。

This cannot establish robustness to independent playing styles, but it tests whether the conclusion depends on one frozen checkpoint. Each run uses three copies of one available base-run checkpoint.

```bash
mkdir -p results
tmux new-session -d -s mahjong_c_opponents \
  'for spec in \
    base_e12=../01_base_epoch12/model.pkl \
    base_e14=../02_base_epoch14/model.pkl \
    base_e16=../03_base_epoch16_reference/model.pkl \
    base_e17=../04_base_epoch17/model.pkl
  do
    name=${spec%%=*}
    opponent=${spec#*=}
    echo "===== ${name} started at $(date -Is) ====="
    python -u evaluate.py \
      --opponent "$opponent" \
      --challenger final_hybrid=model.pkl \
      --risk-challenger final_hybrid \
      --postprocess light \
      --opponent-postprocess light \
      --walls 500 \
      --seed 2026086000 \
      --bootstrap-samples 10000 \
      --output "results/local_opponent_${name}" || exit $?
    echo "===== ${name} finished at $(date -Is) ====="
  done > results/local_opponent_sensitivity.log 2>&1'
```

会话 / session: `mahjong_c_opponents`；总日志 / combined log: `results/local_opponent_sensitivity.log`。四种对手按 E12、E14、E16、E17 的顺序串行运行。

计算量：4 种对手设置 × 500 个牌墙 × 4 个座位 = 8,000 局候选模型对局。

Workload: 4 opponent settings × 500 walls × 4 seats = 8,000 candidate games.

## D. 分离拟合与测试的风险校准（已完成） / Disjoint fit/test risk calibration — completed

将 `DATA_DIR` 指向包含 `count.json` 和全部编号 `.npz` 文件的预处理数据目录。该目录不会随 Git 仓库或精简实验包上传；换用新的 AutoDL 实例时，需要从训练实例迁移完整的预处理 `data/`，或者用相同原始数据和 `--policy-weighting outcome` 重新运行预处理。脚本使用 90%–95% 的比赛拟合 Platt scaling，仅用 95%–100% 报告最终校准指标，避免使用同一批样本拟合和测试。

Point the command to the processed `data` directory containing `count.json` and numbered `.npz` files.

```bash
# 先查找远程实例中是否已经存在预处理数据；不要直接启动实验。
find /root/autodl-tmp -type f -name count.json -print 2>/dev/null

# 将下行改为上一步找到的实际 data 目录（不是 count.json 文件本身）。
DATA_DIR=/absolute/path/to/data

# 两项检查都必须成功；只复制 count.json 不足以进行校准。
test -f "$DATA_DIR/count.json" || {
  echo "ERROR: $DATA_DIR/count.json does not exist" >&2
  exit 1
}
find "$DATA_DIR" -maxdepth 1 -type f -name '*.npz' -print -quit | grep -q . || {
  echo "ERROR: no numbered .npz files found in $DATA_DIR" >&2
  exit 1
}

mkdir -p results
tmux new-session -d -s mahjong_d_calibration \
  "python -u calibrate_risk.py \
    --checkpoint model.pkl \
    --data-dir '$DATA_DIR' \
    --device cuda \
    --batch-size 2048 \
    --num-workers 8 \
    --calibration-begin 0.90 \
    --calibration-end 0.95 \
    --test-end 1.00 \
    --output results/final_risk_calibration_platt.json \
    --calibrator-output results/final_risk_calibrator.json \
    > results/final_risk_calibration_platt.log 2>&1"
```

会话 / session: `mahjong_d_calibration`；日志 / log: `results/final_risk_calibration_platt.log`。如果第一条 `find` 没有输出，说明本实例没有可用于校准的预处理数据，应先迁移或重新生成数据，不能通过创建空文件绕过检查。

该命令会同时输出校准前后总体点炮风险、逐对手点炮风险和对手听牌预测的 AUROC/AUPRC、Brier score、10 分箱校准误差和可靠性分箱，并生成与最终 checkpoint SHA-256 绑定的 `final_risk_calibrator.json`。

This reports prevalence, approximate AUROC/AUPRC, Brier score, 10-bin calibration error, reliability-bin data, and positive deal-in severity MAE for aggregate risk, opponent-specific risk, and opponent tenpai.

## E. 新牌墙 checkpoint sweep（已完成） / Fresh checkpoint-selection sweep — completed

已有的 500 牌墙结果足以作为开发阶段的 checkpoint 选择证据。只有在时间充足时才需要重新运行本项。

The archived 500-wall sweep already supports a development-only checkpoint-selection discussion. Run this only if time remains.

```bash
mkdir -p results
tmux new-session -d -s mahjong_e_sweep \
  "python -u evaluate.py \
    --opponent ../03_base_epoch16_reference/model.pkl \
    --challenger base_e12=../01_base_epoch12/model.pkl \
    --challenger base_e14=../02_base_epoch14/model.pkl \
    --challenger base_e16=../03_base_epoch16_reference/model.pkl \
    --challenger base_e17=../04_base_epoch17/model.pkl \
    --postprocess light \
    --opponent-postprocess light \
    --walls 500 \
    --seed 2026088000 \
    --bootstrap-samples 10000 \
    --output results/fresh_base_checkpoint_sweep \
    > results/fresh_base_checkpoint_sweep.log 2>&1"
```

会话 / session: `mahjong_e_sweep`；日志 / log: `results/fresh_base_checkpoint_sweep.log`。

计算量：4 种策略 × 500 个牌墙 × 4 个座位 = 8,000 局。

Workload: 4 policies × 500 walls × 4 seats = 8,000 games.

## F. 校准控制器消融（已完成，未采用） / Calibrated-controller ablation — completed, not adopted

该实验使用相同模型、相同 risk controller 和相同牌墙，唯一差别是是否对辅助 logits 使用 D 生成的 Platt 参数。只有该实验确认平均分、名次和点炮率没有恶化后，才能把校准设为最终默认配置。

This isolates post-hoc calibration from model and controller changes. The paired test found a significant score reduction, so the archived uncalibrated competition configuration remains the default.

```bash
mkdir -p results
tmux new-session -d -s mahjong_f_calibrated \
  "python -u evaluate.py \
    --opponent ../03_base_epoch16_reference/model.pkl \
    --challenger risk_uncalibrated=model.pkl \
    --challenger risk_calibrated=model.pkl \
    --risk-challenger risk_uncalibrated \
    --risk-challenger risk_calibrated \
    --risk-calibration risk_calibrated=results/final_risk_calibrator.json \
    --postprocess light \
    --opponent-postprocess light \
    --walls 1000 \
    --seed 2026091000 \
    --bootstrap-samples 10000 \
    --output results/calibrated_risk_ablation \
    > results/calibrated_risk_ablation.log 2>&1"
```

会话 / session: `mahjong_f_calibrated`；日志 / log: `results/calibrated_risk_ablation.log`。

计算量：2 种策略 × 1000 个牌墙 × 4 个座位 = 8,000 局。

Workload: 2 policies × 1000 walls × 4 seats = 8,000 games.

## G. Outcome weighting 受控训练消融（待运行，优先补充） / Controlled outcome-weighting training ablation — pending, highest priority

> **Authoritative protocol / 唯一执行指南：** 请使用 [`OWBC_CONTROLLED_RETRAINING.md`](OWBC_CONTROLLED_RETRAINING.md)。该文件覆盖本地校验、Git/安全传输、远端预处理、固定种子训练、进程监控、模型冻结、配对评测、结果打包、下载和校验。下方命令仅保留为历史速查，不应与独立指南混用。

该实验用于回答“提升是否来自 outcome-weighted behavior cloning”。`uniform_all` 与 `outcome_weighted` 必须从同一个 Base-E17 checkpoint 出发，采用同一个训练随机种子、相同数据划分、epoch 数和损失权重；唯一变化是预处理阶段的 `--policy-weighting`。这比直接比较 Base-E17 与 Final-Hybrid 更接近因果消融。

This experiment isolates outcome-weighted behavior cloning. Both variants start from the same Base-E17 checkpoint and share the seed, split, epochs, optimizer, and auxiliary-loss weights. Only preprocessing-time policy weighting changes.

### G0. 清理未完成的旧 G 实验 / Reset an incomplete G run

如果 G 曾经运行到一半，先执行以下块。它只终止 `mahjong_g_preprocess`、`mahjong_g_train`、`mahjong_g_eval` 三个 G 专用会话，然后删除隔离的 G 运行目录和以 `outcome_weighting_ablation` 开头的旧 G 评测文件。它不会删除 `src/SL/data`、任何 `model.pkl` 或 A--F 结果。输入确认词前会显示待删除目录的空间占用和旧结果文件。

If G was interrupted, run this reset block before G1. It deliberately refuses to delete anything unless the resolved paths exactly match the documented G directories.

```bash
ROOT=$(realpath -m /root/autodl-tmp/Chinese-Standard-Mahjong-Lab)
RUNS=$(realpath -m "$ROOT/docs/paper/ablation_runs/outcome_weighting")
FINAL=$(realpath -m "$ROOT/docs/paper/reproducibility_bundle/05_final_outcome_weighted_hybrid")
RESULTS=$(realpath -m "$FINAL/results")
EXPECTED_RUNS="$ROOT/docs/paper/ablation_runs/outcome_weighting"
EXPECTED_RESULTS="$ROOT/docs/paper/reproducibility_bundle/05_final_outcome_weighted_hybrid/results"

test -d "$ROOT/docs/paper/reproducibility_bundle" || {
  echo "ERROR: repository root is wrong: $ROOT" >&2
  exit 1
}
if [ "$RUNS" != "$EXPECTED_RUNS" ] || [ "$RESULTS" != "$EXPECTED_RESULTS" ]; then
  echo "ERROR: refusing unexpected cleanup paths" >&2
  printf 'RUNS=%s\nRESULTS=%s\n' "$RUNS" "$RESULTS" >&2
  exit 1
fi

for session in mahjong_g_preprocess mahjong_g_train mahjong_g_eval; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Stopping tmux session: $session"
    tmux kill-session -t "$session"
  fi
done

sleep 2
REMAINING=""
for pid in $(pgrep -f 'python.*(preprocess|supervised|evaluate)\.py' || true); do
  cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
  cmd=$(tr '\0' ' ' 2>/dev/null < "/proc/$pid/cmdline" || true)
  if [[ "$cwd" == "$RUNS"/* || "$cmd" == *"$RUNS"* ]]; then
    REMAINING+="pid=$pid cwd=$cwd cmd=$cmd"$'\n'
  fi
done
if [ -n "$REMAINING" ]; then
  echo "ERROR: a G process is still running; nothing was deleted:" >&2
  printf '%s\n' "$REMAINING" >&2
  exit 1
fi

echo "The following G artifacts will be deleted:"
du -sh "$RUNS" 2>/dev/null || echo "$RUNS does not exist"
find "$RESULTS" -maxdepth 1 -type f \
  -name 'outcome_weighting_ablation*' -print 2>/dev/null || true
read -r -p 'Type DELETE_G to continue: ' CONFIRM
if [ "$CONFIRM" != "DELETE_G" ]; then
  echo "Cancelled; nothing was deleted."
  exit 1
fi

rm -rf -- "$RUNS"
if [ -d "$RESULTS" ]; then
  find "$RESULTS" -maxdepth 1 -type f \
    -name 'outcome_weighting_ablation*' -delete
fi
mkdir -p "$RUNS" "$RESULTS"
echo "G cleanup complete. Continue with G1 below."
df -h /root/autodl-tmp
```

### G1. 重新准备并启动预处理 / Recreate the run and start preprocessing

以下准备步骤会为两个变体各生成一份预处理数据，可能占用较多磁盘。先确认容量；`RAW_DATA` 必须指向原始 `data.txt`，不能指向 `.npz` 目录。

```bash
ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
BUNDLE="$ROOT/docs/paper/reproducibility_bundle"
FINAL="$BUNDLE/05_final_outcome_weighted_hybrid"
RUNS="$ROOT/docs/paper/ablation_runs/outcome_weighting"
RAW_DATA="$ROOT/src/SL/data/data.txt"
BASE_MODEL="$BUNDLE/04_base_epoch17/model.pkl"

test -f "$RAW_DATA" || {
  echo "ERROR: raw data not found: $RAW_DATA" >&2
  exit 1
}
test -f "$BASE_MODEL" || {
  echo "ERROR: Base-E17 checkpoint not found: $BASE_MODEL" >&2
  exit 1
}
df -h "$ROOT"
mkdir -p "$RUNS"

prepare_variant() {
  variant="$1"
  run="$RUNS/$variant"
  mkdir -p "$run/data"
  cp "$FINAL"/agent.py "$run/"
  cp "$FINAL"/dataset.py "$run/"
  cp "$FINAL"/feature.py "$run/"
  cp "$FINAL"/model.py "$run/"
  cp "$FINAL"/preprocess.py "$run/"
  cp "$FINAL"/supervised.py "$run/"
  ln -sfn "$RAW_DATA" "$run/data/data.txt"
}

prepare_variant all
prepare_variant outcome
```

在一个 tmux 会话中串行预处理，避免两个进程同时争用 CPU、内存和磁盘。执行后即完成 G 的重新启动：

```bash
RUNS=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab/docs/paper/ablation_runs/outcome_weighting

tmux new-session -d -s mahjong_g_preprocess \
  "set -e
   cd '$RUNS/all'
   python -u preprocess.py --workers 16 --policy-weighting all \
     > preprocess.log 2>&1
   cd '$RUNS/outcome'
   python -u preprocess.py --workers 16 --policy-weighting outcome \
     > preprocess.log 2>&1"
```

### G2. 预处理结束后启动训练 / Start training after preprocessing

检查完成后再启动训练。两个训练也必须串行执行，以免显存不足：

```bash
ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
BUNDLE="$ROOT/docs/paper/reproducibility_bundle"
RUNS="$ROOT/docs/paper/ablation_runs/outcome_weighting"
BASE_MODEL="$BUNDLE/04_base_epoch17/model.pkl"

tmux ls
tail -n 20 "$RUNS/all/preprocess.log"
tail -n 20 "$RUNS/outcome/preprocess.log"
test -f "$RUNS/all/data/count.json" || exit 1
test -f "$RUNS/outcome/data/count.json" || exit 1

tmux new-session -d -s mahjong_g_train \
  "set -e
   cd '$RUNS/all'
   python -u supervised.py \
     --device cuda --amp bf16 \
     --batch-size 2048 --num-workers 8 --prefetch-factor 4 --cache-size 512 \
     --epochs 12 --seed 20261001 --no-cudnn-benchmark \
     --resume '$BASE_MODEL' \
     --action-value-loss-weight 0.20 \
     --risk-severity-opp-loss-weight 0.10 \
     --logdir model > training.log 2>&1
   cd '$RUNS/outcome'
   python -u supervised.py \
     --device cuda --amp bf16 \
     --batch-size 2048 --num-workers 8 --prefetch-factor 4 --cache-size 512 \
     --epochs 12 --seed 20261001 --no-cudnn-benchmark \
     --resume '$BASE_MODEL' \
     --action-value-loss-weight 0.20 \
     --risk-severity-opp-loss-weight 0.10 \
     --logdir model > training.log 2>&1"
```

### G3. 两个训练结束后进行配对评测 / Run paired evaluation after both training jobs

两者训练完成后，在同一批新牌墙上比较：

```bash
ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
BUNDLE="$ROOT/docs/paper/reproducibility_bundle"
FINAL="$BUNDLE/05_final_outcome_weighted_hybrid"
RUNS="$ROOT/docs/paper/ablation_runs/outcome_weighting"

test -f "$RUNS/all/model/checkpoint/best.pkl" || exit 1
test -f "$RUNS/outcome/model/checkpoint/best.pkl" || exit 1
cd "$FINAL"
mkdir -p results

tmux new-session -d -s mahjong_g_eval \
  "python -u evaluate.py \
    --opponent ../03_base_epoch16_reference/model.pkl \
    --challenger uniform_all='$RUNS/all/model/checkpoint/best.pkl' \
    --challenger outcome_weighted='$RUNS/outcome/model/checkpoint/best.pkl' \
    --risk-challenger uniform_all \
    --risk-challenger outcome_weighted \
    --postprocess light \
    --opponent-postprocess light \
    --walls 1000 \
    --seed 2026103000 \
    --bootstrap-samples 10000 \
    --output results/outcome_weighting_ablation \
    > results/outcome_weighting_ablation.log 2>&1"
```

会话 / sessions: `mahjong_g_preprocess`, `mahjong_g_train`, `mahjong_g_eval`。训练阶段只运行一次随机种子，因此论文仍需将“单训练谱系”列为限制。评测量为 8,000 局；训练成本另计。

## H. 总体风险与逐对手风险消融（待运行） / Aggregate versus opponent-specific risk ablation — pending

该实验固定最终 checkpoint、light controller、对手和牌墙，只切换控制器读取的风险头。`aggregate_risk` 使用 34 维总体风险头；`opponent_specific` 使用 3×34 逐对手风险头。由于不重新训练，它检验的是辅助风险表示对控制器的实际贡献，而不是两个 head 的独立学习能力。

This same-checkpoint comparison changes only which auxiliary risk head is consumed by the controller.

```bash
cd /root/autodl-tmp/Chinese-Standard-Mahjong-Lab/docs/paper/reproducibility_bundle/05_final_outcome_weighted_hybrid
mkdir -p results

tmux new-session -d -s mahjong_h_risk_scope \
  "python -u evaluate.py \
    --opponent ../03_base_epoch16_reference/model.pkl \
    --challenger aggregate_risk=model.pkl \
    --challenger opponent_specific=model.pkl \
    --risk-challenger aggregate_risk \
    --risk-challenger opponent_specific \
    --aggregate-risk-challenger aggregate_risk \
    --postprocess light \
    --opponent-postprocess light \
    --walls 1000 \
    --seed 2026105000 \
    --bootstrap-samples 10000 \
    --output results/risk_scope_ablation \
    > results/risk_scope_ablation.log 2>&1"
```

会话 / session: `mahjong_h_risk_scope`；日志 / log: `results/risk_scope_ablation.log`；计算量：8,000 局。由于 `aggregate_risk` 是列表中的第一个 challenger，paired 表中的差值为 `opponent_specific - aggregate_risk`。

## I. 弃牌时序输入压力测试（待运行） / Ordered-discard sequence stress test — pending

该实验固定模型和控制器，仅在 `zero_sequence` 分支把送入 GRU 的弃牌 tile 序列设为 padding、player 序列设为 0；`ordered_sequence` 保留真实有序弃牌。它能诊断模型是否实际利用时序输入，但不是完整的“移除 GRU 并重新训练”架构消融，论文中不能把结果表述成 GRU 的严格因果增益。

This is an inference-time stress test, not a retrained no-GRU architecture ablation.

```bash
cd /root/autodl-tmp/Chinese-Standard-Mahjong-Lab/docs/paper/reproducibility_bundle/05_final_outcome_weighted_hybrid
mkdir -p results

tmux new-session -d -s mahjong_i_sequence \
  "python -u evaluate.py \
    --opponent ../03_base_epoch16_reference/model.pkl \
    --challenger zero_sequence=model.pkl \
    --challenger ordered_sequence=model.pkl \
    --risk-challenger zero_sequence \
    --risk-challenger ordered_sequence \
    --zero-sequence-challenger zero_sequence \
    --postprocess light \
    --opponent-postprocess light \
    --walls 1000 \
    --seed 2026107000 \
    --bootstrap-samples 10000 \
    --output results/sequence_input_stress_test \
    > results/sequence_input_stress_test.log 2>&1"
```

会话 / session: `mahjong_i_sequence`；日志 / log: `results/sequence_input_stress_test.log`；计算量：8,000 局。paired 差值为 `ordered_sequence - zero_sequence`。

## J. 风险可靠性图（待生成） / Risk reliability diagrams — pending generation

D 已经保存 10 个可靠性分箱。以下命令从既有 JSON 画出校准前后的总体点炮风险、逐对手点炮风险和对手听牌概率；不会重新跑模型。如果缺少 Matplotlib，先执行 `python -m pip install matplotlib`。

Experiment D already stores ten reliability bins. The command below plots raw and Platt-calibrated reliability without rerunning inference.

```bash
cd /root/autodl-tmp/Chinese-Standard-Mahjong-Lab/docs/paper/reproducibility_bundle/05_final_outcome_weighted_hybrid
test -f results/final_risk_calibration_platt.json || exit 1

python - <<'PY'
import json
from pathlib import Path
import matplotlib.pyplot as plt

source = Path('results/final_risk_calibration_platt.json')
output = Path('results/final_risk_reliability.pdf')
payload = json.loads(source.read_text(encoding='utf-8'))['test']
heads = [
    ('aggregate_risk', 'Aggregate deal-in risk'),
    ('opponent_risk', 'Opponent-specific deal-in risk'),
    ('opponent_tenpai', 'Opponent tenpai'),
]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.25), constrained_layout=True)
for ax, (key, title) in zip(axes, heads):
    ax.plot([0, 1], [0, 1], '--', color='0.55', linewidth=1, label='ideal')
    for group, label, color in [
        ('raw', 'raw', '#d97706'),
        ('calibrated', 'Platt calibrated', '#0f766e'),
    ]:
        bins = [b for b in payload[group][key]['reliability'] if b['count'] > 0]
        x = [b['mean_probability'] for b in bins]
        y = [b['positive_frequency'] for b in bins]
        ax.plot(x, y, marker='o', linewidth=1.6, markersize=4,
                color=color, label=label)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted probability')
    ax.grid(alpha=0.2)
axes[0].set_ylabel('Empirical frequency')
axes[-1].legend(frameon=False, fontsize=8, loc='lower right')
fig.savefig(output, bbox_inches='tight')
print(output.resolve())
PY
```

该图支持“概率校准质量”的论述，不支持“校准控制器更强”的论述；F 已经表明校准后的控制器得分下降。现有校准 JSON 没有按早/中/晚局面或三个对手分别保存样本，因此阶段/逐对手可靠性图仍属于可选代码扩展，当前论文不应声称已经完成该分层实验。

## 完成与下载前检查 / Completion checks before download

不要仅凭 tmux 会话消失判断实验成功。对每个已运行实验，检查日志末尾是否有 traceback，并确认结果文件已经生成：

```bash
tmux ls
tail -n 40 results/fresh_final_vs_base_e17.log
tail -n 40 results/local_opponent_sensitivity.log
tail -n 40 results/final_risk_calibration_platt.log
tail -n 40 results/fresh_base_checkpoint_sweep.log
tail -n 40 results/calibrated_risk_ablation.log
tail -n 40 results/outcome_weighting_ablation.log
tail -n 40 results/risk_scope_ablation.log
tail -n 40 results/sequence_input_stress_test.log
find results -maxdepth 1 -type f -printf '%f\t%k KB\n' | sort
```

To verify completion, inspect the end of each log for a traceback and confirm that the corresponding result files exist. A missing tmux session only means that its shell command has exited; it does not by itself prove success.

## 报告顺序 / Reporting order

中文建议：

1. A 作为控制器消融。
2. B 作为新牌墙上的主要 checkpoint 比较。
3. C 必须称为本地 checkpoint 敏感性，而不是异构对手评测。
4. D 用于比较辅助风险头校准前后的预测质量。
5. F 用于决定校准是否可以进入最终控制器。
6. G 成功完成后，作为 outcome weighting 的受控训练消融。
7. H 作为相同 checkpoint 下的风险表示消融。
8. I 只能称为时序输入压力测试，不能称为重新训练后的 GRU 架构消融。
9. 现有训练曲线和旧评测只能作为开发阶段证据。

按牌墙 bootstrap 只能估计对局层面的不确定性，不能替代训练随机种子之间的变异。论文必须说明所有 checkpoint 来自同一条训练谱系，并相应限制结论。

English:

1. Report A as the controller ablation.
2. Report B as the primary fresh-wall checkpoint comparison.
3. Report C as local checkpoint sensitivity, not heterogeneous-opponent evaluation.
4. Report D as raw-versus-calibrated auxiliary-head quality.
5. Use F to decide whether calibration enters the final controller.
6. Report G as the controlled outcome-weighting training ablation if it completes successfully.
7. Report H as a same-checkpoint risk-representation ablation.
8. Report I only as a sequence-input stress test, not as a retrained GRU ablation.
9. Retain the existing training curve and archived results as development evidence.

Bootstrap over walls does not replace variation over training seeds. The paper must state that all checkpoints come from one training lineage and restrict claims accordingly.
