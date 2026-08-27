# OWBC controlled retraining guide / 结果加权控制重训指南

This optional experiment is intended for a reviewer request. It is not required to reproduce the currently reported paper results. The primary comparison is `uniform_all` versus `outcome_weighted`; an optional `winner_only` arm is described at the end.

本实验只在审稿人要求因果消融时运行。论文现有结果不依赖它。主要比较为 `uniform_all` 与 `outcome_weighted`；文末另给出可选的 `winner_only` 第三组。

## 1. Causal question and frozen design / 因果问题与冻结设计

Both primary arms must use:

- the same Base-E17 initialization;
- the same 130-tensor training architecture, including both auxiliary heads;
- the same raw records and 90/10 indexing split;
- the same seed, optimizer, learning-rate schedule, batch size, epochs, augmentation, and auxiliary-loss coefficients;
- the same frozen inference controller, opponents, walls, seats, and bootstrap seed.

The only intended difference is `--policy-weighting all` versus `--policy-weighting outcome` during preprocessing. Do not tune either model on the confirmatory wall range. One seed supports only a **single-seed controlled ablation**, not an across-seed stability claim.

两组只能改变预处理时的策略权重。不能为某一组单独调整学习率、epoch、控制器或测试牌墙。单个随机种子的结果只能写成单种子控制消融。

## 2. Required local files / 本地必需文件

The following files are intentionally excluded from Git and must be transferred separately:

```text
src/SL/data/data.txt
docs/paper/reproducibility_bundle/03_base_epoch16_reference/model.pkl
docs/paper/reproducibility_bundle/04_base_epoch17/model.pkl
```

Base-E17 initializes both training arms. Base-E16 is used only as the frozen evaluation opponent. Before uploading, run the following from **local Git Bash**, not PowerShell:

```bash
LOCAL_ROOT=/d/PKU/CODE/26spring-ai/Homework/Chinese-Standard-Mahjong-Lab

test -f "$LOCAL_ROOT/src/SL/data/data.txt" || exit 1
test -f "$LOCAL_ROOT/docs/paper/reproducibility_bundle/03_base_epoch16_reference/model.pkl" || exit 1
test -f "$LOCAL_ROOT/docs/paper/reproducibility_bundle/04_base_epoch17/model.pkl" || exit 1

sha256sum "$LOCAL_ROOT/src/SL/data/data.txt"
sha256sum "$LOCAL_ROOT/docs/paper/reproducibility_bundle/03_base_epoch16_reference/model.pkl"
sha256sum "$LOCAL_ROOT/docs/paper/reproducibility_bundle/04_base_epoch17/model.pkl"
```

The raw-data hash expected by the manuscript is:

```text
2a95ba17a058fe3fbf2c240abbd3e05a88ee95a215cf8cb6c00ea4ff5193441b
```

Checkpoint hashes must match the corresponding `manifest.json` files.

## 3. Upload to AutoDL / 上传到 AutoDL

Set the current SSH endpoint in local Git Bash. Replace the example host and port whenever AutoDL creates a new instance:

```bash
PORT=45040
HOST=connect.westd.seetacloud.com
REMOTE="root@$HOST"
LOCAL_ROOT=/d/PKU/CODE/26spring-ai/Homework/Chinese-Standard-Mahjong-Lab
REMOTE_ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
```

If the remote repository was cloned after the reproducibility bundle was committed, update the tracked source first:

```bash
ssh -p "$PORT" "$REMOTE" \
  "cd '$REMOTE_ROOT' && git pull --ff-only"
```

Otherwise, create a clean archive from local Git Bash. This excludes local-only checkpoints, logs, PID files, and smoke-test artifacts instead of copying the entire working directory:

```bash
ARCHIVE=/tmp/mahjong_reproducibility_bundle.tar.gz

tar -C "$LOCAL_ROOT/docs/paper" \
  --exclude='*/model.pkl' \
  --exclude='*/results/*.log' \
  --exclude='*.pid' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*smoke*' \
  -czf "$ARCHIVE" reproducibility_bundle

ssh -p "$PORT" "$REMOTE" \
  "mkdir -p '$REMOTE_ROOT/docs/paper' /tmp/mahjong_bundle_upload"

scp -P "$PORT" "$ARCHIVE" \
  "$REMOTE:/tmp/mahjong_bundle_upload/reproducibility_bundle.tar.gz"

ssh -p "$PORT" "$REMOTE" \
  "tar -C '$REMOTE_ROOT/docs/paper' -xzf /tmp/mahjong_bundle_upload/reproducibility_bundle.tar.gz"
```

Upload the ignored data and the two required checkpoints separately:

```bash
ssh -p "$PORT" "$REMOTE" \
  "mkdir -p \
    '$REMOTE_ROOT/src/SL/data' \
    '$REMOTE_ROOT/docs/paper/reproducibility_bundle/03_base_epoch16_reference' \
    '$REMOTE_ROOT/docs/paper/reproducibility_bundle/04_base_epoch17'"

scp -P "$PORT" \
  "$LOCAL_ROOT/src/SL/data/data.txt" \
  "$REMOTE:$REMOTE_ROOT/src/SL/data/data.txt"

scp -P "$PORT" \
  "$LOCAL_ROOT/docs/paper/reproducibility_bundle/03_base_epoch16_reference/model.pkl" \
  "$REMOTE:$REMOTE_ROOT/docs/paper/reproducibility_bundle/03_base_epoch16_reference/model.pkl"

scp -P "$PORT" \
  "$LOCAL_ROOT/docs/paper/reproducibility_bundle/04_base_epoch17/model.pkl" \
  "$REMOTE:$REMOTE_ROOT/docs/paper/reproducibility_bundle/04_base_epoch17/model.pkl"
```

## 4. Remote verification and environment record / 远端校验与环境记录

Log in and define stable paths:

```bash
ssh -p "$PORT" "$REMOTE"

ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
BUNDLE="$ROOT/docs/paper/reproducibility_bundle"
FINAL="$BUNDLE/05_final_outcome_weighted_hybrid"
RUNS="$ROOT/docs/paper/ablation_runs/outcome_weighting"
RAW_DATA="$ROOT/src/SL/data/data.txt"
BASE16="$BUNDLE/03_base_epoch16_reference/model.pkl"
BASE17="$BUNDLE/04_base_epoch17/model.pkl"

test -f "$RAW_DATA" || exit 1
test -f "$BASE16" || exit 1
test -f "$BASE17" || exit 1
test -f "$FINAL/supervised.py" || exit 1
test -f "$FINAL/preprocess.py" || exit 1

sha256sum "$RAW_DATA" "$BASE16" "$BASE17"
python "$FINAL/preprocess.py" -h | grep -- '--policy-weighting'
python "$FINAL/supervised.py" -h | grep -E -- '--seed|--no-cudnn-benchmark'
nvidia-smi
df -h /root/autodl-tmp
```

Create a provenance record before training:

```bash
mkdir -p "$RUNS"
{
  date -Is
  python --version
  python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  sha256sum "$RAW_DATA" "$BASE16" "$BASE17"
  git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo 'uncommitted transfer snapshot'
} > "$RUNS/provenance.txt"
```

## 5. Prepare isolated variants / 准备隔离目录

Each arm receives the same source snapshot and a symlink to the same raw archive. Generated arrays remain isolated and can consume substantial disk space.

```bash
set -e
ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
BUNDLE="$ROOT/docs/paper/reproducibility_bundle"
FINAL="$BUNDLE/05_final_outcome_weighted_hybrid"
RUNS="$ROOT/docs/paper/ablation_runs/outcome_weighting"
RAW_DATA="$ROOT/src/SL/data/data.txt"
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

prepare_variant uniform_all
prepare_variant outcome_weighted
```

Do not continue unless the data disk has space for two generated datasets plus two checkpoint directories:

```bash
df -h /root/autodl-tmp
du -sh "$ROOT/src/SL/data" 2>/dev/null || true
```

## 6. Preprocess in tmux / 后台预处理

Run the two jobs serially to avoid CPU, memory, and disk contention:

```bash
ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
RUNS="$ROOT/docs/paper/ablation_runs/outcome_weighting"

tmux new-session -d -s owbc_preprocess \
  "set -e
   cd '$RUNS/uniform_all'
   python -u preprocess.py --workers 16 --policy-weighting all \
     > preprocess.log 2>&1
   cd '$RUNS/outcome_weighted'
   python -u preprocess.py --workers 16 --policy-weighting outcome \
     > preprocess.log 2>&1"
```

Monitor without stopping the jobs:

```bash
tmux ls
tail -F "$RUNS/uniform_all/preprocess.log"
tail -F "$RUNS/outcome_weighted/preprocess.log"
pgrep -af 'python.*preprocess.py'
df -h /root/autodl-tmp
```

After completion:

```bash
test -f "$RUNS/uniform_all/data/count.json" || exit 1
test -f "$RUNS/outcome_weighted/data/count.json" || exit 1
tail -n 20 "$RUNS/uniform_all/preprocess.log"
tail -n 20 "$RUNS/outcome_weighted/preprocess.log"
```

## 7. Train both arms with one frozen seed / 固定种子训练

The commands are intentionally identical except for the preprocessed working directory. Training is serial to avoid GPU contention.

```bash
ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
BUNDLE="$ROOT/docs/paper/reproducibility_bundle"
RUNS="$ROOT/docs/paper/ablation_runs/outcome_weighting"
BASE17="$BUNDLE/04_base_epoch17/model.pkl"

tmux new-session -d -s owbc_train \
  "set -e
   cd '$RUNS/uniform_all'
   python -u supervised.py \
     --device cuda --amp bf16 \
     --batch-size 2048 --num-workers 8 --prefetch-factor 4 --cache-size 512 \
     --epochs 12 --seed 20261001 --no-cudnn-benchmark \
     --resume '$BASE17' \
     --action-value-loss-weight 0.20 \
     --risk-severity-opp-loss-weight 0.10 \
     --logdir model > training.log 2>&1
   cd '$RUNS/outcome_weighted'
   python -u supervised.py \
     --device cuda --amp bf16 \
     --batch-size 2048 --num-workers 8 --prefetch-factor 4 --cache-size 512 \
     --epochs 12 --seed 20261001 --no-cudnn-benchmark \
     --resume '$BASE17' \
     --action-value-loss-weight 0.20 \
     --risk-severity-opp-loss-weight 0.10 \
     --logdir model > training.log 2>&1"
```

Useful monitoring commands:

```bash
tmux attach -t owbc_train
tail -F "$RUNS/uniform_all/training.log"
tail -F "$RUNS/outcome_weighted/training.log"
pgrep -af 'python.*supervised.py'
watch -n 2 nvidia-smi
```

Detach from tmux with `Ctrl+b`, then `d`. Exit `tail` or `watch` with `Ctrl+C`; this does not stop training.

## 8. Freeze and hash checkpoints / 冻结与哈希

After both jobs finish:

```bash
ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
RUNS="$ROOT/docs/paper/ablation_runs/outcome_weighting"
UNIFORM_MODEL="$RUNS/uniform_all/model/checkpoint/best.pkl"
OUTCOME_MODEL="$RUNS/outcome_weighted/model/checkpoint/best.pkl"

test -f "$UNIFORM_MODEL" || exit 1
test -f "$OUTCOME_MODEL" || exit 1

(
  cd "$(dirname "$UNIFORM_MODEL")"
  sha256sum best.pkl
) > "$RUNS/uniform_all/model.sha256"
(
  cd "$(dirname "$OUTCOME_MODEL")"
  sha256sum best.pkl
) > "$RUNS/outcome_weighted/model.sha256"

tail -n 40 "$RUNS/uniform_all/training.log"
tail -n 40 "$RUNS/outcome_weighted/training.log"
```

Do not rename one arm to `Final-Hybrid`. These are new controlled-ablation models and must retain their own labels.

## 9. Confirmatory paired evaluation / 确认性配对评测

Use `uniform_all` as the first challenger so it becomes the paired baseline:

```bash
ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
BUNDLE="$ROOT/docs/paper/reproducibility_bundle"
FINAL="$BUNDLE/05_final_outcome_weighted_hybrid"
RUNS="$ROOT/docs/paper/ablation_runs/outcome_weighting"
BASE16="$BUNDLE/03_base_epoch16_reference/model.pkl"
UNIFORM_MODEL="$RUNS/uniform_all/model/checkpoint/best.pkl"
OUTCOME_MODEL="$RUNS/outcome_weighted/model/checkpoint/best.pkl"

cd "$FINAL"
mkdir -p results

tmux new-session -d -s owbc_eval \
  "python -u evaluate.py \
    --opponent '$BASE16' \
    --challenger uniform_all='$UNIFORM_MODEL' \
    --challenger outcome_weighted='$OUTCOME_MODEL' \
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

This evaluates 2 models × 1000 walls × 4 seats = 8000 games. Inspect:

```bash
tail -F "$FINAL/results/outcome_weighting_ablation.log"
cat "$FINAL/results/outcome_weighting_ablation_summary.csv"
cat "$FINAL/results/outcome_weighting_ablation_paired.json"
```

Report the paired average-score difference and wall-clustered 95% interval as primary. Win rate, deal-in rate, severity, rank, and invalid rate are secondary. Do not call OWBC significantly better if the interval crosses zero.

## 10. Package only necessary outputs / 打包必要结果

The export contains the two best checkpoints, logs, source snapshots, hashes, provenance, and evaluator outputs, but excludes generated `.npz` data and intermediate epoch checkpoints.

```bash
ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
BUNDLE="$ROOT/docs/paper/reproducibility_bundle"
FINAL="$BUNDLE/05_final_outcome_weighted_hybrid"
RUNS="$ROOT/docs/paper/ablation_runs/outcome_weighting"
EXPORT="$RUNS/export"
rm -rf "$EXPORT"
mkdir -p "$EXPORT/uniform_all" "$EXPORT/outcome_weighted" "$EXPORT/evaluation"

for variant in uniform_all outcome_weighted; do
  cp "$RUNS/$variant/model/checkpoint/best.pkl" "$EXPORT/$variant/best.pkl"
  cp "$RUNS/$variant/model.sha256" "$EXPORT/$variant/"
  cp "$RUNS/$variant/data/count.json" "$EXPORT/$variant/data_count.json"
  cp "$RUNS/$variant/preprocess.log" "$EXPORT/$variant/"
  cp "$RUNS/$variant/training.log" "$EXPORT/$variant/"
  cp "$RUNS/$variant"/{agent.py,dataset.py,feature.py,model.py,preprocess.py,supervised.py} \
    "$EXPORT/$variant/"
done

cp "$RUNS/provenance.txt" "$EXPORT/"
cp "$FINAL"/results/outcome_weighting_ablation_* "$EXPORT/evaluation/"

ARCHIVE_DIR="$RUNS/download"
ARCHIVE_NAME=owbc_controlled_retraining_20261001.tar.gz
mkdir -p "$ARCHIVE_DIR"
tar -C "$RUNS" -czf "$ARCHIVE_DIR/$ARCHIVE_NAME" export
(
  cd "$ARCHIVE_DIR"
  sha256sum "$ARCHIVE_NAME" > "$ARCHIVE_NAME.sha256"
)
ls -lh "$ARCHIVE_DIR/$ARCHIVE_NAME" "$ARCHIVE_DIR/$ARCHIVE_NAME.sha256"
```

## 11. Download and verify locally / 下载并校验

Run from local Git Bash using the same `PORT`, `REMOTE`, and `LOCAL_ROOT` variables from Section 3:

```bash
PORT=45040
HOST=connect.westd.seetacloud.com
REMOTE="root@$HOST"
LOCAL_ROOT=/d/PKU/CODE/26spring-ai/Homework/Chinese-Standard-Mahjong-Lab
REMOTE_ROOT=/root/autodl-tmp/Chinese-Standard-Mahjong-Lab
LOCAL_OUT="$LOCAL_ROOT/docs/paper/ablation_runs/outcome_weighting_download"
REMOTE_ARCHIVE="$REMOTE_ROOT/docs/paper/ablation_runs/outcome_weighting/download"
ARCHIVE_NAME=owbc_controlled_retraining_20261001.tar.gz

mkdir -p "$LOCAL_OUT"
scp -P "$PORT" \
  "$REMOTE:$REMOTE_ARCHIVE/$ARCHIVE_NAME" \
  "$REMOTE:$REMOTE_ARCHIVE/$ARCHIVE_NAME.sha256" \
  "$LOCAL_OUT/"

cd "$LOCAL_OUT"
sha256sum -c "$ARCHIVE_NAME.sha256"
tar -xzf "$ARCHIVE_NAME"
(cd export/uniform_all && sha256sum -c model.sha256)
(cd export/outcome_weighted && sha256sum -c model.sha256)
```

`docs/paper/ablation_runs/` remains private because the parent paper directory is ignored. Copy only reviewed CSV/JSON evidence into the public reproducibility bundle if the experiment is added to the manuscript.

## 12. Optional winner-only arm / 可选 winner-only 第三组

If a reviewer explicitly requests the three-way comparison, repeat Sections 5--8 with a `winner_only` directory and:

```bash
python -u preprocess.py --workers 16 --policy-weighting winner \
  > preprocess.log 2>&1
```

Train it with the exact command and seed used by the other two arms. Add it as a third challenger in Section 9. Because winner-only changes which players contribute samples, report its number of training samples alongside the two all-player variants. The primary causal contrast for outcome weighting should remain `outcome_weighted - uniform_all`.

## 13. Reporting checklist / 论文报告清单

Archive and report:

- source-data, initialization-checkpoint, and trained-checkpoint SHA-256 hashes;
- repository commit, Python/PyTorch/CUDA versions, GPU, seed, and complete commands;
- sample counts for every arm;
- validation curves and selected-checkpoint rule;
- fresh wall range, opponent hash, four-seat rotation, excluded-wall counts, and bootstrap seed;
- paired average-score difference with 95% wall-cluster interval;
- the explicit limitation that only one training seed was run.
