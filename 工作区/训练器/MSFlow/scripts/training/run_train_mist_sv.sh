#!/usr/bin/env bash
# 用 SpectraVerse 数据集训练 MIST 编码器（支持正/负离子分开训练）
#
# 用法:
#   cd /stor1/AIMS4Meta/trainers/denovo/2_phase
#   bash scripts/training/run_train_mist_sv.sh
#
#   MODE=pos bash scripts/training/run_train_mist_sv.sh   # 正离子
#   PRETRAINED="" bash scripts/training/run_train_mist_sv.sh  # 从头训练

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ══════════════════════════════════════════════════════════════════════════════
# 训练参数（在此处修改）
# ══════════════════════════════════════════════════════════════════════════════

# ── 基础设置 ──────────────────────────────────────────────────────────────────
MODE="neg"          # 离子模式: neg | pos | both
GPU="0"             # 使用的 GPU 编号

# ── 预训练权重 ────────────────────────────────────────────────────────────────
# 微调: 填写权重路径（从正离子权重迁移到负离子）
# 从头训练: 设为空字符串 ""
PRETRAINED="/stor1/AIMS4Meta/code/MSFlow-main/checkpoints/Encoder/encoder_msg_cddd.pt"
# PRETRAINED="/stor1/AIMS4Meta/code/MSFlow-main/checkpoints/Encoder/encoder_canpous_cddd.pt"
# PRETRAINED=""   # 从头训练

# ── 训练目标 ──────────────────────────────────────────────────────────────────
# 指纹类型: morgan4096（高精度）| morgan2048（标准）| morgan1024（轻量）
FP_NAMES="morgan4096"
# 损失函数: cosine（推荐，对指纹向量更合适）| bce | mse
LOSS_FN="cosine"

# ── 模型架构 ──────────────────────────────────────────────────────────────────
# 隐藏层维度（预训练权重为 512，修改会导致权重不兼容）
HIDDEN_SIZE="512"
# Transformer 注意力层数（预训练权重为 2）
PEAK_ATTN_LAYERS="2"
# 输出精化层数（SpectraEncoderGrowing 的 growing 层数，预训练权重为 4）
REFINE_LAYERS="4"
# 注意力头数
NUM_HEADS="8"
# Dropout
SPECTRA_DROPOUT="0.1"
# 是否使用 pairwise featurization（预训练权重为 True）
PAIRWISE_FEAT="--pairwise-featurization"
# 迭代预测策略: none（标准）| growing（配合 SpectraEncoderGrowing）
ITERATIVE_PREDS="growing"
ITERATIVE_LOSS_WEIGHT="0.5"

# ── 训练超参数 ────────────────────────────────────────────────────────────────
BATCH_SIZE="64"
LR="1e-4"           # 微调建议 1e-5；从头训练建议 1e-4
WEIGHT_DECAY="0.0"
LR_DECAY_FRAC="0.995"   # 学习率衰减系数（每 decay_steps 步衰减一次）
MAX_EPOCHS="100"
PATIENCE="20"       # 早停耐心值（验证集 loss 连续 N epoch 不下降则停止）

# ── 数据增强 ──────────────────────────────────────────────────────────────────
# 是否开启峰增强（随机删除/缩放峰，提升泛化）
AUGMENT_DATA=""             # 开启: "--augment-data"；关闭: ""
AUGMENT_PROB="0.5"          # 每条谱图被增强的概率
REMOVE_PROB="0.5"           # 增强时随机删除峰的概率
INTEN_PROB="0.1"            # 增强时随机缩放峰强度的概率
REMOVE_WEIGHTS="exp"        # 删除权重策略: exp | uniform | quadratic
MAX_PEAKS=""                # 最大峰数（空=不限制，建议 50~200）

# ══════════════════════════════════════════════════════════════════════════════
# 以下内容无需修改
# ══════════════════════════════════════════════════════════════════════════════

MIST_TRAIN="$BASE_DIR/../../code/mist-main_v2/src/mist/train_mist.py"
MIST_SRC="$BASE_DIR/../../code/mist-main_v2/src"
DIFFMS_SRC="$BASE_DIR/../../code/DiffMS-master/src"

NEG_SUBFORM="/stor1/AIMS4Meta/datasets/msflow_datasets/SpectraVerse/spectraverse_neg/subformulae"
POS_SUBFORM="/stor1/AIMS4Meta/datasets/msflow_datasets/spectraverse_pos/subformulae"

SPLITS_DIR="$BASE_DIR/data/splits/$MODE"
LABELS_FILE="$SPLITS_DIR/labels.tsv"
SPLIT_FILE="$SPLITS_DIR/mist_split.tsv"
SAVE_DIR="$BASE_DIR/checkpoints/mist_sv_$MODE"

if [ "$MODE" = "both" ]; then
    SUBFORM_DIR="$BASE_DIR/data/subformulae_combined"
elif [ "$MODE" = "neg" ]; then
    SUBFORM_DIR="$NEG_SUBFORM"
else
    SUBFORM_DIR="$POS_SUBFORM"
fi

echo "=== SpectraVerse MIST 训练 [mode=$MODE] ==="

# Step 1: 数据划分
if [ ! -f "$LABELS_FILE" ] || [ ! -f "$SPLIT_FILE" ]; then
    echo "==> Step 1: 生成数据划分..."
    cd "$BASE_DIR"
    python scripts/training/split_dataset.py --mode "$MODE"
else
    echo "==> Step 1: 数据划分已存在，跳过"
fi

# Step 2: both 模式需要 symlink 合并目录
if [ "$MODE" = "both" ] && [ ! -d "$SUBFORM_DIR" ]; then
    echo "==> Step 2: 创建合并 subformulae 目录（symlink）..."
    mkdir -p "$SUBFORM_DIR"
    find "$NEG_SUBFORM" -name "*.json" -exec ln -sf {} "$SUBFORM_DIR/" \;
    find "$POS_SUBFORM" -name "*.json" -exec ln -sf {} "$SUBFORM_DIR/" \;
    echo "   链接数: $(ls "$SUBFORM_DIR" | wc -l)"
fi

# Step 3: 训练
echo "==> Step 3: 启动训练..."
echo "   mode:       $MODE"
echo "   pretrained: ${PRETRAINED:-（从头训练）}"
echo "   lr=$LR  batch=$BATCH_SIZE  epochs=$MAX_EPOCHS  fp=$FP_NAMES  loss=$LOSS_FN"

export PYTHONPATH="$MIST_SRC:$DIFFMS_SRC:$PYTHONPATH"
mkdir -p "$SAVE_DIR"

# 拼接可选参数
CKPT_ARG=""
[ -n "$PRETRAINED" ] && [ -f "$PRETRAINED" ] && CKPT_ARG="--ckpt-file $PRETRAINED"

MAX_PEAKS_ARG=""
[ -n "$MAX_PEAKS" ] && MAX_PEAKS_ARG="--max-peaks $MAX_PEAKS"

CUDA_VISIBLE_DEVICES=$GPU python "$MIST_TRAIN" \
    --labels-file        "$LABELS_FILE" \
    --subform-folder     "$SUBFORM_DIR" \
    --split-file         "$SPLIT_FILE" \
    --save-dir           "$SAVE_DIR" \
    --fp-names           $FP_NAMES \
    --loss-fn            $LOSS_FN \
    --hidden-size        $HIDDEN_SIZE \
    --peak-attn-layers   $PEAK_ATTN_LAYERS \
    --refine-layers      $REFINE_LAYERS \
    --num-heads          $NUM_HEADS \
    --spectra-dropout    $SPECTRA_DROPOUT \
    --iterative-preds    $ITERATIVE_PREDS \
    --iterative-loss-weight $ITERATIVE_LOSS_WEIGHT \
    --batch-size         $BATCH_SIZE \
    --learning-rate      $LR \
    --weight-decay       $WEIGHT_DECAY \
    --lr-decay-frac      $LR_DECAY_FRAC \
    --max-epochs         $MAX_EPOCHS \
    --patience           $PATIENCE \
    --augment-prob       $AUGMENT_PROB \
    --remove-prob        $REMOVE_PROB \
    --inten-prob         $INTEN_PROB \
    --remove-weights     $REMOVE_WEIGHTS \
    --cache-featurizers \
    --scheduler \
    $PAIRWISE_FEAT \
    $AUGMENT_DATA \
    $MAX_PEAKS_ARG \
    $CKPT_ARG
