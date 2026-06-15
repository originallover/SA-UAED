#!/usr/bin/env bash

# export KALDI_ROOT=/path/to/kaldi
# export LSDIR=/path/to/LibriSpeech

set -e

# ============ 配置参数 ============
# KALDI_ROOT=${KALDI_ROOT:-/path/to/kaldi}
# export LSDIR="/data/lib/librispeech/data"
#输出目录
OUTPUT_DIR="./conversation_data"
#DESED sound原始数据路径
DESED_DIR="./conversation_data/synthetic/audio/eval/soundbank"
#声音事件索引文件路径
SOUND_events_dir="${OUTPUT_DIR}/sound_events_index.json"

# 数据集配置 这是默认设置值，也可自定义
TRAIN_HOURS=500        # 训练集时长（小时）
EVAL_HOURS=5           # 评估集时长（小时）
TEST_HOURS=5
CONV_DURATION=32.0     # 每个对话时长（秒）
NUM_SPEAKERS=3         # 说话人数量
SAMPLING_RATE=16000    # 采样率

FRAME_RATE=50 # 帧率

# 计算对话数量
TRAIN_CONVS=$((TRAIN_HOURS * 3600 / 32))  
EVAL_CONVS=$((EVAL_HOURS * 3600 / 32))    
TEST_CONVS=$((TEST_HOURS * 3600 / 32))

echo "=========================================="
echo "UAED Conversation Data Generation"
echo "=========================================="
echo "Training conversations: $TRAIN_CONVS"
echo "Evaluation conversations: $EVAL_CONVS"
echo "Test conversations: $TEST_CONVS"
echo "=========================================="

#=============准备声音事件索引=====================

# echo "Preparing sound events index..."
# python prepare_sound_events.py\
#     --soundbank-dir $DESED_DIR\
#     --output-file $SOUND_events_dir


# ============ 步骤 1: 准备 Librispeech 数据 ，将每个说话人的每个章节的音频合为一段长音频============
# echo ""
# echo "Step 1: Preparing Librispeech data..."
# #需根据实际路径调整脚本里的目录（较为复杂）
# bash prepareKaldidata_LibriSpeech.sh

# ============ 步骤 2: 生成对话定义文件 ============
echo ""
echo "Step 2: Generating conversation definitions..."

# 训练集
python generate_conversation_definitions.py \
    --kaldi-data-dir ./Kaldidatadir/alltrain_persession \
    --output-dir $OUTPUT_DIR/train/conversations \
    --sound-events-dir $SOUND_events_dir \
    --num-conversations $TRAIN_CONVS \
    --duration $CONV_DURATION \
    --num-speakers $NUM_SPEAKERS \
    --sampling-rate $SAMPLING_RATE \
    --seed 42


# 评估集
python generate_conversation_definitions.py \
    --kaldi-data-dir ./Kaldidatadir/dev_clean_persession \
    --output-dir $OUTPUT_DIR/eval/conversations \
    --sound-events-dir $SOUND_events_dir \
    --num-conversations $EVAL_CONVS \
    --duration $CONV_DURATION \
    --num-speakers $NUM_SPEAKERS \
    --sampling-rate $SAMPLING_RATE \
    --seed 123

#测试集
python generate_conversation_definitions.py \
    --kaldi-data-dir ./Kaldidatadir/dev_clean_persession \
    --output-dir $OUTPUT_DIR/test/conversations \
    --sound-events-dir $SOUND_events_dir \
    --num-conversations $TEST_CONVS \
    --duration $CONV_DURATION \
    --num-speakers $NUM_SPEAKERS \
    --sampling-rate $SAMPLING_RATE \
    --seed 42

# ============ 步骤 3: 合成对话音频 ============
echo ""
echo "Step 3: Synthesizing conversation audio..."

# 训练集
python synthesize_conversations.py \
    --conv-list $OUTPUT_DIR/train/conversations/conversations.list \
    --conv-dir $OUTPUT_DIR/train/conversations \
    --wav-scp ./Kaldidatadir/alltrain_persession/wav.scp \
    --output-dir $OUTPUT_DIR/train/audio \
    --sampling-rate $SAMPLING_RATE \
    --num-workers 8

# 评估集
python synthesize_conversations.py \
    --conv-list $OUTPUT_DIR/eval/conversations/conversations.list \
    --conv-dir $OUTPUT_DIR/eval/conversations \
    --wav-scp ./Kaldidatadir/dev_clean_persession/wav.scp \
    --output-dir $OUTPUT_DIR/eval/audio \
    --sampling-rate $SAMPLING_RATE \
    --num-workers 8

# 测试集
python synthesize_conversations.py \
    --conv-list $OUTPUT_DIR/test/conversations/conversations.list \
    --conv-dir $OUTPUT_DIR/test/conversations \
    --wav-scp ./Kaldidatadir/dev_clean_persession/wav.scp \
    --output-dir $OUTPUT_DIR/test/audio \
    --sampling-rate $SAMPLING_RATE \
    --num-workers 8

# ============ 步骤 4: 生成标签 ============
echo ""
echo "Step 4: Generating labels..."

for subset in train eval; do
    python generate_conversation_labels.py \
        --conv-list $OUTPUT_DIR/$subset/conversations/conversations.list \
        --conv-dir $OUTPUT_DIR/$subset/conversations \
        --output-dir $OUTPUT_DIR/$subset/labels \
        --duration $CONV_DURATION \
        --frame-rate $FRAME_RATE
done

python generate_conversation_labels.py \
        --conv-list $OUTPUT_DIR/test/conversations/conversations.list \
        --conv-dir $OUTPUT_DIR/test/conversations \
        --output-dir $OUTPUT_DIR/test/labels \
        --duration $CONV_DURATION \
        --frame-rate $FRAME_RATE

echo ""
echo "=========================================="
echo "Data generation complete!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
