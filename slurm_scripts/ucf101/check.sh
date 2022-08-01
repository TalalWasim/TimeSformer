#!/bin/bash
#SBATCH --job-name=linear_eval_check
#SBATCH --partition=default-long
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1



PROJECT_PATH="./"
DATA_PATH="../datasets/ucf101/annotations_svt"
DATASET="ucf101"

EXP_NAME="svt_videomae_30_slice_1_converted.pth"
CHECKPOINT="../pretrained/$EXP_NAME"

cd "$PROJECT_PATH" || exit

python ./tools/run_net.py --cfg ./configs/UCF101/LINEAR_divST_8x32_224.yaml \
NUM_GPUS 1 \
TRAIN.BATCH_SIZE 16 \
TEST.BATCH_SIZE 16 \
DATA.PATH_TO_DATA_DIR $DATA_PATH \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.FINETUNE True \
OUTPUT_DIR './results/check_linear_ucf101'

