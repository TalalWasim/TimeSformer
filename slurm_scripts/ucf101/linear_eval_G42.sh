#!/bin/bash
#SBATCH --job-name=ucf_linear
#SBATCH --partition=multigpu
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:16



PROJECT_PATH="./"
DATA_PATH="../datasets/ucf101/annotations_svt"
DATASET="ucf101"

EXP_NAME="svt_masked_30_mae_vmae_mini"
CHECKPOINT="../pretrained/$EXP_NAME.pth"

cd "$PROJECT_PATH" || exit

python ./tools/run_net.py --cfg ./configs/UCF101/LINEAR_divST_8x32_224.yaml \
NUM_GPUS 16 \
TRAIN.BATCH_SIZE 256 \
TEST.BATCH_SIZE 256 \
DATA.PATH_TO_DATA_DIR $DATA_PATH \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.FINETUNE True \
OUTPUT_DIR "./results/ucf101/linear/$EXP_NAME"