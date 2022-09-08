#!/bin/bash
#SBATCH --job-name=ucf_finetune_tube2
#SBATCH --partition=multigpu
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:16



PROJECT_PATH="./"
DATA_PATH="../datasets/ucf101/annotations_svt"
DATASET="ucf101"

EXP_NAME="svt_masked_30_vmae_tube_2_mini"
CHECKPOINT="../pretrained/$EXP_NAME.pth"

cd "$PROJECT_PATH" || exit

python ./tools/run_net.py --cfg ./configs/UCF101/FINETUNE_divST_8x32_224.yaml \
NUM_GPUS 16 \
TRAIN.BATCH_SIZE 256 \
TEST.BATCH_SIZE 256 \
DATA.PATH_TO_DATA_DIR $DATA_PATH \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.FINETUNE True \
MODEL.TUBELET_SIZE 2 \
DATA.NUM_FRAMES 16 \
OUTPUT_DIR "./results/ucf101/finetune/$EXP_NAME"