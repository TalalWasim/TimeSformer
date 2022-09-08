#!/bin/bash
#SBATCH --job-name=linear_eval_masked
#SBATCH --partition=multigpu
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:16



PROJECT_PATH="./"
DATA_PATH="../datasets/kinetics-dataset/k400_resized_2/annotations_svt"
DATASET="kinetics"

EXP_NAME="svt_masked_30_separate_local_mae_vmae_converted.pth"
CHECKPOINT="../pretrained/$EXP_NAME"

cd "$PROJECT_PATH" || exit

python ./tools/run_net.py --cfg ./configs/Kinetics/LINEAR_divST_8x32_224_OLD.yaml \
NUM_GPUS 16 \
TRAIN.BATCH_SIZE 256 \
TEST.BATCH_SIZE 256 \
DATA.PATH_TO_DATA_DIR $DATA_PATH \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.FINETUNE True \
OUTPUT_DIR './results/svt_masked_30_separate_local_mae_vmae_linear_kinetics'