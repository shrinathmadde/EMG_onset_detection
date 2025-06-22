#!/bin/bash
#SBATCH --job-name=emg_onset
#SBATCH --output="$HOME/emg_onset_detection/logs/emg_onset_%j.out"
#SBATCH --error="$HOME/emg_onset_detection/logs/emg_onset_%j.err"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=scc-gpu
#SBATCH --gres=gpu:1

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Directory: $(pwd)"

# Load Python 3.8.18 which is available on the system
module load python/3.8.18
module load cuda/12.2.1

# Check Python version
python --version

# Install required packages using python's pip
python -m pip install --user numpy scipy matplotlib tensorflow scikit-learn seaborn pandas pywt tqdm

# Define directories
PROJECT_DIR="$HOME/emg_onset_detection"
CODE_DIR="${PROJECT_DIR}/LOL_project/shr_scripts"
DATA_DIR="${PROJECT_DIR}/LOL_project/epoched_EMG_data"
RESULTS_DIR="${PROJECT_DIR}/LOL_project/results"
DATASET_DIR="${PROJECT_DIR}/LOL_project/dataset"
TRAIN_IMG_DIR="${RESULTS_DIR}/train_img"

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${RESULTS_DIR}/${TIMESTAMP}"
mkdir -p ${RUN_DIR}
mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${DATASET_DIR}
mkdir -p ${TRAIN_IMG_DIR}

# Change to the code directory
cd ${CODE_DIR}

# Print TensorFlow version for debugging
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# Step 1: Generate simulated EMG dataset with proper background noise
echo "Starting EMG data generation..."
python generate_emg_data_v4.py \
    --output_dir ${DATASET_DIR} \
    --n_signals_per_config 8 \
    --non_negative True \
    --min_noise 0.05

# Step 2: Train the DEMANN model on the simulated dataset
echo "Starting DEMANN model training..."
python train_demann.py \
    --dataset_dir ${DATASET_DIR} \
    --output_dir ${RUN_DIR} \
    --batch_size 32 \
    --epochs 100 \
    --patience 15

# Step 3: Run predictions on real EMG data
echo "Starting predictions on real EMG data..."
python predict_with_pretrained.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${TRAIN_IMG_DIR} \
    --model_path "${RUN_DIR}/demann_model.h5" \
    --sampling_freq 2000

# Step 4: Copy important results to main results directory
echo "Copying key results to results directory..."
cp ${RUN_DIR}/training_history.png ${RESULTS_DIR}/
cp ${RUN_DIR}/evaluation_results.txt ${RESULTS_DIR}/

# Print end time
echo "End time: $(date)"