#!/bin/bash
#SBATCH --job-name=feature_selection
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=L40Sday
#SBATCH --mail-type=ALL
#  --gres=gpu:1
#  --partition=day

# =============================================================================
# WIND POWER FEATURE SELECTION HYPERPARAMETER SWEEP
# =============================================================================
# This script runs systematic feature selection experiments for wind power forecasting
# Focus: Finding optimal feature combinations before model comparison
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

echo "Starting Wind Power Feature Selection Sweep"
echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"  
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Create logs directory if it doesn't exist
mkdir -p logs

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/feature_selection_sweep_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo "Results will be saved to: ${RESULTS_DIR}"

# =============================================================================
# SINGULARITY SETUP
# =============================================================================

# Set paths for singularity
SINGULARITY_IMAGE="ttm_new.sif"
TTM_PATH="/ttm"

# Function to run singularity commands
run_singularity() {
    singularity exec --nv --pwd ${TTM_PATH} -B external/ttm:${TTM_PATH} ${SINGULARITY_IMAGE} "$@"
}

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Base configuration (fixed across experiments)
MODEL_TYPE="finetuned"  # Use zero-shot for fast feature selection
EVAL_SPLIT="valid"        # Use validation set for feature selection
NUM_EPOCHS=50            # Quick training if needed
BATCH_SIZE=64
CONTEXT_LENGTH=512
PREDICTION_LENGTH=24

# Spatial configurations to test
SPATIAL_CONFIGS=("simple" "coastal_inland" "latitude_bands")

# Base weather features (core meteorological variables)
BASE_FEATURES_MINIMAL="u100,v100,u10,v10"
BASE_FEATURES_WIND_TEMP="u100,v100,u10,v10,t2m"
BASE_FEATURES_WIND_PRESSURE="u100,v100,u10,v10,msl"
BASE_FEATURES_WIND_FULL="u100,v100,u10,v10,t2m,msl,cdir"
BASE_FEATURES_ALL="u100,v100,u10,v10,msl,t2m,cdir,blh,tcc,tp"

# Feature combinations to test
declare -A FEATURE_CONFIGS=(
    ["minimal_wind"]="${BASE_FEATURES_MINIMAL}"
    ["wind_plus_temp"]="${BASE_FEATURES_WIND_TEMP}"
    ["wind_plus_pressure"]="${BASE_FEATURES_WIND_PRESSURE}"
    ["wind_extended"]="${BASE_FEATURES_WIND_FULL}"
    ["all_weather"]="${BASE_FEATURES_ALL}"
)

# Derived features options (wind speed/direction engineering)
DERIVED_OPTIONS=("true" "false")

# Temporal features options (hour, month, seasonal)
TEMPORAL_OPTIONS=("true" "false")

# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

EXPERIMENT_COUNT=0
TOTAL_EXPERIMENTS=$((${#SPATIAL_CONFIGS[@]} * ${#FEATURE_CONFIGS[@]} * ${#DERIVED_OPTIONS[@]} * ${#TEMPORAL_OPTIONS[@]}))

echo "Total experiments planned: ${TOTAL_EXPERIMENTS}"
echo ""

# Iterate through all combinations
for spatial_config in "${SPATIAL_CONFIGS[@]}"; do
    for feature_name in "${!FEATURE_CONFIGS[@]}"; do
        base_features="${FEATURE_CONFIGS[$feature_name]}"
        
        for derived in "${DERIVED_OPTIONS[@]}"; do
            for temporal in "${TEMPORAL_OPTIONS[@]}"; do
                
                EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
                
                # Create experiment name
                #EXP_NAME="exp_${EXPERIMENT_COUNT}_${spatial_config}_${feature_name}_d${derived}_t${temporal}"
                EXP_DIR="${RESULTS_DIR}"
                
                echo "Experiment ${EXPERIMENT_COUNT}/${TOTAL_EXPERIMENTS}: ${EXP_NAME}"
                echo "   Spatial: ${spatial_config}"
                echo "   Features: ${feature_name} (${base_features})"
                echo "   Derived: ${derived}, Temporal: ${temporal}"
                
                # Run experiment via singularity
                run_singularity python experiment_runner.py \
                    --spatial_config "${spatial_config}" \
                    --model_type "${MODEL_TYPE}" \
                    --eval_split "${EVAL_SPLIT}" \
                    --base_features "${base_features}" \
                    --derived_features "${derived}" \
                    --temporal_features "${temporal}" \
                    --lag_features "false" \
                    --context_length "${CONTEXT_LENGTH}" \
                    --prediction_length "${PREDICTION_LENGTH}" \
                    --num_epochs "${NUM_EPOCHS}" \
                    --batch_size "${BATCH_SIZE}" \
                    --out_dir "${EXP_DIR}" \
                    --dataset_path "/ttm/data/Realised_Supply_Germany.csv" \
                    --weather_path "/ttm/data/Weather_Data_Germany.csv" \
                    --weather_path_2 "/ttm/data/Weather_Data_Germany_2022.csv"
                
                if [ $? -eq 0 ]; then
                    echo "   Success"
                else
                    echo "   Failed"
                fi
                echo ""
                
            done
        done
    done
done