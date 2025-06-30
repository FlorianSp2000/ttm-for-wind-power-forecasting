### Overview

Overview of file structure:

```
├── data/                           # Dataset files
├── images/                         # Generated plots and visualizations
├── results/                        # Experiment outputs and model results
├── wind_prediction_ttm.ipynb       # Main analysis notebook (EDA, ETL, Training, Evaluation)
├── app.py                          # Streamlit dashboard application
├── ttm.def                         # Singularity container definition
├── preprocess.py                   
├── feature_engineering.py          # Feature creation and transformation
├── experiment_runner.py            # Experiment execution framework
├── eval_utils.py                   
├── plot_utils.py                   
├── iterative_feature_pruning.py    # Feature selection optimization
├── analyze_permutation_results.py  # Permutation experiment analysis
├── experiment_eval.py              # Experiment results evaluation
├── channel_attention_map.py        # Attention mechanism visualization
├── feature_preselection.sh         
└── feature_pruning.sh              
```

All relevant steps (EDA, ETL, Training, Evaluation etc.) are featured in `wind_prediction_ttm.ipynb`. Some experiments, particularly regarding feature selection, were conducted through separate shell scripts. Part of the final model evaluation and its visualization can be found in the accompanying Streamlit app.

### Running the Streamlit App

To launch the interactive dashboard, run the following command in your terminal from the project directory:

```bash
streamlit run app.py
```

### Reproduce Feature Selection Results

#### Build Singularity Image

```bash
singularity build --fakeroot ttm_new.sif ttm.def
```

#### Run Feature Preselection Experiment

```bash
sbatch feature_preselection.sh
```

**Or with Singularity image:**
```bash
singularity exec --nv --pwd /ttm -B external/ttm:/ttm ttm_new.sif bash feature_preselection.sh
``` 

#### Evaluate Feature Selection Experiment

```bash
singularity exec --nv --pwd /ttm -B external/ttm:/ttm ttm_new.sif python experiment_eval.py /ttm/results/feature_selection_sweep_20250628_170430 --output_dir /ttm/results/feature_selection_sweep_20250628_170430
```

#### Start Jupyter Server with Singularity Image

```bash
singularity exec --nv -B external/ttm:/ttm ttm_new.sif jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --notebook-dir=/ttm
```

#### Run Permutation Importance Test

```bash
sbatch feature_pruning.sh
```

**Analyze results:**
```bash
# Create analysis output directory and run analyzer
singularity exec --nv --pwd /ttm -B external/ttm:/ttm ttm_new.sif python analyze_permutation_results.py results/permutation_importance_20250629_143022 --output_dir results/permutation_importance_20250629_143022/analysis
```

### Requirements

- Python 3.x
- Singularity (for containerized experiments)
- Required Python packages (see notebook for details)

### Usage Notes

- The main analysis workflow is contained in `wind_prediction_ttm.ipynb`
- For reproducible experiments, use the provided Singularity container
- Results from feature selection experiments are timestamped and stored in the `results/` directory
- The Streamlit app provides an interactive interface for model evaluation and visualization