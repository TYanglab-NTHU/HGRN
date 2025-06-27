# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EGNN is a Graph Neural Network framework for predicting electrochemical properties (E₁/₂ redox potentials) of organometallic compounds and materials. The model uses a hierarchical approach with three levels of graph neural networks to process ligand-level, binding atom, and complex-level interactions.

## Environment Setup

```bash
# Create environment (if not already created)
conda env create -f environment.yml

# Activate environment on this HPC system
source /home/u5066474/miniconda3/etc/profile.d/conda.sh
conda activate EGNN
```

Key dependencies: PyTorch 1.13.0 + CUDA 11.7, torch-geometric 2.3.1, RDKit 2022.9.5

## Common Commands

### Training Workflows

**Pretraining (organic compounds):**
```bash
python pretrain.py --i_organic "data/organic_compounds.csv"
```

**Main training (organometallic):**
```bash
python train.py -i "data/organometallic_data.csv"
```

**Polydentate training:**
```bash
python polydentate_train.py -i "data/polydentate_data.csv"
```

**Prediction/Inference:**
```bash
python prediction.py --model_path "checkpoint/model.pkl" --input_csv "test_data.csv"
```

### Zero-Shot Applications

**Metal oxides:**
```bash
cd zero_shot/metal_oxide && python sample_MO.py
```

**MOFs:**
```bash
cd zero_shot/MOF && python sample_mof.py
```

## Architecture Overview

### Hierarchical Model Structure
The core EGNN model (`models/model.py`) implements a three-level hierarchy:

1. **GCN1**: Ligand-level message passing for individual ligands
2. **GCN2**: Binding atom interactions between ligands and metal center
3. **GCN3**: Complex-level interactions across the entire organometallic system

Each level has gate mechanisms to control information flow between subgraphs.

### Model Variants
- `OMGNN_RNN`: Main organometallic model with hierarchical processing
- `OGNN_RNN_allmask`: Pretrained organic model for transfer learning
- `polydentate_OMGNN_RNN`: Specialized for polydentate ligand complexes

### Data Processing Pipeline
- **chemutils.py**: SMILES → molecular graph conversion, 153-dimensional atom features
- **datautils.py**: Multi-level subgraph generation (ligand → binding → complex)
- **trainutils_v2.py**: Training evaluation, loss computation, visualization

## Data Format Requirements

Input CSV must contain:
- `smiles`: SMILES string of the organometallic complex
- `Metal`: Metal oxidation state (e.g., "Fe2+", "Ni0")
- `E12`: Target redox potentials (comma-separated for multiple peaks)
- `Reaction`: "reduction" or "oxidation"
- `redox_site_smiles`: SMILES of redox-active ligands
- `Solvent`: Solvent information (optional)

## Key Training Parameters

- `--num_features 153`: Atom feature dimensionality
- `--depth1/2/3`: GCN layer depths for each hierarchical level
- `--dropout 0.2`: Regularization strength
- `--pretrain True`: Enable transfer learning from organic pretraining
- `--multitask True`: Enable both regression and classification heads

## Transfer Learning Workflow

1. First pretrain on organic compounds using `pretrain.py`
2. Load pretrained GCN1 weights for organometallic training
3. Fine-tune all layers on organometallic data with `train.py`

## Model Capabilities

### Multi-task Learning
- **Regression**: Predicts E₁/₂ redox potential values (RMSE/MAE metrics)
- **Classification**: Predicts number of redox peaks per ligand (accuracy metric)

### Advanced Features
- Sequential redox site prediction for multiple electron transfer events
- Solvent effect modeling through dielectric constant features
- Support for both reduction and oxidation reactions
- Redox site tracking and assignment

## Zero-Shot Applications

The framework supports three specialized domains in `zero_shot/`:

1. **MOFs**: Requires CIF files, uses periodic boundary conditions
2. **Metal Oxides**: Solid-state electrochemistry applications
3. **Polynuclear TMCs**: Bimetallic transition metal complexes

Each domain has specialized data loaders and chemical utility functions.

## Development Notes

### Model Checkpoints
- Models saved as pickle files in `checkpoint/`
- Include both model state and configuration parameters
- Use `model.pkl` for main organometallic models

### Evaluation and Monitoring
- Training includes both regression and classification losses
- Early stopping recommended after epoch 200 based on validation loss
- Parity plots generated automatically for regression evaluation

### Extension Points
- Add new GNN backends by extending base classes in `models/`
- Implement custom atom features in `chemutils.py:atom_features()`
- Add domain-specific data processing in new `zero_shot/` subdirectories

## Working Directory Notes

- 現在全部都在EGNN-main下面做