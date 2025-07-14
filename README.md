# HGRN
![Figure abstract]

## About
**HGRN** 


This repository contains the code we used in training of HGRN as described in the manuscript. 

## Installation
```sh
git clone https://github.com/TYanglab-NTHU/HGRN
cd EGNN
```

```sh
conda env create -f environment.yaml
conda activate HGRN
```

HGRN has been tested on PyTorch version 1.13.0 and CUDA version 11.7.
We highly recommend users to use GPU for accelerating ligand generation

Note for machines with GPUs: You may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/). If you're encountering issues with LiveTransForm not using a GPU on your system after following the instructions below, check which version of PyTorch you have installed in your environment using conda list | grep torch or similar. If the PyTorch line includes cpu, please uninstall it using conda remove pytorch and reinstall a GPU-enabled version using the instructions at the link above.

## Usage

## Pretrain Organic Compounds for E1/2 prediction

```
python pretrain.py --i_organic "<organic_compound.csv>"
```

## Training Organometallic Compounds for E1/2 prediction
```
python train.py -i "<1st_TMCs_E12.csv>"
```

## Zero shot for E1/2 prediction
Before predicting E₁/₂ for metal oxides and MOFs, we preprocess their CIF files using the **Chic** (https://github.com/tcnicholas/chic) and **pymatgen** libraries.  
Make sure both packages are installed and that your structure files can be parsed by them:
### Polynuclear TMCs
```
cd zero_shot/polynuclear_tmc
python sample_polynuclear.py
```
### Metal Oxide 
```
cd zero_shot/metal_oxide
python sample_MO.py
```
### MOF 
```
cd zero_shot/MOF
python sample_mof.py
```
## Train Sequential values prediction for organic compounds 
Dataset should be a two-column file (CSV, TSV, etc.) with:

| **SMILES**         | **Targets**            |
| ------------------ | ---------------------- |
| `CCO`              | `"0.12,0.34,0.56"`       |
| `c1ccccc1O`        | `"1.23,4.56,7.89"`       |
| `CC(N)C(=O)O`      | `"2.5,3.0"`              |

- **SMILES**: the molecular string.
- **Targets**: one or more numerical values separated by commas, e.g.  
  ```text
  "val1,val2,val3,..."

```
cd HGRN-main/
python organic_train_self.py  --i_organic <organic_data.csv> --label_column <target_column>
```
## Train Sequential values prediction for TMCs compounds 
Dataset should be a three or two-column file (CSV, TSV, etc.) with:

| **SMILES**         | **Targets**              | **site**
| ------------------ | ----------------------   | --------------------
|                    | `"0.12,0.34     "`       | "[Fe+2]/C1=CNC(=C2C=CC=CN2)C=C1"
|                    | `"1.23,4.56     "`       | "[Co+2]/[Co+2]"

- **SMILES**: the molecular string.
- **Targets**: one or more numerical values separated by commas, e.g.  
  ```text
  "val1,val2,val3,..."
- **site**: one or more str separated by slash, e.g.  
  ```text
  "site1_SMILE/site2_SMILE/site3_SMILE,..."
However, if you cannot obtain the 'site' information, it cannot be used; moreover, HGRN applies updates uniformly across all nodes rather than targeting a specific site, which leads to worse performance compared to training with site-specific updates.

### with 'site'
```
python complex_train_self.py  -i <TMCs_data.csv> --label_column <target_column>
```
### without 'site'
```
python complex_train_self.py  -i <TMCs_data.csv> --label_column <target_column> --global_graph True
```

## License
License details can be found in the LICENSE file.
# Requirements
* RDKit (version >= 2022.09.5)
* Python (version >= 3.7.13)
* PyTorch (version >= 1.13.0)
* Openbabel (version >= 3.1.0)
