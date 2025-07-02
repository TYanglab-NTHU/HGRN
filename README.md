# EGNN
![Figure abstract]

## About
**EGNN** 


This repository contains the code we used in training of EGNN as described in the manuscript. 

## Installation
```sh
git clone https://github.com/TYanglab-NTHU/EGNN
cd EGNN
```

```sh
conda env create -f environment.yml
conda activate EGNN
```

EGNN has been tested on PyTorch version 1.13.0 and CUDA version 11.7.
We highly recommend users to use GPU for accelerating ligand generation

Note for machines with GPUs: You may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/). If you're encountering issues with LiveTransForm not using a GPU on your system after following the instructions below, check which version of PyTorch you have installed in your environment using conda list | grep torch or similar. If the PyTorch line includes cpu, please uninstall it using conda remove pytorch and reinstall a GPU-enabled version using the instructions at the link above.

## Usage

## Pretrain Organic Compounds for E1/2 prediction

```
python pretrain.py --i_organic "<organic_compound.csv>"
```

## Training Organometallic Compounds for E1/2 prediction
```
python train.py -i "<organo_compound.csv>"
```

## Zero shot for E1/2 prediction

Before predicting E₁/₂ for metal oxides and MOFs, we preprocess their CIF files using the **Chic** (https://github.com/tcnicholas/chic) and **pymatgen** libraries.  
Make sure both packages are installed and that your structure files can be parsed by them:
### Metal Oxide 
```
python sample_MO.py
```
### MOF 
```
python sample_mof.py
```
## Train Sequential values prediction for organic compounds 
Input data format one column for SMILE and one column for target values and use "val1,val2,val3..." to depart each value.
```
python organic_train_self.py  --label_column <target_column>
```

## License
License details can be found in the LICENSE file.
# Requirements
* RDKit (version >= 2022.09.5)
* Python (version >= 3.7.13)
* PyTorch (version >= 1.13.0)
* Openbabel (version >= 3.1.0)